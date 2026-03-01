import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "n/a", "na", "null"}:
        return ""
    return " ".join(text.split())


def parse_numeric_token(value: object) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None

    time_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([smhd])", text)
    if time_match:
        amount = float(time_match.group(1))
        unit = time_match.group(2)
        if unit == "s":
            return amount / 3600.0
        if unit == "m":
            return amount / 60.0
        if unit == "h":
            return amount
        if unit == "d":
            return amount * 24.0

    multiplier = 1.0
    if text.endswith("k"):
        multiplier = 1_000.0
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text.endswith("b"):
        multiplier = 1_000_000_000.0
        text = text[:-1]

    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def map_binary_outcome(series: pd.Series) -> pd.Series:
    normalized = series.map(normalize_text)
    positive_labels = {"yes", "true", "1", "y", "win", "won"}
    negative_labels = {"no", "false", "0", "n", "lose", "lost"}

    mapped = []
    for value in normalized:
        if value in positive_labels:
            mapped.append(1.0)
        elif value in negative_labels:
            mapped.append(0.0)
        else:
            numeric = parse_numeric_token(value)
            if numeric in {0.0, 1.0}:
                mapped.append(float(numeric))
            else:
                mapped.append(np.nan)

    mapped_series = pd.Series(mapped, index=series.index, dtype="float32")
    class_count = mapped_series.dropna().nunique()
    if class_count < 2:
        raise RuntimeError(
            "Could not derive a binary target from 'Outcome'. "
            "Expected labels like Yes/No, True/False, or 1/0."
        )
    return mapped_series


def build_feature_matrix(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    all_columns = [column for column in df.columns if column != target_col]
    target = map_binary_outcome(df[target_col])
    valid_rows = target.notna()

    feature_df = df.loc[valid_rows, all_columns].copy()
    target = target.loc[valid_rows].astype("float32")

    numeric_columns: dict[str, pd.Series] = {}
    categorical_columns: dict[str, pd.Series] = {}

    for column in feature_df.columns:
        parsed = feature_df[column].map(parse_numeric_token)
        valid_ratio = parsed.notna().mean()
        if valid_ratio >= 0.8:
            filled = parsed.astype("float32")
            median = (
                float(np.nanmedian(filled.to_numpy(dtype="float32")))
                if filled.notna().any()
                else 0.0
            )
            numeric_columns[column] = filled.fillna(median)
        else:
            categorical_columns[column] = (
                feature_df[column].map(normalize_text).replace("", "<missing>")
            )

    numeric_frame = pd.DataFrame(numeric_columns, index=feature_df.index)
    if categorical_columns:
        categorical_frame = pd.DataFrame(categorical_columns, index=feature_df.index)
        categorical_frame = pd.get_dummies(categorical_frame, dtype="float32")
    else:
        categorical_frame = pd.DataFrame(index=feature_df.index)

    full_features = pd.concat([numeric_frame, categorical_frame], axis=1)
    if full_features.empty:
        raise RuntimeError("No usable features found after preprocessing.")

    return full_features.astype("float32"), target


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for cls in (0.0, 1.0):
        class_indices = np.where(y == cls)[0]
        rng.shuffle(class_indices)

        n = len(class_indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_parts.append(class_indices[:n_train])
        val_parts.append(class_indices[n_train : n_train + n_val])
        test_parts.append(class_indices[n_train + n_val :])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()

    tp = float(((predictions == 1) & (targets == 1)).sum().item())
    tn = float(((predictions == 0) & (targets == 0)).sum().item())
    fp = float(((predictions == 1) & (targets == 0)).sum().item())
    fn = float(((predictions == 0) & (targets == 1)).sum().item())

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    model: nn.Module, features: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(features).squeeze(1)
        return compute_metrics(logits, targets)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch binary classifier for insider outcome prediction."
    )
    parser.add_argument(
        "--input", default="insider_data.csv", help="Path to CSV data file"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    df.columns = [column.strip() for column in df.columns]

    target_col = next(
        (column for column in df.columns if column.lower() == "outcome"), None
    )
    if target_col is None:
        raise RuntimeError("CSV is missing required target column: Outcome")

    features_df, target_series = build_feature_matrix(df, target_col)
    x_all = features_df.to_numpy(dtype="float32")
    y_all = target_series.to_numpy(dtype="float32")

    train_idx, val_idx, test_idx = stratified_split_indices(
        y_all,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=args.seed,
    )

    x_train = x_all[train_idx]
    x_val = x_all[val_idx]
    x_test = x_all[test_idx]
    y_train = y_all[train_idx]
    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    train_mean = x_train.mean(axis=0)
    train_std = x_train.std(axis=0)
    train_std[train_std < 1e-8] = 1.0

    x_train = (x_train - train_mean) / train_std
    x_val = (x_val - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    train_features = torch.tensor(x_train, dtype=torch.float32)
    train_targets = torch.tensor(y_train, dtype=torch.float32)
    val_features = torch.tensor(x_val, dtype=torch.float32)
    val_targets = torch.tensor(y_val, dtype=torch.float32)
    test_features = torch.tensor(x_test, dtype=torch.float32)
    test_targets = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )

    input_dim = train_features.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    positive_count = float((y_train == 1).sum())
    negative_count = float((y_train == 0).sum())
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)], dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_features, val_targets)
            print(
                f"Epoch {epoch:02d}/{args.epochs} - "
                f"val_acc={val_metrics['accuracy']:.4f}, "
                f"val_f1={val_metrics['f1']:.4f}"
            )

    test_metrics = evaluate(model, test_features, test_targets)

    print("\nFinal test metrics")
    print(f"Rows used:         {len(features_df)}")
    print(f"Feature count:     {features_df.shape[1]}")
    print(f"Train/Val/Test:    {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Test accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test precision:    {test_metrics['precision']:.4f}")
    print(f"Test recall:       {test_metrics['recall']:.4f}")
    print(f"Test F1 score:     {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
