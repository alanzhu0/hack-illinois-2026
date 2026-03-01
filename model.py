import argparse
import hashlib
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
    df: pd.DataFrame,
    target_col: str,
    min_category_count: int,
    max_unique_ratio: float,
    max_unique_values: int,
    hash_buckets: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    all_columns = [column for column in df.columns if column != target_col]
    target = map_binary_outcome(df[target_col])
    valid_rows = target.notna()

    feature_df = df.loc[valid_rows, all_columns].copy()
    target = target.loc[valid_rows].astype("float32")

    numeric_columns: dict[str, pd.Series] = {}
    categorical_columns: dict[str, pd.Series] = {}
    transformed_high_cardinality: list[str] = []
    dropped_identifier_like: list[str] = []

    forced_drop_columns = {
        "name",
        "tx hash",
        "profile",
        "market link",
        "polymarket",
    }

    for column in feature_df.columns:
        normalized_column_name = normalize_text(column)
        if normalized_column_name in forced_drop_columns:
            dropped_identifier_like.append(column)
            continue

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
            categorical = (
                feature_df[column].map(normalize_text).replace("", "<missing>")
            )
            non_missing = categorical[categorical != "<missing>"]

            unique_count = int(non_missing.nunique())
            unique_ratio = (
                unique_count / max(int(non_missing.shape[0]), 1)
                if non_missing.shape[0] > 0
                else 0.0
            )

            if unique_count > max_unique_values or unique_ratio > max_unique_ratio:
                transformed_high_cardinality.append(column)

                frequency_map = non_missing.value_counts(normalize=True)
                numeric_columns[f"{column}__freq"] = (
                    categorical.map(frequency_map).fillna(0.0).astype("float32")
                )

                def stable_bucket(value: str) -> str:
                    if value == "<missing>":
                        return "<missing>"
                    digest = hashlib.md5(
                        value.encode("utf-8"), usedforsecurity=False
                    ).hexdigest()
                    bucket = int(digest, 16) % max(hash_buckets, 1)
                    return f"bucket_{bucket}"

                categorical_columns[f"{column}__bucket"] = categorical.map(
                    stable_bucket
                )
                continue

            value_counts = non_missing.value_counts()
            frequent_values = set(
                value_counts[value_counts >= min_category_count].index.tolist()
            )
            categorical = categorical.where(
                categorical.isin(frequent_values) | (categorical == "<missing>"),
                "<rare>",
            )
            categorical_columns[column] = categorical

    numeric_frame = pd.DataFrame(numeric_columns, index=feature_df.index)
    if categorical_columns:
        categorical_frame = pd.DataFrame(categorical_columns, index=feature_df.index)
        categorical_frame = pd.get_dummies(categorical_frame, dtype="float32")
    else:
        categorical_frame = pd.DataFrame(index=feature_df.index)

    full_features = pd.concat([numeric_frame, categorical_frame], axis=1)
    if full_features.empty:
        raise RuntimeError("No usable features found after preprocessing.")

    preprocessing_info = {
        "transformed_high_cardinality": transformed_high_cardinality,
        "dropped_identifier_like": dropped_identifier_like,
        "used_numeric": list(numeric_columns.keys()),
        "used_categorical": list(categorical_columns.keys()),
    }
    return full_features.astype("float32"), target, preprocessing_info


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


def build_model(
    input_dim: int,
    hidden_dim1: int,
    hidden_dim2: int,
    dropout: float,
    input_dropout: float,
) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=input_dropout),
        nn.Linear(input_dim, hidden_dim1),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.ReLU(),
        nn.Linear(hidden_dim2, 1),
    )


def run_experiment(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim1: int,
    hidden_dim2: int,
    dropout: float,
    input_dropout: float,
    patience: int,
) -> tuple[nn.Module, dict[str, float], dict[str, float], int]:
    torch.manual_seed(seed)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )

    input_dim = train_features.shape[1]
    model = build_model(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
        input_dropout=input_dropout,
    )

    positive_count = float((train_targets == 1).sum().item())
    negative_count = float((train_targets == 0).sum().item())
    pos_weight = torch.tensor(
        [negative_count / max(positive_count, 1.0)], dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch_features).squeeze(1)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_features, val_targets)
        val_f1 = val_metrics["f1"]

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    final_val_metrics = evaluate(model, val_features, val_targets)
    final_test_metrics = evaluate(model, test_features, test_targets)
    return model, final_val_metrics, final_test_metrics, best_epoch


def tune_hyperparameters(
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    *,
    seed: int,
    trials: int,
    tune_epochs: int,
    patience: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)

    lr_choices = [3e-4, 5e-4, 8e-4, 1e-3, 2e-3]
    batch_choices = [128, 256, 512]
    hidden_choices = [(64, 32), (128, 64), (128, 32), (256, 64), (256, 128)]
    dropout_choices = [0.2, 0.3, 0.4, 0.5]
    input_dropout_choices = [0.0, 0.05, 0.1, 0.15]
    weight_decay_choices = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3]

    best_config: dict[str, float] = {}
    best_val_f1 = -1.0
    best_val_acc = -1.0

    print(f"Running hyperparameter tuning for {trials} trial(s)...")

    for trial in range(1, trials + 1):
        hidden_dim1, hidden_dim2 = hidden_choices[rng.integers(0, len(hidden_choices))]
        config = {
            "lr": float(lr_choices[rng.integers(0, len(lr_choices))]),
            "batch_size": int(batch_choices[rng.integers(0, len(batch_choices))]),
            "hidden_dim1": int(hidden_dim1),
            "hidden_dim2": int(hidden_dim2),
            "dropout": float(dropout_choices[rng.integers(0, len(dropout_choices))]),
            "input_dropout": float(
                input_dropout_choices[rng.integers(0, len(input_dropout_choices))]
            ),
            "weight_decay": float(
                weight_decay_choices[rng.integers(0, len(weight_decay_choices))]
            ),
        }

        _, val_metrics, test_metrics, best_epoch = run_experiment(
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features,
            test_targets,
            seed=seed + trial,
            epochs=tune_epochs,
            batch_size=int(config["batch_size"]),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            hidden_dim1=int(config["hidden_dim1"]),
            hidden_dim2=int(config["hidden_dim2"]),
            dropout=float(config["dropout"]),
            input_dropout=float(config["input_dropout"]),
            patience=patience,
        )

        print(
            f"Trial {trial:02d}/{trials}: "
            f"val_f1={val_metrics['f1']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"test_f1={test_metrics['f1']:.4f}, "
            f"best_epoch={best_epoch}, "
            f"config={config}"
        )

        if val_metrics["f1"] > best_val_f1 + 1e-6 or (
            abs(val_metrics["f1"] - best_val_f1) <= 1e-6
            and val_metrics["accuracy"] > best_val_acc
        ):
            best_val_f1 = val_metrics["f1"]
            best_val_acc = val_metrics["accuracy"]
            best_config = config

    print("Best tuning config selected by validation F1:")
    print(best_config)
    return best_config


def save_model_checkpoint(
    output_path: Path,
    model: nn.Module,
    args: argparse.Namespace,
    feature_columns: list[str],
    train_mean: np.ndarray,
    train_std: np.ndarray,
    preprocessing_info: dict[str, list[str]],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    best_epoch: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_columns,
        "train_mean": train_mean.astype("float32"),
        "train_std": train_std.astype("float32"),
        "preprocessing_info": preprocessing_info,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim1": args.hidden_dim1,
            "hidden_dim2": args.hidden_dim2,
            "dropout": args.dropout,
            "input_dropout": args.input_dropout,
            "patience": args.patience,
            "min_category_count": args.min_category_count,
            "max_unique_ratio": args.max_unique_ratio,
            "max_unique_values": args.max_unique_values,
            "hash_buckets": args.hash_buckets,
            "seed": args.seed,
        },
        "metrics": {
            "best_epoch": best_epoch,
            "val": val_metrics,
            "test": test_metrics,
        },
    }
    torch.save(checkpoint, output_path)


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
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-3, help="AdamW weight decay"
    )
    parser.add_argument(
        "--hidden-dim1", type=int, default=256, help="Hidden layer 1 size"
    )
    parser.add_argument(
        "--hidden-dim2", type=int, default=128, help="Hidden layer 2 size"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.4, help="Dropout probability"
    )
    parser.add_argument(
        "--input-dropout",
        type=float,
        default=0.0,
        help="Input feature dropout probability",
    )
    parser.add_argument(
        "--patience", type=int, default=4, help="Early stopping patience"
    )
    parser.add_argument(
        "--min-category-count",
        type=int,
        default=5,
        help="Minimum frequency for categorical values before bucketing as <rare>",
    )
    parser.add_argument(
        "--max-unique-ratio",
        type=float,
        default=0.98,
        help="Drop categorical columns when unique/non-missing ratio exceeds this threshold",
    )
    parser.add_argument(
        "--max-unique-values",
        type=int,
        default=1000,
        help="Drop categorical columns when unique values exceed this count",
    )
    parser.add_argument(
        "--hash-buckets",
        type=int,
        default=64,
        help="Number of hash buckets for high-cardinality categorical compression",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=0,
        help="If > 0, run random hyperparameter tuning with this many trials",
    )
    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=12,
        help="Max epochs per tuning trial",
    )
    parser.add_argument(
        "--save-model",
        default="",
        help="Optional path to save trained model checkpoint (.pt)",
    )
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

    features_df, target_series, preprocessing_info = build_feature_matrix(
        df,
        target_col,
        min_category_count=args.min_category_count,
        max_unique_ratio=args.max_unique_ratio,
        max_unique_values=args.max_unique_values,
        hash_buckets=args.hash_buckets,
    )
    print(f"\nTraining features ({features_df.shape[1]} total):")
    for feature_name in features_df.columns:
        print(f"- {feature_name}")

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

    if args.tune_trials > 0:
        tuned = tune_hyperparameters(
            train_features,
            train_targets,
            val_features,
            val_targets,
            test_features,
            test_targets,
            seed=args.seed,
            trials=args.tune_trials,
            tune_epochs=args.tune_epochs,
            patience=args.patience,
        )
        args.lr = float(tuned["lr"])
        args.batch_size = int(tuned["batch_size"])
        args.hidden_dim1 = int(tuned["hidden_dim1"])
        args.hidden_dim2 = int(tuned["hidden_dim2"])
        args.dropout = float(tuned["dropout"])
        args.input_dropout = float(tuned["input_dropout"])
        args.weight_decay = float(tuned["weight_decay"])

    model, val_metrics, test_metrics, best_epoch = run_experiment(
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        test_targets,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        patience=args.patience,
    )

    print("\nFinal test metrics")
    print(f"Rows used:         {len(features_df)}")
    print(f"Feature count:     {features_df.shape[1]}")
    print(
        "Leakage handling:  "
        f"{len(preprocessing_info['dropped_identifier_like'])} identifier-like, "
        f"{len(preprocessing_info['transformed_high_cardinality'])} high-cardinality transformed"
    )
    print(f"Train/Val/Test:    {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    print(f"Best epoch:        {best_epoch}")
    print(
        "Chosen params:     "
        f"lr={args.lr}, batch={args.batch_size}, wd={args.weight_decay}, "
        f"h1={args.hidden_dim1}, h2={args.hidden_dim2}, drop={args.dropout}, "
        f"in_drop={args.input_dropout}"
    )
    print(f"Val accuracy:      {val_metrics['accuracy']:.4f}")
    print(f"Val F1 score:      {val_metrics['f1']:.4f}")
    print(f"Test accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Test precision:    {test_metrics['precision']:.4f}")
    print(f"Test recall:       {test_metrics['recall']:.4f}")
    print(f"Test F1 score:     {test_metrics['f1']:.4f}")

    if args.save_model:
        save_path = Path(args.save_model)

        previous_accuracy: Optional[float] = None
        if save_path.exists():
            previous_checkpoint = torch.load(
                save_path,
                map_location="cpu",
                weights_only=False,
            )
            metrics = (
                previous_checkpoint.get("metrics", {})
                if isinstance(previous_checkpoint, dict)
                else {}
            )
            test_section = metrics.get("test", {}) if isinstance(metrics, dict) else {}
            prev_acc_raw = (
                test_section.get("accuracy") if isinstance(test_section, dict) else None
            )
            if prev_acc_raw is not None:
                previous_accuracy = float(prev_acc_raw)

        if (
            previous_accuracy is not None
            and test_metrics["accuracy"] <= previous_accuracy
        ):
            print(
                "Skipped save:      "
                f"current test accuracy {test_metrics['accuracy']:.4f} "
                f"<= previous {previous_accuracy:.4f}"
            )
            return

        save_model_checkpoint(
            output_path=save_path,
            model=model,
            args=args,
            feature_columns=features_df.columns.tolist(),
            train_mean=train_mean,
            train_std=train_std,
            preprocessing_info=preprocessing_info,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_epoch=best_epoch,
        )
        if previous_accuracy is None:
            print(
                "Saved model:       " f"{save_path} (no previous checkpoint to compare)"
            )
        else:
            print(
                "Saved model:       "
                f"{save_path} (improved test accuracy {previous_accuracy:.4f} -> {test_metrics['accuracy']:.4f})"
            )


if __name__ == "__main__":
    main()
