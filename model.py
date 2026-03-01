from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    train_csv: Path = Path("data/train.csv")
    batch_size: int = 256
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    val_split: float = 0.2
    seed: int = 42
    model_out: Path = Path("model/binary_classifier.pt")
    metadata_out: Path = Path("model/binary_classifier_meta.json")


class BinaryClassifier(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def find_outcome_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if str(col).strip().lower() == "outcome":
            return col
    raise ValueError("Could not find an 'outcome' column (case-insensitive).")


def coerce_binary_target(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map(
        {
            "true": 1,
            "false": 0,
            "yes": 1,
            "no": 0,
            "win": 1,
            "lose": 0,
            "won": 1,
            "lost": 0,
            "1": 1,
            "0": 0,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")
    combined = mapped.where(mapped.notna(), numeric)
    return combined


def load_and_preprocess(
    config: TrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
    df = pd.read_csv(config.train_csv)
    outcome_col = find_outcome_column(df)

    y_raw = coerce_binary_target(df[outcome_col])
    keep_mask = y_raw.notna()
    df = df.loc[keep_mask].copy()
    y_raw = y_raw.loc[keep_mask]

    y = (y_raw.astype(float) >= 0.5).astype("float32")

    X_df = df.drop(columns=[outcome_col]).copy()
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    col_medians = X_df.median(numeric_only=True)
    X_df = X_df.fillna(col_medians).fillna(0.0)

    feature_names = [str(col) for col in X_df.columns]
    X = X_df.to_numpy(dtype="float32")

    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X = (X - means) / stds

    preprocessing_meta = {
        "outcome_column": outcome_col,
        "feature_names": feature_names,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "median_fill_values": [
            float(col_medians.get(col, 0.0)) for col in X_df.columns
        ],
    }

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32)
    return X_tensor, y_tensor, feature_names, preprocessing_meta


def make_loaders(
    X: torch.Tensor,
    y: torch.Tensor,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    num_rows = X.size(0)
    if num_rows < 2:
        raise ValueError(
            "Need at least 2 labeled rows after filtering missing outcomes."
        )

    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(num_rows, generator=generator)

    val_size = max(1, int(num_rows * config.val_split))
    train_size = max(1, num_rows - val_size)
    if train_size + val_size > num_rows:
        val_size = num_rows - train_size

    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = xb.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == yb).sum().item()
            total_examples += batch_size

    avg_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return avg_loss, accuracy


def train(config: TrainingConfig) -> None:
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, feature_names, preprocessing_meta = load_and_preprocess(config)
    train_loader, val_loader = make_loaders(X, y, config)

    model = BinaryClassifier(
        in_features=len(feature_names), hidden_dim=config.hidden_dim
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = xb.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (preds == yb).sum().item()
            train_examples += batch_size

        train_loss = train_loss_sum / max(train_examples, 1)
        train_acc = train_correct / max(train_examples, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_names": feature_names,
            }

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    config.model_out.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    torch.save(best_state, config.model_out)

    metadata = {
        "train_csv": str(config.train_csv),
        "n_features": len(feature_names),
        "hidden_dim": config.hidden_dim,
        "epochs": config.epochs,
        "best_val_loss": best_val_loss,
        "preprocessing": preprocessing_meta,
    }
    with config.metadata_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved best model to {config.model_out}")
    print(f"Saved preprocessing metadata to {config.metadata_out}")


if __name__ == "__main__":
    train(TrainingConfig())
