from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingConfig:
    train_csv: Path = Path("data/train.csv")
    val_split: float = 0.2
    seed: int = 42
    max_trials: int = 48
    runs_per_candidate: int = 3
    early_stopping_patience: int = 12
    model_out: Path = Path("model/binary_classifier.pt")
    metadata_out: Path = Path("model/binary_classifier_meta.json")


class BinaryClassifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers: int,
        width_decay: float,
        dropout: float,
        activation: str,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        hidden_dims: list[int] = [hidden_dim]
        for _ in range(1, num_layers):
            next_dim = max(16, int(round(hidden_dims[-1] * width_decay)))
            hidden_dims.append(next_dim)

        layers: list[nn.Module] = []
        prev_dim = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(make_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    raise ValueError(f"Unsupported activation '{name}'")


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
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def make_split_indices(
    y: torch.Tensor, config: TrainingConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = y.size(0)
    if num_rows < 2:
        raise ValueError(
            "Need at least 2 labeled rows after filtering missing outcomes."
        )

    pos_idx = torch.where(y >= 0.5)[0]
    neg_idx = torch.where(y < 0.5)[0]

    if len(pos_idx) < 2 or len(neg_idx) < 2:
        generator = torch.Generator().manual_seed(config.seed)
        perm = torch.randperm(num_rows, generator=generator)
        val_size = max(1, int(num_rows * config.val_split))
        train_size = max(1, num_rows - val_size)
        if train_size + val_size > num_rows:
            val_size = num_rows - train_size
        train_idx = perm[:train_size]
        val_idx = perm[train_size : train_size + val_size]
        return train_idx, val_idx

    pos_gen = torch.Generator().manual_seed(config.seed + 11)
    neg_gen = torch.Generator().manual_seed(config.seed + 23)
    pos_perm = pos_idx[torch.randperm(len(pos_idx), generator=pos_gen)]
    neg_perm = neg_idx[torch.randperm(len(neg_idx), generator=neg_gen)]

    pos_val = max(1, int(round(len(pos_perm) * config.val_split)))
    neg_val = max(1, int(round(len(neg_perm) * config.val_split)))
    pos_val = min(pos_val, len(pos_perm) - 1)
    neg_val = min(neg_val, len(neg_perm) - 1)

    val_idx = torch.cat([pos_perm[:pos_val], neg_perm[:neg_val]])
    train_idx = torch.cat([pos_perm[pos_val:], neg_perm[neg_val:]])

    shuffle_gen = torch.Generator().manual_seed(config.seed + 37)
    train_idx = train_idx[torch.randperm(len(train_idx), generator=shuffle_gen)]
    val_idx = val_idx[torch.randperm(len(val_idx), generator=shuffle_gen)]
    return train_idx, val_idx


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def generate_hyperparameter_candidates(config: TrainingConfig) -> list[dict]:
    search_space = {
        "batch_size": [64, 128, 256],
        "hidden_dim": [64, 128, 256],
        "num_layers": [2, 3, 4],
        "width_decay": [1.0, 0.7, 0.5],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "activation": ["relu", "gelu", "leaky_relu"],
        "use_batch_norm": [False, True],
        "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3],
        "weight_decay": [0.0, 1e-5, 1e-4, 5e-4],
        "optimizer": ["adamw", "radam"],
        "use_pos_weight": [False, True],
        "epochs": [30, 50, 80],
    }

    keys = list(search_space.keys())
    all_candidates = [
        dict(zip(keys, values))
        for values in itertools.product(*(search_space[k] for k in keys))
    ]

    generator = torch.Generator().manual_seed(config.seed)
    perm = torch.randperm(len(all_candidates), generator=generator).tolist()
    max_trials = min(config.max_trials, len(all_candidates))
    return [all_candidates[i] for i in perm[:max_trials]]


def build_optimizer(model: nn.Module, candidate: dict) -> torch.optim.Optimizer:
    optimizer_name = str(candidate["optimizer"]).lower()
    learning_rate = float(candidate["learning_rate"])
    weight_decay = float(candidate["weight_decay"])

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "radam":
        return torch.optim.RAdam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{candidate['optimizer']}'")


def train_single_run(
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: list[str],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    candidate: dict,
    device: torch.device,
    seed: int,
    early_stopping_patience: int,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader = make_loaders(
        X,
        y,
        train_idx,
        val_idx,
        batch_size=int(candidate["batch_size"]),
    )

    model = BinaryClassifier(
        in_features=len(feature_names),
        hidden_dim=int(candidate["hidden_dim"]),
        num_layers=int(candidate["num_layers"]),
        width_decay=float(candidate["width_decay"]),
        dropout=float(candidate["dropout"]),
        activation=str(candidate["activation"]),
        use_batch_norm=bool(candidate["use_batch_norm"]),
    ).to(device)
    if bool(candidate["use_pos_weight"]):
        y_train = y[train_idx]
        pos_count = float((y_train >= 0.5).sum().item())
        neg_count = float((y_train < 0.5).sum().item())
        pos_weight_val = neg_count / max(pos_count, 1.0)
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = build_optimizer(model, candidate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(candidate["epochs"])),
    )

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    epochs = int(candidate["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        improved = (val_acc > best_val_acc) or (
            abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state_dict = clone_state_dict(model)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            break

    if best_state_dict is None:
        raise RuntimeError("Failed to capture best state during candidate training.")

    return {
        "seed": seed,
        "candidate": candidate,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "state_dict": best_state_dict,
    }


def summarize_run_metrics(run_results: Sequence[dict]) -> dict:
    if not run_results:
        raise ValueError("run_results cannot be empty")

    val_accs = [float(r["best_val_acc"]) for r in run_results]
    val_losses = [float(r["best_val_loss"]) for r in run_results]
    mean_acc = sum(val_accs) / len(val_accs)
    mean_loss = sum(val_losses) / len(val_losses)
    std_acc = float(torch.tensor(val_accs).std(unbiased=False).item())

    best_run = run_results[0]
    for run in run_results[1:]:
        is_better = (run["best_val_acc"] > best_run["best_val_acc"]) or (
            abs(run["best_val_acc"] - best_run["best_val_acc"]) < 1e-12
            and run["best_val_loss"] < best_run["best_val_loss"]
        )
        if is_better:
            best_run = run

    return {
        "mean_val_acc": mean_acc,
        "std_val_acc": std_acc,
        "mean_val_loss": mean_loss,
        "best_run": best_run,
        "run_metrics": [
            {
                "run_seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "best_val_acc": r["best_val_acc"],
                "best_val_loss": r["best_val_loss"],
            }
            for r in run_results
        ],
    }


def evaluate_candidate_with_repeats(
    X: torch.Tensor,
    y: torch.Tensor,
    feature_names: list[str],
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    candidate: dict,
    device: torch.device,
    base_seed: int,
    runs_per_candidate: int,
    early_stopping_patience: int,
) -> dict:
    run_results: list[dict] = []
    for run_id in range(runs_per_candidate):
        run_seed = base_seed + run_id * 997
        run_result = train_single_run(
            X,
            y,
            feature_names,
            train_idx,
            val_idx,
            candidate,
            device,
            seed=run_seed,
            early_stopping_patience=early_stopping_patience,
        )
        run_results.append(run_result)

    summary = summarize_run_metrics(run_results)
    return {
        "candidate": candidate,
        "mean_val_acc": summary["mean_val_acc"],
        "std_val_acc": summary["std_val_acc"],
        "mean_val_loss": summary["mean_val_loss"],
        "best_run": summary["best_run"],
        "run_metrics": summary["run_metrics"],
    }


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, feature_names, preprocessing_meta = load_and_preprocess(config)
    train_idx, val_idx = make_split_indices(y, config)
    candidates = generate_hyperparameter_candidates(config)

    print(
        f"Running hyperparameter search over {len(candidates)} candidates "
        f"with {config.runs_per_candidate} runs each..."
    )

    best_result: dict | None = None
    leaderboard: list[dict] = []
    for trial, candidate in enumerate(candidates, start=1):
        result = evaluate_candidate_with_repeats(
            X,
            y,
            feature_names,
            train_idx,
            val_idx,
            candidate,
            device,
            base_seed=config.seed + trial * 1000,
            runs_per_candidate=config.runs_per_candidate,
            early_stopping_patience=config.early_stopping_patience,
        )
        leaderboard.append(result)

        print(
            f"Trial {trial:02d}/{len(candidates)} | "
            f"params={candidate} | "
            f"mean_val_acc={result['mean_val_acc']:.4f}±{result['std_val_acc']:.4f} | "
            f"mean_val_loss={result['mean_val_loss']:.4f} | "
            f"best_run_acc={result['best_run']['best_val_acc']:.4f} "
            f"@ epoch {result['best_run']['best_epoch']}"
        )

        if best_result is None:
            best_result = result
            continue

        is_better = (result["mean_val_acc"] > best_result["mean_val_acc"]) or (
            abs(result["mean_val_acc"] - best_result["mean_val_acc"]) < 1e-12
            and result["mean_val_loss"] < best_result["mean_val_loss"]
        )
        if is_better:
            best_result = result

    config.model_out.parent.mkdir(parents=True, exist_ok=True)
    config.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    if best_result is None:
        raise RuntimeError("Training did not produce a valid model state.")

    best_model = {
        "model_state_dict": best_result["best_run"]["state_dict"],
        "feature_names": feature_names,
        "best_hyperparameters": best_result["candidate"],
        "best_epoch": best_result["best_run"]["best_epoch"],
        "best_val_acc": best_result["best_run"]["best_val_acc"],
        "best_val_loss": best_result["best_run"]["best_val_loss"],
    }
    torch.save(best_model, config.model_out)

    leaderboard_sorted = sorted(
        leaderboard,
        key=lambda r: (r["mean_val_acc"], -r["mean_val_loss"]),
        reverse=True,
    )
    top_results = []
    for row in leaderboard_sorted[:5]:
        top_results.append(
            {
                "hyperparameters": row["candidate"],
                "mean_val_acc": row["mean_val_acc"],
                "std_val_acc": row["std_val_acc"],
                "mean_val_loss": row["mean_val_loss"],
                "best_run_acc": row["best_run"]["best_val_acc"],
                "best_run_loss": row["best_run"]["best_val_loss"],
                "best_run_epoch": row["best_run"]["best_epoch"],
                "run_metrics": row["run_metrics"],
            }
        )

    metadata = {
        "train_csv": str(config.train_csv),
        "n_features": len(feature_names),
        "search_trials": len(candidates),
        "runs_per_candidate": config.runs_per_candidate,
        "early_stopping_patience": config.early_stopping_patience,
        "best_hyperparameters": best_result["candidate"],
        "best_epoch": best_result["best_run"]["best_epoch"],
        "best_val_acc": best_result["best_run"]["best_val_acc"],
        "best_val_loss": best_result["best_run"]["best_val_loss"],
        "best_mean_val_acc": best_result["mean_val_acc"],
        "best_std_val_acc": best_result["std_val_acc"],
        "best_mean_val_loss": best_result["mean_val_loss"],
        "top_results": top_results,
        "preprocessing": preprocessing_meta,
    }
    with config.metadata_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Best hyperparameters:", best_result["candidate"])
    print(
        f"Best mean validation accuracy={best_result['mean_val_acc']:.4f} "
        f"±{best_result['std_val_acc']:.4f}, "
        f"mean loss={best_result['mean_val_loss']:.4f}"
    )
    print(
        f"Best run accuracy={best_result['best_run']['best_val_acc']:.4f}, "
        f"loss={best_result['best_run']['best_val_loss']:.4f}, "
        f"epoch={best_result['best_run']['best_epoch']}"
    )
    print(f"Saved best model to {config.model_out}")
    print(f"Saved preprocessing metadata to {config.metadata_out}")


if __name__ == "__main__":
    train(TrainingConfig())
