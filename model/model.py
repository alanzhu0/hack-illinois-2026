import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLUMN = "Outcome"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_numeric(value: str) -> float:
	text = (value or "").strip()
	if text == "":
		return float("nan")

	lower = text.lower()
	if lower in {"n/a", "na", "none", "null", "nan"}:
		return float("nan")

	text = text.replace(",", "")
	try:
		return float(text)
	except ValueError:
		return float("nan")


def load_csv_dataset(path: Path, target_column: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
	with path.open("r", newline="", encoding="utf-8") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError(f"CSV has no headers: {path}")

		if target_column not in reader.fieldnames:
			raise RuntimeError(f"Target column '{target_column}' not found in {path}")

		feature_columns = [name for name in reader.fieldnames if name != target_column]
		feature_rows: List[List[float]] = []
		target_values: List[float] = []

		for row in reader:
			target_text = (row.get(target_column) or "").strip()
			if target_text == "":
				continue
			target_values.append(float(target_text))
			feature_rows.append([parse_numeric(row.get(column, "")) for column in feature_columns])

	features = np.array(feature_rows, dtype=np.float32)
	targets = np.array(target_values, dtype=np.float32)
	return features, targets, feature_columns


def stratified_train_val_test_indices(
	targets: np.ndarray,
	val_ratio: float,
	test_ratio: float,
	seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	if val_ratio <= 0 or test_ratio <= 0:
		raise ValueError("val_ratio and test_ratio must both be > 0")
	if val_ratio + test_ratio >= 1.0:
		raise ValueError("val_ratio + test_ratio must be < 1")

	rng = np.random.default_rng(seed)

	def split_one_class(class_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		indices = class_indices.copy()
		rng.shuffle(indices)
		n_total = len(indices)
		n_test = max(1, int(n_total * test_ratio))
		n_val = max(1, int(n_total * val_ratio))
		n_train = n_total - n_val - n_test
		if n_train < 1:
			n_train = 1
			if n_val > n_test:
				n_val = max(1, n_val - 1)
			else:
				n_test = max(1, n_test - 1)

		train_slice = indices[:n_train]
		val_slice = indices[n_train:n_train + n_val]
		test_slice = indices[n_train + n_val:]
		return train_slice, val_slice, test_slice

	class0 = np.where(targets == 0)[0]
	class1 = np.where(targets == 1)[0]

	train0, val0, test0 = split_one_class(class0)
	train1, val1, test1 = split_one_class(class1)

	train_indices = np.concatenate([train0, train1])
	val_indices = np.concatenate([val0, val1])
	test_indices = np.concatenate([test0, test1])

	rng.shuffle(train_indices)
	rng.shuffle(val_indices)
	rng.shuffle(test_indices)
	return train_indices, val_indices, test_indices


def impute_and_scale(
	train_x: np.ndarray,
	val_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	medians = np.nanmedian(train_x, axis=0)
	medians = np.where(np.isnan(medians), 0.0, medians)

	train_filled = np.where(np.isnan(train_x), medians, train_x)
	val_filled = np.where(np.isnan(val_x), medians, val_x)

	means = train_filled.mean(axis=0)
	stds = train_filled.std(axis=0)
	stds = np.where(stds < 1e-6, 1.0, stds)

	train_scaled = (train_filled - means) / stds
	val_scaled = (val_filled - means) / stds
	return train_scaled.astype(np.float32), val_scaled.astype(np.float32), medians.astype(np.float32), means.astype(np.float32), stds.astype(np.float32)


class BinaryMLP(nn.Module):
	def __init__(self, input_dim: int, hidden_layers: Sequence[int], dropout: float) -> None:
		super().__init__()
		layers: List[nn.Module] = []
		current_dim = input_dim
		for hidden_dim in hidden_layers:
			layers.append(nn.Linear(current_dim, hidden_dim))
			layers.append(nn.BatchNorm1d(hidden_dim))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			current_dim = hidden_dim
		layers.append(nn.Linear(current_dim, 1))
		self.network = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.network(x).squeeze(1)


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
	return metrics_from_logits_at_threshold(logits, targets, threshold=0.5)


def metrics_from_logits_at_threshold(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> Dict[str, float]:
	probabilities = torch.sigmoid(logits)
	predictions = (probabilities >= threshold).float()

	true_positive = ((predictions == 1) & (targets == 1)).sum().item()
	true_negative = ((predictions == 0) & (targets == 0)).sum().item()
	false_positive = ((predictions == 1) & (targets == 0)).sum().item()
	false_negative = ((predictions == 0) & (targets == 1)).sum().item()

	accuracy = (true_positive + true_negative) / max(1, len(targets))
	precision = true_positive / max(1, true_positive + false_positive)
	recall = true_positive / max(1, true_positive + false_negative)
	f1 = 2 * precision * recall / max(1e-8, precision + recall)

	return {
		"accuracy": float(accuracy),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
	}


def find_best_threshold(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, Dict[str, float]]:
	best_threshold = 0.5
	best_metrics = metrics_from_logits_at_threshold(logits, targets, best_threshold)

	for threshold in np.linspace(0.2, 0.8, 121):
		metrics = metrics_from_logits_at_threshold(logits, targets, float(threshold))
		if metrics["f1"] > best_metrics["f1"]:
			best_metrics = metrics
			best_threshold = float(threshold)

	return best_threshold, best_metrics


@dataclass
class TrainConfig:
	hidden_layers: Tuple[int, ...]
	dropout: float
	learning_rate: float
	weight_decay: float


def collect_logits_and_targets(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, torch.Tensor, torch.Tensor]:
	model.eval()
	all_logits: List[torch.Tensor] = []
	all_targets: List[torch.Tensor] = []
	with torch.no_grad():
		for batch_features, batch_targets in data_loader:
			batch_features = batch_features.to(device)
			batch_targets = batch_targets.to(device)
			logits = model(batch_features)
			all_logits.append(logits.cpu())
			all_targets.append(batch_targets.cpu())

	logits = torch.cat(all_logits)
	targets = torch.cat(all_targets)
	loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
	return float(loss.item()), logits, targets


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Tuple[float, Dict[str, float]]:
	loss, logits, targets = collect_logits_and_targets(model, data_loader, device)
	return loss, metrics_from_logits_at_threshold(logits, targets, threshold)


def train_one_model(
	train_x: np.ndarray,
	train_y: np.ndarray,
	val_x: np.ndarray,
	val_y: np.ndarray,
	config: TrainConfig,
	device: torch.device,
	seed: int,
	epochs: int,
	batch_size: int,
) -> Tuple[nn.Module, Dict[str, float], float]:
	set_seed(seed)

	train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
	val_dataset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	model = BinaryMLP(input_dim=train_x.shape[1], hidden_layers=config.hidden_layers, dropout=config.dropout).to(device)

	positive = float(train_y.sum())
	negative = float(len(train_y) - positive)
	pos_weight = torch.tensor([negative / max(1.0, positive)], dtype=torch.float32, device=device)
	loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer,
		T_max=max(2, epochs),
		eta_min=config.learning_rate * 0.05,
	)

	best_state: Dict[str, torch.Tensor] = {}
	best_f1 = -1.0
	best_threshold = 0.5
	patience = 20
	epochs_without_improvement = 0

	for _ in range(epochs):
		model.train()
		for batch_features, batch_targets in train_loader:
			batch_features = batch_features.to(device)
			batch_targets = batch_targets.to(device)

			optimizer.zero_grad()
			logits = model(batch_features)
			loss = loss_fn(logits, batch_targets)
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
			optimizer.step()

		scheduler.step()

		_, val_logits, val_targets = collect_logits_and_targets(model, val_loader, device)
		candidate_threshold, val_metrics = find_best_threshold(val_logits, val_targets)
		if val_metrics["f1"] > best_f1:
			best_f1 = val_metrics["f1"]
			best_threshold = candidate_threshold
			best_state = {name: parameter.detach().cpu().clone() for name, parameter in model.state_dict().items()}
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		if epochs_without_improvement >= patience:
			break

	model.load_state_dict(best_state)
	_, final_metrics = evaluate(model, val_loader, device, threshold=best_threshold)
	return model, final_metrics, best_threshold


def apply_impute_scale(features: np.ndarray, medians: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
	filled = np.where(np.isnan(features), medians, features)
	return ((filled - means) / stds).astype(np.float32)


def predict_csv(
	model: nn.Module,
	input_path: Path,
	output_path: Path,
	feature_columns: Sequence[str],
	medians: np.ndarray,
	means: np.ndarray,
	stds: np.ndarray,
	threshold: float,
	device: torch.device,
) -> None:
	if not input_path.exists():
		return

	with input_path.open("r", newline="", encoding="utf-8") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError(f"CSV has no headers: {input_path}")

		rows = list(reader)

	feature_rows = []
	for row in rows:
		feature_rows.append([parse_numeric(row.get(column, "")) for column in feature_columns])

	features = np.array(feature_rows, dtype=np.float32)
	features = apply_impute_scale(features, medians, means, stds)

	model.eval()
	with torch.no_grad():
		inputs = torch.from_numpy(features).to(device)
		logits = model(inputs)
		probabilities = torch.sigmoid(logits).cpu().numpy()

	for row, prob in zip(rows, probabilities):
		row["Predicted Probability"] = f"{float(prob):.6f}"
		row["Predicted Outcome"] = "1" if prob >= threshold else "0"

	fieldnames = list(reader.fieldnames)
	if "Predicted Probability" not in fieldnames:
		fieldnames.append("Predicted Probability")
	if "Predicted Outcome" not in fieldnames:
		fieldnames.append("Predicted Outcome")

	with output_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a PyTorch binary classifier for Outcome")
	parser.add_argument("--train", default="train.csv", help="Path to training CSV")
	parser.add_argument("--future", default="future.csv", help="Path to future CSV for inference")
	parser.add_argument("--future-out", default="future_with_predictions.csv", help="Output CSV for future predictions")
	parser.add_argument("--model-out", default="model.pt", help="Path to save trained model checkpoint")
	parser.add_argument("--meta-out", default="model_meta.json", help="Path to save preprocessing metadata")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	args = parser.parse_args()

	base_dir = Path(__file__).resolve().parent
	train_path = (base_dir / args.train).resolve() if not Path(args.train).is_absolute() else Path(args.train)
	future_path = (base_dir / args.future).resolve() if not Path(args.future).is_absolute() else Path(args.future)
	future_out_path = (base_dir / args.future_out).resolve() if not Path(args.future_out).is_absolute() else Path(args.future_out)
	model_out_path = (base_dir / args.model_out).resolve() if not Path(args.model_out).is_absolute() else Path(args.model_out)
	meta_out_path = (base_dir / args.meta_out).resolve() if not Path(args.meta_out).is_absolute() else Path(args.meta_out)

	set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	features, targets, feature_columns = load_csv_dataset(train_path, TARGET_COLUMN)
	train_indices, val_indices, test_indices = stratified_train_val_test_indices(
		targets,
		args.val_ratio,
		args.test_ratio,
		args.seed,
	)

	train_x_raw = features[train_indices]
	val_x_raw = features[val_indices]
	test_x_raw = features[test_indices]
	train_y = targets[train_indices]
	val_y = targets[val_indices]
	test_y = targets[test_indices]

	train_x, val_x, medians, means, stds = impute_and_scale(train_x_raw, val_x_raw)
	test_x = apply_impute_scale(test_x_raw, medians, means, stds)

	candidates = [
		TrainConfig(hidden_layers=(128, 64, 32), dropout=0.20, learning_rate=8e-4, weight_decay=2e-3),
		TrainConfig(hidden_layers=(128, 64, 32), dropout=0.15, learning_rate=1e-3, weight_decay=1.5e-3),
		TrainConfig(hidden_layers=(128, 64, 32), dropout=0.25, learning_rate=6e-4, weight_decay=3e-3),
		TrainConfig(hidden_layers=(128, 64, 32), dropout=0.10, learning_rate=1.2e-3, weight_decay=1e-3),
	]

	best_model = None
	best_metrics: Dict[str, float] = {}
	best_config = None
	best_threshold = 0.5

	for config in candidates:
		model, metrics, threshold = train_one_model(
			train_x=train_x,
			train_y=train_y,
			val_x=val_x,
			val_y=val_y,
			config=config,
			device=device,
			seed=args.seed,
			epochs=args.epochs,
			batch_size=args.batch_size,
		)
		print(
			"Config",
			config,
			"->",
			"accuracy=", f"{metrics['accuracy']:.4f}",
			"precision=", f"{metrics['precision']:.4f}",
			"recall=", f"{metrics['recall']:.4f}",
			"f1=", f"{metrics['f1']:.4f}",
			"threshold=", f"{threshold:.3f}",
		)

		if not best_metrics or metrics["f1"] > best_metrics["f1"]:
			best_model = model
			best_metrics = metrics
			best_config = config
			best_threshold = threshold

	if best_model is None or best_config is None:
		raise RuntimeError("Model training failed to produce a valid candidate")

	test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	_, test_metrics = evaluate(best_model, test_loader, device, threshold=best_threshold)

	torch.save(
		{
			"state_dict": best_model.state_dict(),
			"feature_columns": feature_columns,
			"medians": medians,
			"means": means,
			"stds": stds,
			"hidden_layers": best_config.hidden_layers,
			"dropout": best_config.dropout,
			"threshold": best_threshold,
		},
		model_out_path,
	)

	meta = {
		"train_rows": int(len(features)),
		"train_positive": int(targets.sum()),
		"train_negative": int(len(targets) - targets.sum()),
		"split": {
			"train_rows": int(len(train_indices)),
			"validation_rows": int(len(val_indices)),
			"test_rows": int(len(test_indices)),
			"val_ratio": args.val_ratio,
			"test_ratio": args.test_ratio,
		},
		"validation_metrics": best_metrics,
		"test_metrics": test_metrics,
		"decision_threshold": best_threshold,
		"best_config": {
			"hidden_layers": list(best_config.hidden_layers),
			"dropout": best_config.dropout,
			"learning_rate": best_config.learning_rate,
			"weight_decay": best_config.weight_decay,
		},
		"device": str(device),
	}
	with meta_out_path.open("w", encoding="utf-8") as outfile:
		json.dump(meta, outfile, indent=2)

	predict_csv(
		model=best_model,
		input_path=future_path,
		output_path=future_out_path,
		feature_columns=feature_columns,
		medians=medians,
		means=means,
		stds=stds,
		threshold=best_threshold,
		device=device,
	)

	print(f"Best config: {best_config}")
	print(
		"Best validation metrics:",
		f"accuracy={best_metrics['accuracy']:.4f},",
		f"precision={best_metrics['precision']:.4f},",
		f"recall={best_metrics['recall']:.4f},",
		f"f1={best_metrics['f1']:.4f}",
	)
	print(f"Best decision threshold: {best_threshold:.3f}")
	print(
		"Held-out test metrics:",
		f"accuracy={test_metrics['accuracy']:.4f},",
		f"precision={test_metrics['precision']:.4f},",
		f"recall={test_metrics['recall']:.4f},",
		f"f1={test_metrics['f1']:.4f}",
	)
	print(f"Saved model: {model_out_path}")
	print(f"Saved metadata: {meta_out_path}")
	if future_path.exists():
		print(f"Saved future predictions: {future_out_path}")


if __name__ == "__main__":
	main()
