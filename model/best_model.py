import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import model as train_lib


DEFAULT_BEST_CONFIG = train_lib.TrainConfig(
	hidden_layers=(192, 96, 48),
	activation="silu",
	use_batch_norm=True,
	dropout=0.2,
	learning_rate=8e-4,
	weight_decay=1.5e-3,
)

DEFAULT_PREPROCESS_CONFIG = train_lib.PreprocessConfig(
	scaler="standard",
	clip_quantile=0.99,
)


def load_best_settings(meta_path: Path) -> tuple[train_lib.TrainConfig, train_lib.PreprocessConfig]:
	if not meta_path.exists():
		return DEFAULT_BEST_CONFIG, DEFAULT_PREPROCESS_CONFIG

	with meta_path.open("r", encoding="utf-8") as infile:
		meta = json.load(infile)

	best_config_payload = meta.get("best_config", {})
	best_preprocess_payload = meta.get("best_preprocessing", {})

	try:
		best_config = train_lib.TrainConfig(
			hidden_layers=tuple(int(x) for x in best_config_payload.get("hidden_layers", DEFAULT_BEST_CONFIG.hidden_layers)),
			activation=str(best_config_payload.get("activation", DEFAULT_BEST_CONFIG.activation)),
			use_batch_norm=bool(best_config_payload.get("use_batch_norm", DEFAULT_BEST_CONFIG.use_batch_norm)),
			dropout=float(best_config_payload.get("dropout", DEFAULT_BEST_CONFIG.dropout)),
			learning_rate=float(best_config_payload.get("learning_rate", DEFAULT_BEST_CONFIG.learning_rate)),
			weight_decay=float(best_config_payload.get("weight_decay", DEFAULT_BEST_CONFIG.weight_decay)),
		)
	except (TypeError, ValueError):
		best_config = DEFAULT_BEST_CONFIG

	try:
		best_preprocess = train_lib.PreprocessConfig(
			scaler=str(best_preprocess_payload.get("scaler", DEFAULT_PREPROCESS_CONFIG.scaler)),
			clip_quantile=float(best_preprocess_payload.get("clip_quantile", DEFAULT_PREPROCESS_CONFIG.clip_quantile)),
		)
	except (TypeError, ValueError):
		best_preprocess = DEFAULT_PREPROCESS_CONFIG

	return best_config, best_preprocess


def confidence_accuracy_table(
	probabilities: np.ndarray,
	targets: np.ndarray,
	decision_threshold: float,
	start: float,
	end: float,
	step: float,
) -> List[Dict[str, float]]:
	predicted_positive = probabilities >= decision_threshold
	predicted_labels = predicted_positive.astype(np.float32)
	confidence = np.where(predicted_positive, probabilities, 1.0 - probabilities)
	correct = (predicted_labels == targets)

	base_thresholds = np.arange(start, end + (step * 0.5), step)
	fine_mid_thresholds = np.arange(0.90, 0.941, 0.01)
	fine_tail_thresholds = np.arange(0.96, 1.001, 0.01)
	thresholds = np.unique(np.round(np.concatenate([base_thresholds, fine_mid_thresholds, fine_tail_thresholds]), 4))
	thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
	rows: List[Dict[str, float]] = []
	n_total = max(1, len(targets))

	for threshold in thresholds:
		selected = confidence >= threshold
		n_selected = int(selected.sum())
		if n_selected > 0:
			accuracy = float(correct[selected].mean())
		else:
			accuracy = float("nan")

		rows.append(
			{
				"confidence_threshold": float(threshold),
				"selected": float(n_selected),
				"coverage": float(n_selected / n_total),
				"accuracy": accuracy,
			}
		)

	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Train only the selected best model and report confidence-filtered accuracy")
	parser.add_argument("--train", default="../data/train.csv", help="Path to training CSV")
	parser.add_argument("--future", default="../data/future.csv", help="Path to future CSV for inference")
	parser.add_argument("--future-out", default="../data/future_with_predictions.csv", help="Output CSV for future predictions")
	parser.add_argument("--model-out", default="../models/insider_binary_best_only.pt", help="Path to save trained model checkpoint")
	parser.add_argument("--meta-out", default="best_model_meta.json", help="Path to save metadata")
	parser.add_argument("--source-meta", default="model_meta.json", help="Optional metadata with best config/preprocessing")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--epochs", type=int, default=200)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--val-ratio", type=float, default=0.15)
	parser.add_argument("--test-ratio", type=float, default=0.15)
	parser.add_argument("--curve-start", type=float, default=0.50)
	parser.add_argument("--curve-end", type=float, default=0.95)
	parser.add_argument("--curve-step", type=float, default=0.05)
	args = parser.parse_args()

	if args.curve_step <= 0:
		raise ValueError("curve-step must be > 0")
	if args.curve_start <= 0 or args.curve_end > 1 or args.curve_start > args.curve_end:
		raise ValueError("curve thresholds must satisfy 0 < start <= end <= 1")

	base_dir = Path(__file__).resolve().parent
	train_path = (base_dir / args.train).resolve() if not Path(args.train).is_absolute() else Path(args.train)
	future_path = (base_dir / args.future).resolve() if not Path(args.future).is_absolute() else Path(args.future)
	future_out_path = (base_dir / args.future_out).resolve() if not Path(args.future_out).is_absolute() else Path(args.future_out)
	model_out_path = (base_dir / args.model_out).resolve() if not Path(args.model_out).is_absolute() else Path(args.model_out)
	meta_out_path = (base_dir / args.meta_out).resolve() if not Path(args.meta_out).is_absolute() else Path(args.meta_out)
	source_meta_path = (base_dir / args.source_meta).resolve() if not Path(args.source_meta).is_absolute() else Path(args.source_meta)

	train_lib.set_seed(args.seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	best_config, best_preprocess = load_best_settings(source_meta_path)

	features, targets, feature_columns = train_lib.load_csv_dataset(train_path, train_lib.TARGET_COLUMN)
	train_indices, val_indices, test_indices = train_lib.stratified_train_val_test_indices(
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

	train_x, val_x, test_x, medians, means, stds, lower_bounds, upper_bounds = train_lib.impute_and_scale(
		train_x_raw,
		val_x_raw,
		test_x_raw,
		scaler=best_preprocess.scaler,
		clip_quantile=best_preprocess.clip_quantile,
	)
	if test_x is None:
		raise RuntimeError("Preprocessing failed to produce test features")

	best_model, validation_metrics, decision_threshold = train_lib.train_one_model(
		train_x=train_x,
		train_y=train_y,
		val_x=val_x,
		val_y=val_y,
		config=best_config,
		device=device,
		seed=args.seed,
		epochs=args.epochs,
		batch_size=args.batch_size,
		threshold_objective="accuracy",
	)

	test_dataset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
	_, test_metrics = train_lib.evaluate(best_model, test_loader, device, threshold=decision_threshold)

	_, test_logits, test_targets = train_lib.collect_logits_and_targets(best_model, test_loader, device)
	test_probabilities = torch.sigmoid(test_logits).numpy()
	test_targets_np = test_targets.numpy()

	confidence_accuracy = confidence_accuracy_table(
		probabilities=test_probabilities,
		targets=test_targets_np,
		decision_threshold=decision_threshold,
		start=args.curve_start,
		end=args.curve_end,
		step=args.curve_step,
	)

	torch.save(
		{
			"state_dict": best_model.state_dict(),
			"feature_columns": feature_columns,
			"medians": medians,
			"means": means,
			"stds": stds,
			"lower_bounds": lower_bounds,
			"upper_bounds": upper_bounds,
			"hidden_layers": best_config.hidden_layers,
			"activation": best_config.activation,
			"use_batch_norm": best_config.use_batch_norm,
			"dropout": best_config.dropout,
			"scaler": best_preprocess.scaler,
			"clip_quantile": best_preprocess.clip_quantile,
			"threshold": decision_threshold,
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
		"validation_metrics": validation_metrics,
		"test_metrics": test_metrics,
		"decision_threshold": decision_threshold,
		"best_config": {
			"hidden_layers": list(best_config.hidden_layers),
			"activation": best_config.activation,
			"use_batch_norm": best_config.use_batch_norm,
			"dropout": best_config.dropout,
			"learning_rate": best_config.learning_rate,
			"weight_decay": best_config.weight_decay,
		},
		"best_preprocessing": {
			"scaler": best_preprocess.scaler,
			"clip_quantile": best_preprocess.clip_quantile,
		},
		"confidence_accuracy": confidence_accuracy,
		"device": str(device),
	}
	with meta_out_path.open("w", encoding="utf-8") as outfile:
		json.dump(meta, outfile, indent=2)

	train_lib.predict_csv(
		model=best_model,
		input_path=future_path,
		output_path=future_out_path,
		feature_columns=feature_columns,
		medians=medians,
		means=means,
		stds=stds,
		lower_bounds=lower_bounds,
		upper_bounds=upper_bounds,
		threshold=decision_threshold,
		device=device,
	)

	print(f"Best config used: {best_config}")
	print(f"Best preprocessing used: {best_preprocess}")
	print(
		"Validation metrics:",
		f"accuracy={validation_metrics['accuracy']:.4f},",
		f"precision={validation_metrics['precision']:.4f},",
		f"recall={validation_metrics['recall']:.4f},",
		f"f1={validation_metrics['f1']:.4f}",
	)
	print(
		"Held-out test metrics:",
		f"accuracy={test_metrics['accuracy']:.4f},",
		f"precision={test_metrics['precision']:.4f},",
		f"recall={test_metrics['recall']:.4f},",
		f"f1={test_metrics['f1']:.4f}",
	)
	print(f"Decision threshold: {decision_threshold:.3f}")
	print("\nAccuracy by confidence threshold (on test split):")
	print(" threshold | selected | coverage | accuracy")
	for row in confidence_accuracy:
		accuracy_text = f"{row['accuracy']:.4f}" if not np.isnan(row["accuracy"]) else "n/a"
		print(
			f"   {row['confidence_threshold']:.2f}   |"
			f"   {int(row['selected'])}   |"
			f"  {row['coverage']:.3f}   |"
			f"  {accuracy_text}"
		)
	print(f"Saved model: {model_out_path}")
	print(f"Saved metadata: {meta_out_path}")
	if future_path.exists():
		print(f"Saved future predictions: {future_out_path}")


if __name__ == "__main__":
	main()
