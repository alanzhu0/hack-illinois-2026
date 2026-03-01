import csv
import re
import shutil
from pathlib import Path
from typing import Dict, List


DROP_COLUMNS = {
	"Name",
	"Market",
	"Active Market",
	"Active Hold",
	"Tx Hash",
	"Polymarket",
	"Profile",
	"Market Link",
}

OUTPUT_DROP_COLUMNS = DROP_COLUMNS

DURATION_MULTIPLIER_MINUTES = {
	"m": 1,
	"h": 60,
	"d": 1440,
}

MAGNITUDE_MULTIPLIERS = {
	"k": 1_000,
	"m": 1_000_000,
	"b": 1_000_000_000,
}

MISSING_VALUE_TOKENS = {
	"n/a",
	"na",
	"none",
	"null",
	"nan",
	"-",
	"--",
}

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"


def clean_header(header: str) -> str:
	cleaned = re.sub(r"[\r\n]+", " ", header or "")
	cleaned = re.sub(r"\s+", " ", cleaned)
	return cleaned.strip()


def format_number(value: float) -> str:
	if value.is_integer():
		return str(int(value))
	return f"{value:.10f}".rstrip("0").rstrip(".")


def normalize_yes_no(value: str) -> str:
	text = value.strip().lower()
	if text == "1":
		return "1"
	if text == "0":
		return "0"
	if text == "yes":
		return "1"
	if text == "no":
		return "0"
	return ""


def convert_value(value: str) -> str:
	if value is None:
		return ""

	text = value.strip()
	if text == "":
		return ""

	if text.lower() in MISSING_VALUE_TOKENS:
		return ""

	parenthesized_negative = text.startswith("(") and text.endswith(")")
	if parenthesized_negative:
		text = text[1:-1].strip()

	text = text.replace(",", "")

	text = text.replace("$", "")

	yes_no = normalize_yes_no(text)
	if yes_no != "":
		return yes_no

	percent_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)%", text)
	if percent_match:
		percent = float(percent_match.group(1)) / 100.0
		return format_number(percent)

	duration_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([mhd])", text, flags=re.IGNORECASE)
	if duration_match:
		amount = float(duration_match.group(1))
		unit = duration_match.group(2).lower()
		minutes = amount * DURATION_MULTIPLIER_MINUTES[unit]
		if parenthesized_negative and minutes > 0:
			minutes *= -1
		return format_number(minutes)

	magnitude_match = re.fullmatch(rf"({NUMBER_PATTERN})([kmb])", text, flags=re.IGNORECASE)
	if magnitude_match:
		amount = float(magnitude_match.group(1))
		unit = magnitude_match.group(2).lower()
		numeric_value = amount * MAGNITUDE_MULTIPLIERS[unit]
		if parenthesized_negative and numeric_value > 0:
			numeric_value *= -1
		return format_number(numeric_value)

	number_match = re.fullmatch(NUMBER_PATTERN, text, flags=re.IGNORECASE)
	if number_match:
		numeric_value = float(number_match.group(0))
		if parenthesized_negative and numeric_value > 0:
			numeric_value *= -1
		return format_number(numeric_value)

	return text


def resolve_input_path(repo_root: Path, scrape_dir: Path) -> Path:
	candidate_paths = [
		scrape_dir / "insider_data.csv",
		repo_root / "insider_data.csv",
	]

	for candidate in candidate_paths:
		if candidate.exists():
			return candidate

	raise FileNotFoundError(
		f"Could not find insider_data.csv in either {scrape_dir} or {repo_root}"
	)


def preprocess_csv(input_path: Path, output_path: Path, train_path: Path, future_path: Path) -> None:
	with input_path.open("r", newline="", encoding="utf-8-sig") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError("CSV has no header row")

		cleaned_fieldnames = [clean_header(name) for name in reader.fieldnames]
		reader.fieldnames = cleaned_fieldnames

		model_fieldnames = [name for name in cleaned_fieldnames if name not in OUTPUT_DROP_COLUMNS]
		future_fieldnames = cleaned_fieldnames

		processed_rows_full: List[Dict[str, str]] = []
		processed_rows_model: List[Dict[str, str]] = []
		for raw_row in reader:
			clean_row = {clean_header(key): value for key, value in raw_row.items()}
			processed_row_full: Dict[str, str] = {}
			for key in cleaned_fieldnames:
				processed_row_full[key] = convert_value(clean_row.get(key, ""))
			processed_rows_full.append(processed_row_full)
			processed_rows_model.append({key: processed_row_full[key] for key in model_fieldnames})

	with output_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=model_fieldnames)
		writer.writeheader()
		writer.writerows(processed_rows_model)

	train_rows = [row for row in processed_rows_model if (row.get("Outcome") or "").strip() != ""]
	future_rows = [row for row in processed_rows_full if (row.get("Outcome") or "").strip() == ""]

	with train_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=model_fieldnames)
		writer.writeheader()
		writer.writerows(train_rows)

	with future_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=future_fieldnames)
		writer.writeheader()
		writer.writerows(future_rows)

	print(f"Wrote {len(processed_rows_model)} rows to {output_path}")
	print(f"Wrote {len(train_rows)} rows to {train_path}")
	print(f"Wrote {len(future_rows)} rows to {future_path}")


def main() -> None:
	repo_root = Path(__file__).resolve().parent.parent
	model_dir = repo_root / "model"
	data_dir = repo_root / "data"
	scrape_dir = repo_root / "scrape"

	input_path = resolve_input_path(repo_root, scrape_dir)
	output_path = data_dir / "data_cleaned.csv"
	train_path = data_dir / "train.csv"
	future_path = data_dir / "future.csv"
	preprocess_csv(input_path, output_path, train_path, future_path)

	legacy_output_path = model_dir / "data_cleaned.csv"
	legacy_train_path = model_dir / "train.csv"
	legacy_future_path = model_dir / "future.csv"

	shutil.copyfile(output_path, legacy_output_path)
	shutil.copyfile(train_path, legacy_train_path)
	shutil.copyfile(future_path, legacy_future_path)
	print(f"Mirrored outputs to {model_dir}")


if __name__ == "__main__":
	main()
