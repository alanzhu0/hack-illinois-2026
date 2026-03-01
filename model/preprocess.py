import csv
import re
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
		return format_number(minutes)

	magnitude_match = re.fullmatch(r"(-?\d+(?:\.\d+)?)([kmb])", text, flags=re.IGNORECASE)
	if magnitude_match:
		amount = float(magnitude_match.group(1))
		unit = magnitude_match.group(2).lower()
		multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[unit]
		return format_number(amount * multiplier)

	return text


def preprocess_csv(input_path: Path, output_path: Path, train_path: Path, future_path: Path) -> None:
	with input_path.open("r", newline="", encoding="utf-8-sig") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError("CSV has no header row")

		cleaned_fieldnames = [clean_header(name) for name in reader.fieldnames]
		reader.fieldnames = cleaned_fieldnames

		fieldnames = [name for name in cleaned_fieldnames if name not in OUTPUT_DROP_COLUMNS]

		processed_rows: List[Dict[str, str]] = []
		for raw_row in reader:
			clean_row = {clean_header(key): value for key, value in raw_row.items()}
			processed_row: Dict[str, str] = {}
			for key in cleaned_fieldnames:
				if key in OUTPUT_DROP_COLUMNS:
					continue
				processed_row[key] = convert_value(clean_row.get(key, ""))
			processed_rows.append(processed_row)

	with output_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(processed_rows)

	train_rows = [row for row in processed_rows if (row.get("Outcome") or "").strip() != ""]
	future_rows = [row for row in processed_rows if (row.get("Outcome") or "").strip() == ""]

	with train_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(train_rows)

	with future_path.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(future_rows)

	print(f"Wrote {len(processed_rows)} rows to {output_path}")
	print(f"Wrote {len(train_rows)} rows to {train_path}")
	print(f"Wrote {len(future_rows)} rows to {future_path}")


def main() -> None:
	root = Path(__file__).resolve().parent.parent
	input_path = root / "insider_data.csv"
	preprocess_dir = Path(__file__).resolve().parent
	output_path = preprocess_dir / "data_cleaned.csv"
	train_path = preprocess_dir / "train.csv"
	future_path = preprocess_dir / "future.csv"
	preprocess_csv(input_path, output_path, train_path, future_path)


if __name__ == "__main__":
	main()
