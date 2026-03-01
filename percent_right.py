import argparse
import csv
from pathlib import Path


def normalize_value(value: str) -> str:
	return " ".join((value or "").strip().lower().split())


def calculate_percent_correct(csv_path: Path) -> tuple[int, int, float]:
	with csv_path.open("r", newline="", encoding="utf-8-sig") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError("CSV has no header row.")

		required = {"Bet", "Outcome"}
		missing = required.difference(reader.fieldnames)
		if missing:
			raise RuntimeError(f"Missing required columns: {', '.join(sorted(missing))}")

		total_with_outcome = 0
		correct = 0

		for row in reader:
			outcome = normalize_value(row.get("Outcome", ""))
			if not outcome:
				continue

			bet = normalize_value(row.get("Bet", ""))
			total_with_outcome += 1
			if bet == outcome:
				correct += 1

	percent = (correct / total_with_outcome * 100) if total_with_outcome else 0.0
	return correct, total_with_outcome, percent


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Calculate what percent of rows with an Outcome value were predicted "
			"correctly (Bet == Outcome)."
		)
	)
	parser.add_argument(
		"--input",
		default="insider_data.csv",
		help="Path to CSV file (default: insider_data.csv)",
	)
	args = parser.parse_args()

	csv_path = Path(args.input)
	if not csv_path.exists():
		raise FileNotFoundError(f"Input CSV not found: {csv_path}")

	correct, total, percent = calculate_percent_correct(csv_path)

	print(f"Correct predictions: {correct}")
	print(f"Rows with outcome:   {total}")
	print(f"Percent correct:     {percent:.2f}%")


if __name__ == "__main__":
	main()
