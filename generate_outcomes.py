import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


USER_AGENT = (
	"Mozilla/5.0 (X11; Linux x86_64) "
	"AppleWebKit/537.36 (KHTML, like Gecko) "
	"Chrome/125.0.0.0 Safari/537.36"
)


def fetch_html(url: str, timeout: int = 20, retries: int = 3) -> str:
	last_error: Optional[Exception] = None
	for attempt in range(1, retries + 1):
		try:
			req = Request(url, headers={"User-Agent": USER_AGENT})
			with urlopen(req, timeout=timeout) as response:
				content_type = response.headers.get_content_charset() or "utf-8"
				return response.read().decode(content_type, errors="replace")
		except (HTTPError, URLError, TimeoutError, OSError) as error:
			last_error = error
			if attempt < retries:
				time.sleep(0.8 * attempt)
	raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def extract_outcome(html: str) -> Optional[str]:
	patterns = [
		r'The winning outcome is\s*([^\.<"\']+)',
		r'Final outcome:\s*([^<\n\r]+)',
		# r'Outcome proposed:\s*([^<\n\r]+)',
	]

	for pattern in patterns:
		match = re.search(pattern, html, flags=re.IGNORECASE)
		if match:
			outcome = match.group(1).strip()
			outcome = re.sub(r"\s+", " ", outcome)
			if outcome:
				return outcome
	return None


def parse_outcome_from_url(url: str, timeout: int, retries: int) -> str:
	if not url:
		return ""

	html = fetch_html(url, timeout=timeout, retries=retries)
	outcome = extract_outcome(html)
	return outcome or ""


def update_csv_with_outcomes(
	input_csv: Path,
	output_csv: Path,
	timeout: int,
	retries: int,
	delay_seconds: float,
) -> None:
	with input_csv.open("r", newline="", encoding="utf-8-sig") as infile:
		reader = csv.DictReader(infile)
		rows = list(reader)
		if not reader.fieldnames:
			raise RuntimeError("CSV has no header row.")
		fieldnames = list(reader.fieldnames)

	link_column = fieldnames[-1]
	outcome_column = "Outcome"
	if outcome_column not in fieldnames:
		fieldnames.append(outcome_column)

	unique_links = []
	seen_links = set()
	for row in rows:
		link = (row.get(link_column) or "").strip()
		if link and link not in seen_links:
			seen_links.add(link)
			unique_links.append(link)

	outcome_by_link: Dict[str, str] = {}
	total = len(unique_links)

	for index, link in enumerate(unique_links, start=1):
		try:
			outcome_by_link[link] = parse_outcome_from_url(link, timeout=timeout, retries=retries)
		except Exception as error:
			outcome_by_link[link] = ""
			print(f"[{index}/{total}] Failed: {link} -> {error}")
		else:
			print(f"[{index}/{total}] Parsed: {link} -> {outcome_by_link[link] or 'N/A'}")

		if delay_seconds > 0:
			time.sleep(delay_seconds)

	for row in rows:
		link = (row.get(link_column) or "").strip()
		row[outcome_column] = outcome_by_link.get(link, "")

	with output_csv.open("w", newline="", encoding="utf-8") as outfile:
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Wrote {len(rows)} rows to {output_csv}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Read insider_data.csv, fetch each row's market link (last column), "
			"extract Polymarket outcome, and write an Outcome column."
		)
	)
	parser.add_argument("--input", default="insider_data.csv", help="Input CSV path")
	parser.add_argument(
		"--output",
		default="insider_data.csv",
		help="Output CSV path (default overwrites input)",
	)
	parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout in seconds")
	parser.add_argument("--retries", type=int, default=3, help="Fetch retries per URL")
	parser.add_argument(
		"--delay",
		type=float,
		default=0.2,
		help="Delay (seconds) between URL fetches",
	)
	args = parser.parse_args()

	input_csv = Path(args.input)
	output_csv = Path(args.output)

	if not input_csv.exists():
		raise FileNotFoundError(f"Input CSV not found: {input_csv}")

	update_csv_with_outcomes(
		input_csv=input_csv,
		output_csv=output_csv,
		timeout=args.timeout,
		retries=args.retries,
		delay_seconds=args.delay,
	)


if __name__ == "__main__":
	main()
