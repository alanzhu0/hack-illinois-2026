import argparse
import csv
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


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


def parse_outcome_task(link: str, timeout: int, retries: int) -> Tuple[str, str, Optional[Exception]]:
	try:
		outcome = parse_outcome_from_url(link, timeout=timeout, retries=retries)
		return link, outcome, None
	except Exception as error:
		return link, "", error


def update_csv_with_outcomes(
	input_csv: Path,
	output_csv: Path,
	timeout: int,
	retries: int,
	delay_seconds: float,
	workers: int,
) -> None:		
	with input_csv.open("r", newline="", encoding="utf-8-sig") as infile:
		reader = csv.DictReader(infile)
		if not reader.fieldnames:
			raise RuntimeError("CSV has no header row.")
		fieldnames = [re.sub(r"[\r\n]+", " ", name).strip() for name in reader.fieldnames]
		reader.fieldnames = fieldnames
		rows = list(reader)

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

	with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
		future_to_link = {}
		for index, link in tqdm(enumerate(unique_links, start=1), total=total, desc="Submitting tasks"):
			future = executor.submit(parse_outcome_task, link, timeout, retries)
			future_to_link[future] = link
			if delay_seconds > 0 and index < total:
				time.sleep(delay_seconds)

		completed = 0
		for future in tqdm(as_completed(future_to_link), total=total, desc="Processing tasks"):
			completed += 1
			link, outcome, error = future.result()
			outcome_by_link[link] = outcome
			if error is not None:
				print(f"[{completed}/{total}] Failed: {link} -> {error}", flush=True)
			else:
				pass
				# print(f"[{completed}/{total}] Parsed: {link} -> {outcome or 'N/A'}", flush=True)

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
		default=0.01,
		help="Delay (seconds) between task submissions",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=8,
		help="Number of concurrent URL fetch workers",
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
		workers=args.workers,
	)


if __name__ == "__main__":
	main()
