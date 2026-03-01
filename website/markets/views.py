import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.utils.text import slugify


POLYMARKET_GAMMA_BASE = "https://gamma-api.polymarket.com"
POLYMARKET_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
POLYMARKET_PRICE_CACHE_TTL_SECONDS = 15 * 60.0
POLYMARKET_PRICE_CACHE_REFRESH_AHEAD_SECONDS = 90.0
POLYMARKET_PRICE_CACHE_FAILED_RETRY_SECONDS = 20.0
POLYMARKET_PRICE_CACHE_MIN_REFRESH_SECONDS = 10.0
POLYMARKET_PRICE_FETCH_MAX_WORKERS = 4
POLYMARKET_PRICE_FETCH_BATCH_LIMIT = 40

_price_cache_lock = Lock()
_price_cache: dict[str, tuple[float, float, float | None]] = {}
_price_fetch_inflight: set[str] = set()


def _to_float(value: str | None, default: float = 0.0) -> float:
	if value is None:
		return default
	try:
		return float(value)
	except (TypeError, ValueError):
		return default


def _market_link(row: dict[str, str]) -> str:
	market_link = (row.get("Market Link") or "").strip()
	if market_link:
		return market_link

	polymarket_link = (row.get("Polymarket") or "").strip()
	if polymarket_link:
		return polymarket_link

	market_name = (row.get("Market") or "").strip()
	if market_name:
		return f"https://polymarket.com/market/{slugify(market_name)}"

	return "https://polymarket.com"


def _parse_json_array_field(value: object) -> list[object]:
	if isinstance(value, list):
		return value
	if isinstance(value, str):
		try:
			parsed = json.loads(value)
			if isinstance(parsed, list):
				return parsed
		except json.JSONDecodeError:
			return []
	return []


def _extract_market_slug_and_kind(market_url: str) -> tuple[str | None, str | None]:
	if not market_url:
		return None, None

	parsed = urlparse(market_url)
	parts = [part for part in parsed.path.split("/") if part]
	if len(parts) < 2:
		return None, None

	section, slug = parts[0], parts[1]
	if section in {"market", "event"} and slug:
		return slug, section

	return None, None


def _extract_yes_price_from_market_payload(payload: dict[str, object]) -> float | None:
	outcomes = [str(item).strip().lower() for item in _parse_json_array_field(payload.get("outcomes"))]
	prices = _parse_json_array_field(payload.get("outcomePrices"))
	if not outcomes or not prices:
		return None

	try:
		yes_index = outcomes.index("yes")
	except ValueError:
		return None

	if yes_index >= len(prices):
		return None

	try:
		value = float(prices[yes_index])
	except (TypeError, ValueError):
		return None

	return max(0.0, min(1.0, value))


def _fetch_current_yes_price(market_url: str, timeout_seconds: int = 8) -> float | None:
	slug, kind = _extract_market_slug_and_kind(market_url)
	if not slug or not kind:
		return None

	endpoint = (
		f"{POLYMARKET_GAMMA_BASE}/markets?slug={slug}"
		if kind == "market"
		else f"{POLYMARKET_GAMMA_BASE}/events?slug={slug}"
	)

	req = Request(endpoint, headers={"User-Agent": POLYMARKET_USER_AGENT})
	with urlopen(req, timeout=timeout_seconds) as response:
		data = json.loads(response.read().decode("utf-8"))

	if not isinstance(data, list) or not data:
		return None

	if kind == "market":
		if not isinstance(data[0], dict):
			return None
		return _extract_yes_price_from_market_payload(data[0])

	event_payload = data[0]
	if not isinstance(event_payload, dict):
		return None

	markets = event_payload.get("markets")
	if not isinstance(markets, list):
		return None

	for market_payload in markets:
		if isinstance(market_payload, dict):
			price = _extract_yes_price_from_market_payload(market_payload)
			if price is not None:
				return price

	return None


def _fetch_current_yes_price_map(rows: list[dict[str, object]]) -> dict[str, float | None]:
	unique_urls = {
		str(row["market_url"])
		for row in rows
		if row.get("market_url") and str(row["market_url"]).startswith("http")
	}
	if not unique_urls:
		return {}

	now = monotonic()
	price_by_url: dict[str, float | None] = {}
	urls_to_fetch: list[str] = []

	with _price_cache_lock:
		for url in unique_urls:
			cached_entry = _price_cache.get(url)
			if cached_entry is None:
				if url not in _price_fetch_inflight:
					_price_fetch_inflight.add(url)
					urls_to_fetch.append(url)
				continue

			refresh_at, expires_at, cached_price = cached_entry
			if cached_price is not None:
				price_by_url[url] = cached_price

			if expires_at <= now:
				if url not in _price_fetch_inflight:
					_price_fetch_inflight.add(url)
					urls_to_fetch.append(url)
				continue

			if refresh_at <= now:
				if url not in _price_fetch_inflight:
					_price_fetch_inflight.add(url)
					urls_to_fetch.append(url)

	if len(urls_to_fetch) > POLYMARKET_PRICE_FETCH_BATCH_LIMIT:
		with _price_cache_lock:
			for url in urls_to_fetch[POLYMARKET_PRICE_FETCH_BATCH_LIMIT:]:
				_price_fetch_inflight.discard(url)
		urls_to_fetch = urls_to_fetch[:POLYMARKET_PRICE_FETCH_BATCH_LIMIT]

	if not urls_to_fetch:
		return price_by_url

	max_workers = min(POLYMARKET_PRICE_FETCH_MAX_WORKERS, max(1, len(urls_to_fetch)))

	with ThreadPoolExecutor(max_workers=max_workers) as executor:
		futures = {executor.submit(_fetch_current_yes_price, url): url for url in urls_to_fetch}
		for future in as_completed(futures):
			url = futures[future]
			try:
				price = future.result()
			except Exception:
				price = None
			finally:
				with _price_cache_lock:
					_price_fetch_inflight.discard(url)

			fetched_at = monotonic()
			if price is None:
				fallback_price = price_by_url.get(url)
				if fallback_price is not None:
					price_by_url[url] = fallback_price
					with _price_cache_lock:
						_price_cache[url] = (
							fetched_at + POLYMARKET_PRICE_CACHE_FAILED_RETRY_SECONDS,
							fetched_at + POLYMARKET_PRICE_CACHE_FAILED_RETRY_SECONDS,
							fallback_price,
						)
					continue

				price_by_url[url] = None
				with _price_cache_lock:
					_price_cache[url] = (
						fetched_at + POLYMARKET_PRICE_CACHE_FAILED_RETRY_SECONDS,
						fetched_at + POLYMARKET_PRICE_CACHE_FAILED_RETRY_SECONDS,
						None,
					)
				continue

			refresh_in = max(
				POLYMARKET_PRICE_CACHE_MIN_REFRESH_SECONDS,
				POLYMARKET_PRICE_CACHE_TTL_SECONDS - POLYMARKET_PRICE_CACHE_REFRESH_AHEAD_SECONDS,
			)
			refresh_at = fetched_at + min(refresh_in, POLYMARKET_PRICE_CACHE_TTL_SECONDS)
			expires_at = fetched_at + POLYMARKET_PRICE_CACHE_TTL_SECONDS

			price_by_url[url] = price
			with _price_cache_lock:
				_price_cache[url] = (refresh_at, expires_at, price)

	return price_by_url


def _load_prediction_rows() -> list[dict[str, object]]:
	csv_path = Path(settings.BASE_DIR).parent / "data" / "future_with_predictions.csv"
	rows: list[dict[str, object]] = []

	if not csv_path.exists():
		return rows

	with csv_path.open("r", newline="", encoding="utf-8") as infile:
		reader = csv.DictReader(infile)
		for row in reader:
			probability = _to_float(row.get("Predicted Probability"), 0.0)
			predicted_outcome = int(round(_to_float(row.get("Predicted Outcome"), 1.0)))
			confidence = probability if predicted_outcome == 1 else (1.0 - probability)

			rows.append(
				{
					"name": row.get("Name", ""),
					"market": row.get("Market", ""),
					"market_url": _market_link(row),
					"outcome": "YES" if predicted_outcome == 1 else "NO",
					"probability": probability,
					"confidence": confidence,
					"confidence_pct": confidence * 100.0,
					"price": _to_float(row.get("Price"), 0.0),
					"price_usd": f"${_to_float(row.get('Price'), 0.0):,.2f}",
					"size": int(_to_float(row.get("Size"), 0.0)),
					"size_usd": f"${_to_float(row.get('Size'), 0.0):,.0f}",
				}
			)

	return rows


def market_table(request):
	rows = _load_prediction_rows()
	live_yes_price_by_url = _fetch_current_yes_price_map(rows)
	for row in rows:
		market_url = str(row.get("market_url") or "")
		live_yes_price = live_yes_price_by_url.get(market_url)
		row["current_yes_price"] = live_yes_price
		row["current_yes_price_usd"] = (
			f"${live_yes_price:.2f}" if live_yes_price is not None else "n/a"
		)
	rows.sort(key=lambda item: item["confidence"], reverse=True)

	return render(
		request,
		"markets/index.html",
		{
			"rows": rows,
			"row_count": len(rows),
			"last_updated": timezone.now(),
		},
	)


def current_yes_prices_api(request):
	rows = _load_prediction_rows()
	price_by_url = _fetch_current_yes_price_map(rows)
	prices = {
		url: (f"${price:.2f}" if price is not None else None)
		for url, price in price_by_url.items()
	}
	return JsonResponse({"prices": prices})


def _load_model_stats() -> dict[str, object]:
	stats_path = Path(settings.BASE_DIR).parent / "model" / "stats.txt"
	model_dir = Path(settings.BASE_DIR).parent / "model"

	def _coerce_feature_names(value: object) -> list[str]:
		if not isinstance(value, list):
			return []

		feature_names: list[str] = []
		for item in value:
			name = str(item).strip()
			if name:
				feature_names.append(name)
		return feature_names

	def _extract_feature_names(meta_payload: object) -> list[str]:
		if not isinstance(meta_payload, dict):
			return []

		for container_key in ("best_preprocessing", "preprocessing"):
			container = meta_payload.get(container_key)
			if isinstance(container, dict):
				for field_name in ("feature_names", "features", "columns"):
					feature_names = _coerce_feature_names(container.get(field_name))
					if feature_names:
						return feature_names

		for field_name in ("feature_names", "features", "columns"):
			feature_names = _coerce_feature_names(meta_payload.get(field_name))
			if feature_names:
				return feature_names

		return []

	def _load_training_features() -> tuple[list[str], str]:
		meta_candidates = [
			model_dir / "best_model_meta.json",
			model_dir / "model_meta.json",
			model_dir / "binary_classifier_meta.json",
		]

		for meta_path in meta_candidates:
			if not meta_path.exists():
				continue

			try:
				with meta_path.open("r", encoding="utf-8") as infile:
					meta_payload = json.load(infile)
			except (OSError, json.JSONDecodeError):
				continue

			feature_names = _extract_feature_names(meta_payload)
			if feature_names:
				return feature_names, str(meta_path)

		return [], "n/a"

	feature_names, feature_source = _load_training_features()
	if not stats_path.exists():
		return {
			"summary_items": [],
			"threshold_rows": [],
			"artifacts": [],
			"feature_names": feature_names,
			"feature_count": len(feature_names),
			"feature_source": feature_source,
			"stats_path": str(stats_path),
		}

	with stats_path.open("r", encoding="utf-8") as infile:
		lines = [line.rstrip("\n") for line in infile]

	summary_raw: dict[str, str] = {}
	artifacts: list[dict[str, str]] = []
	threshold_rows: list[dict[str, str]] = []
	in_threshold_section = False

	for raw_line in lines:
		line = raw_line.strip()
		if not line:
			continue

		if line.lower().startswith("accuracy by confidence threshold"):
			in_threshold_section = True
			continue

		if in_threshold_section and "|" in line and not line.lower().startswith("threshold"):
			parts = [part.strip() for part in line.split("|")]
			if len(parts) == 4:
				threshold_rows.append(
					{
						"threshold": parts[0],
						"selected": parts[1],
						"coverage": parts[2],
						"accuracy": parts[3],
					}
				)
				continue

		if ":" not in line:
			continue

		key, value = [piece.strip() for piece in line.split(":", 1)]
		if key.startswith("Saved "):
			artifacts.append({"label": key, "value": value})
		else:
			summary_raw[key] = value

	summary_items = [
		{"label": "Best config", "value": summary_raw.get("Best config used", "n/a")},
		{"label": "Best preprocessing", "value": summary_raw.get("Best preprocessing used", "n/a")},
		{"label": "Validation metrics", "value": summary_raw.get("Validation metrics", "n/a")},
		{"label": "Held-out test metrics", "value": summary_raw.get("Held-out test metrics", "n/a")},
		{"label": "Decision threshold", "value": summary_raw.get("Decision threshold", "n/a")},
	]

	return {
		"summary_items": summary_items,
		"threshold_rows": threshold_rows,
		"artifacts": artifacts,
		"feature_names": feature_names,
		"feature_count": len(feature_names),
		"feature_source": feature_source,
		"stats_path": str(stats_path),
	}


def stats_page(request):
	stats_context = _load_model_stats()
	return render(
		request,
		"markets/stats.html",
		{
			"summary_items": stats_context["summary_items"],
			"threshold_rows": stats_context["threshold_rows"],
			"artifacts": stats_context["artifacts"],
			"feature_names": stats_context["feature_names"],
			"feature_count": stats_context["feature_count"],
			"feature_source": stats_context["feature_source"],
			"stats_path": stats_context["stats_path"],
			"last_updated": timezone.now(),
		},
	)


def grouped_markets(request):
	rows = _load_prediction_rows()
	live_yes_price_by_url = _fetch_current_yes_price_map(rows)
	grouped: dict[str, dict[str, object]] = {}

	for row in rows:
		market_name = str(row["market"])
		trader_name = str(row["name"])
		probability = float(row["probability"])
		confidence = float(row["confidence"])

		if market_name not in grouped:
			grouped[market_name] = {
				"market": market_name,
				"market_url": row["market_url"],
				"trader_map": {},
			}

		trader_map: dict[str, dict[str, object]] = grouped[market_name]["trader_map"]
		if trader_name not in trader_map:
			trader_map[trader_name] = {"name": trader_name, "probabilities": [], "confidences": []}

		trader_map[trader_name]["probabilities"].append(probability)
		trader_map[trader_name]["confidences"].append(confidence)

	market_rows: list[dict[str, object]] = []
	for market_data in grouped.values():
		trader_rows: list[dict[str, object]] = []
		trader_confidences: list[float] = []
		weighted_probability_numerator = 0.0
		weighted_probability_denominator = 0.0
		yes_trader_count = 0
		no_trader_count = 0

		for trader_data in market_data["trader_map"].values():
			prob_values = trader_data["probabilities"]
			conf_values = trader_data["confidences"]
			avg_probability = sum(prob_values) / len(prob_values)
			avg_confidence = sum(conf_values) / len(conf_values)
			is_yes_position = avg_probability >= 0.5
			if is_yes_position:
				yes_trader_count += 1
			else:
				no_trader_count += 1
			trader_confidences.append(avg_confidence)
			weight = max(0.01, avg_confidence)
			weighted_probability_numerator += avg_probability * weight
			weighted_probability_denominator += weight

			trader_rows.append(
				{
					"name": trader_data["name"],
					"probability_pct": avg_probability * 100.0,
					"confidence_pct": avg_confidence * 100.0,
					"outcome": "YES" if is_yes_position else "NO",
				}
			)

		trader_rows.sort(key=lambda item: item["confidence_pct"], reverse=True)

		average_confidence = (
			sum(trader_confidences) / len(trader_confidences)
			if trader_confidences
			else 0.0
		)

		consensus_probability = (
			weighted_probability_numerator / weighted_probability_denominator
			if weighted_probability_denominator > 0
			else 0.5
		)
		certainty_score = min(1.0, abs(consensus_probability - 0.5) * 2.0)
		trader_count = len(trader_rows)
		minority_count = min(yes_trader_count, no_trader_count)
		confidence_mass = sum(confidence ** 1.6 for confidence in trader_confidences)
		low_confidence_penalty = sum(max(0.0, 0.65 - confidence) for confidence in trader_confidences)

		market_score = (
			(10.0 * confidence_mass)
			+ (1.0 * trader_count)
			+ (0.8 * certainty_score * trader_count)
			+ (3.2 * average_confidence)
			- (3.0 * low_confidence_penalty)
			- (1.2 * minority_count)
		)
		consensus_outcome = "YES" if consensus_probability >= 0.5 else "NO"

		market_url = str(market_data["market_url"])
		current_yes_price = live_yes_price_by_url.get(market_url)
		market_rows.append(
			{
				"market": market_data["market"],
				"market_url": market_url,
				"trader_count": trader_count,
				"yes_trader_count": yes_trader_count,
				"no_trader_count": no_trader_count,
				"market_score": market_score,
				"consensus_probability": consensus_probability,
				"consensus_outcome": consensus_outcome,
				"current_yes_price": current_yes_price,
				"current_yes_price_usd": f"${current_yes_price:.2f}" if current_yes_price is not None else "n/a",
				"traders": trader_rows,
			}
		)

	market_rows.sort(key=lambda item: item["market_score"], reverse=True)

	return render(
		request,
		"markets/grouped.html",
		{
			"markets": market_rows,
			"market_count": len(market_rows),
			"last_updated": timezone.now(),
		},
	)
