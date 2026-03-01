import csv
import math
from pathlib import Path

from django.conf import settings
from django.shortcuts import render
from django.utils import timezone
from django.utils.text import slugify


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


def grouped_markets(request):
	rows = _load_prediction_rows()
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
		sum_of_squares = 0.0

		for trader_data in market_data["trader_map"].values():
			prob_values = trader_data["probabilities"]
			conf_values = trader_data["confidences"]
			avg_probability = sum(prob_values) / len(prob_values)
			avg_confidence = sum(conf_values) / len(conf_values)
			sum_of_squares += avg_probability * avg_probability

			trader_rows.append(
				{
					"name": trader_data["name"],
					"probability_pct": avg_probability * 100.0,
					"confidence_pct": avg_confidence * 100.0,
					"outcome": "YES" if avg_probability >= 0.5 else "NO",
				}
			)

		trader_rows.sort(key=lambda item: item["confidence_pct"], reverse=True)

		market_score = math.sqrt(sum_of_squares)
		market_rows.append(
			{
				"market": market_data["market"],
				"market_url": market_data["market_url"],
				"trader_count": len(trader_rows),
				"market_score": market_score,
				"market_score_pct": market_score * 100.0,
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
