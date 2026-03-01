# PolyPredict

PolyPredict is a Django website that displays a formatted table of predictions from `data/future_with_predictions.csv`, sorted by model confidence.

## Run locally

From the repository root:

```bash
cd website
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`.

## Notes

- Table rows are ranked by predicted confidence (higher first).
- Market links use `Market Link` from the CSV when available.
- If no direct link is available, the site falls back to Polymarket URL fields and then a slug-based market URL.
