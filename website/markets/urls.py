from django.urls import path

from markets.views import current_yes_prices_api, grouped_markets, market_table, stats_page


urlpatterns = [
	path("", market_table, name="market-table"),
	path("markets/", grouped_markets, name="grouped-markets"),
	path("stats/", stats_page, name="stats-page"),
	path("api/current-yes-prices/", current_yes_prices_api, name="current-yes-prices-api"),
]
