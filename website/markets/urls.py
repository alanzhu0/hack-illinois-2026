from django.urls import path

from markets.views import grouped_markets, market_table


urlpatterns = [
	path("", market_table, name="market-table"),
	path("markets/", grouped_markets, name="grouped-markets"),
]
