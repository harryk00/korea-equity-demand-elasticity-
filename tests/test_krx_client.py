import pandas as pd
import pytest

from korea_sd.data.krx_client import PykrxMarketProvider
from korea_sd.data.provider_base import MarketDataError


class FakeStock:
    def get_market_ohlcv_by_date(self, start, end, ticker=None):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame(
            {
                "시가": [100, 110],
                "고가": [120, 115],
                "저가": [95, 105],
                "종가": [110, 112],
                "거래량": [1000, 1200],
            },
            index=idx,
        )

    def get_market_cap_by_date(self, start, end, ticker=None):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        return pd.DataFrame(
            {
                "시가총액": [110_000_000, 112_000_000],
                "거래량": [1000, 1200],
                "거래대금": [109_000, 134_000],
                "상장주식수": [1_000_000, 1_000_000],
            },
            index=idx,
        )


def test_krx_normalises_and_reconstructs_trading_value():
    p = PykrxMarketProvider(pause_seconds=0, stock_module=FakeStock())
    out = p.daily_stock_ohlcv("5930", "2024-01-01", "2024-01-05")
    assert out["ticker"].unique().tolist() == ["005930"]
    assert out["shares_outstanding"].tolist() == [1_000_000, 1_000_000]
    # OHLCV lacks trading value, but market-cap endpoint supplies it.
    assert out["trading_value"].tolist() == [109_000, 134_000]


class MissingShares(FakeStock):
    def get_market_cap_by_date(self, start, end, ticker=None):
        return pd.DataFrame({"시가총액": [1]}, index=pd.to_datetime(["2024-01-02"]))


def test_krx_rejects_missing_listed_shares():
    p = PykrxMarketProvider(pause_seconds=0, stock_module=MissingShares())
    with pytest.raises(MarketDataError):
        p.daily_stock_ohlcv("005930", "2024-01-01", "2024-01-05")
