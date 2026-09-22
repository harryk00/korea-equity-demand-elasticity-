from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataError(RuntimeError):
    """Raised when a market-data request is incomplete or invalid."""


class MarketDataProvider(ABC):
    """Minimal interface required by the KRX download pipeline."""

    @abstractmethod
    def ticker_master(self, date) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def daily_stock_ohlcv(self, ticker: str, start, end) -> pd.DataFrame:
        raise NotImplementedError
