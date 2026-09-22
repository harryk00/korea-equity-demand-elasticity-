from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import pandas as pd

from .provider_base import MarketDataError, MarketDataProvider


def _yyyymmdd(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _normalise_ticker(ticker: Any) -> str:
    return str(ticker).strip().zfill(6)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join(str(x) for x in tup if str(x) != "") for tup in out.columns]
    return out


def _find_column(columns, candidates: list[str]) -> str | None:
    cols = [str(c).strip() for c in columns]
    exact = {c: c for c in cols}
    for cand in candidates:
        if cand in exact:
            return exact[cand]
    # pykrx releases sometimes append units/sub-labels. Prefer unique substring.
    for cand in candidates:
        matches = [c for c in cols if cand in c]
        if len(matches) == 1:
            return matches[0]
    return None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


@dataclass
class PykrxMarketProvider(MarketDataProvider):
    """Strict pykrx adapter.

    Rules
    -----
    1. Price, volume and listed-share failures are hard failures.
    2. Trading value may be reconstructed as close * volume only when missing.
    3. Partial ticker batches are never silently written by the caller.
    """

    pause_seconds: float = 0.7
    stock_module: Any | None = None

    @property
    def stock(self):
        if self.stock_module is not None:
            return self.stock_module
        try:
            from pykrx import stock
        except Exception as exc:
            raise MarketDataError(
                "Could not import pykrx. Run: python -m pip install pykrx==1.2.8"
            ) from exc
        self.stock_module = stock
        return self.stock_module

    def _pause(self) -> None:
        if self.pause_seconds > 0:
            time.sleep(self.pause_seconds)

    def ticker_master(self, date) -> pd.DataFrame:
        d = _yyyymmdd(date)
        rows: list[dict[str, Any]] = []
        for market in ("KOSPI", "KOSDAQ", "KONEX"):
            try:
                self._pause()
                tickers = self.stock.get_market_ticker_list(d, market=market)
            except Exception as exc:
                raise MarketDataError(
                    f"Ticker-master request failed for {market} on {d}: {exc}"
                ) from exc
            if tickers is None:
                raise MarketDataError(
                    f"Ticker-master provider returned None for {market} on {d}."
                )
            for ticker in tickers:
                t = _normalise_ticker(ticker)
                try:
                    name = self.stock.get_market_ticker_name(t)
                except Exception as exc:
                    raise MarketDataError(
                        f"Ticker-name request failed for {t}: {exc}"
                    ) from exc
                rows.append(
                    {
                        "ticker": t,
                        "market": market,
                        "name": name,
                        "asof_date": pd.Timestamp(date).normalize(),
                    }
                )
        if not rows:
            raise MarketDataError("Ticker master returned zero rows.")
        return pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)

    def _get_ohlcv(self, ticker: str, start, end) -> pd.DataFrame:
        try:
            self._pause()
            raw = self.stock.get_market_ohlcv_by_date(
                _yyyymmdd(start), _yyyymmdd(end), ticker=ticker
            )
        except Exception as exc:
            raise MarketDataError(f"OHLCV request failed for {ticker}: {exc}") from exc
        if raw is None or raw.empty:
            raise MarketDataError(f"OHLCV returned no rows for {ticker}.")
        return _flatten_columns(raw.reset_index())

    def _get_cap(self, ticker: str, start, end) -> pd.DataFrame:
        fn = getattr(self.stock, "get_market_cap_by_date", None)
        if fn is None:
            raise MarketDataError(
                "pykrx stock module does not expose get_market_cap_by_date; "
                "listed-share count is required for this research pipeline."
            )
        try:
            self._pause()
            raw = fn(_yyyymmdd(start), _yyyymmdd(end), ticker=ticker)
        except TypeError:
            # Some releases take ticker positionally.
            try:
                raw = fn(_yyyymmdd(start), _yyyymmdd(end), ticker)
            except Exception as exc:
                raise MarketDataError(f"Market-cap request failed for {ticker}: {exc}") from exc
        except Exception as exc:
            raise MarketDataError(f"Market-cap request failed for {ticker}: {exc}") from exc
        if raw is None or raw.empty:
            raise MarketDataError(f"Market-cap/listed-share data returned no rows for {ticker}.")
        return _flatten_columns(raw.reset_index())

    @staticmethod
    def _normalise_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "date": ["날짜", "date", "Date", "index"],
            "open": ["시가", "open", "Open"],
            "high": ["고가", "high", "High"],
            "low": ["저가", "low", "Low"],
            "close": ["종가", "close", "Close"],
            "volume": ["거래량", "volume", "Volume"],
            "trading_value": ["거래대금", "거래대금_", "trading_value", "value"],
        }
        out: dict[str, pd.Series] = {}
        for dest, candidates in mapping.items():
            col = _find_column(raw.columns, candidates)
            if col is not None:
                out[dest] = raw[col]
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(out)
        if missing:
            raise MarketDataError(
                f"OHLCV payload missing canonical columns {sorted(missing)}; "
                f"raw columns={list(raw.columns)}"
            )
        df = pd.DataFrame(out)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        for c in ["open", "high", "low", "close", "volume", "trading_value"]:
            if c in df.columns:
                df[c] = _numeric(df[c])
        if "trading_value" not in df:
            df["trading_value"] = np.nan
        return df

    @staticmethod
    def _normalise_cap(raw: pd.DataFrame) -> pd.DataFrame:
        mapping = {
            "date": ["날짜", "date", "Date", "index"],
            "market_cap": ["시가총액", "market_cap", "market cap"],
            "shares_outstanding": ["상장주식수", "상장주식수_", "listed_shares", "shares_outstanding"],
            "cap_trading_value": ["거래대금", "trading_value", "value"],
        }
        out: dict[str, pd.Series] = {}
        for dest, candidates in mapping.items():
            col = _find_column(raw.columns, candidates)
            if col is not None:
                out[dest] = raw[col]
        if "date" not in out or "shares_outstanding" not in out:
            raise MarketDataError(
                "Market-cap payload must include date and listed-share count; "
                f"raw columns={list(raw.columns)}"
            )
        df = pd.DataFrame(out)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        for c in ["market_cap", "shares_outstanding", "cap_trading_value"]:
            if c in df.columns:
                df[c] = _numeric(df[c])
        return df

    def daily_stock_ohlcv(self, ticker: str, start, end) -> pd.DataFrame:
        ticker = _normalise_ticker(ticker)
        px = self._normalise_ohlcv(self._get_ohlcv(ticker, start, end))
        cap = self._normalise_cap(self._get_cap(ticker, start, end))
        merged = px.merge(cap, on="date", how="left", validate="one_to_one")

        if "cap_trading_value" in merged.columns:
            mask = merged["trading_value"].isna() & merged["cap_trading_value"].notna()
            merged.loc[mask, "trading_value"] = merged.loc[mask, "cap_trading_value"]
            merged = merged.drop(columns=["cap_trading_value"])
        missing_value = merged["trading_value"].isna()
        merged.loc[missing_value, "trading_value"] = (
            merged.loc[missing_value, "close"] * merged.loc[missing_value, "volume"]
        )

        merged["ticker"] = ticker
        required = ["date", "close", "volume", "shares_outstanding"]
        if merged[required].isna().any().any():
            bad = merged.loc[merged[required].isna().any(axis=1), required].head(5)
            raise MarketDataError(
                f"Incomplete core market data for {ticker}; examples={bad.to_dict('records')}"
            )
        if (merged["shares_outstanding"] <= 0).any():
            raise MarketDataError(f"Non-positive shares_outstanding encountered for {ticker}.")

        if "market_cap" not in merged.columns:
            merged["market_cap"] = merged["close"] * merged["shares_outstanding"]
        else:
            m = merged["market_cap"].isna()
            merged.loc[m, "market_cap"] = (
                merged.loc[m, "close"] * merged.loc[m, "shares_outstanding"]
            )

        columns = [
            "ticker", "date", "open", "high", "low", "close", "volume",
            "trading_value", "market_cap", "shares_outstanding",
        ]
        return (
            merged[columns]
            .sort_values("date")
            .drop_duplicates("date", keep="last")
            .reset_index(drop=True)
        )
