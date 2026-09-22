from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .kis_client import KisApiError, KisOpenApiClient
from .provider_base import MarketDataError, MarketDataProvider


API_PATH = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
TR_ID = "FHKST03010100"


def _ticker(value: Any) -> str:
    return str(value).strip().zfill(6)


def _yyyymmdd(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


@dataclass
class KisMarketProvider(MarketDataProvider):
    """Historical daily Korean-equity market data from KIS Open API.

    This provider intentionally returns only fields that are historically
    observable from the period-price endpoint. In particular, it does *not*
    copy the current listed-share count from output1 across historical dates.
    Historical shares outstanding/free float should come from point-in-time
    DART or another documented snapshot source.
    """

    client: KisOpenApiClient
    adjusted: bool = True
    chunk_calendar_days: int = 90

    @classmethod
    def from_env(cls, **kwargs: Any) -> "KisMarketProvider":
        client_kwargs = kwargs.pop("client_kwargs", {})
        return cls(client=KisOpenApiClient.from_env(**client_kwargs), **kwargs)

    def ticker_master(self, date) -> pd.DataFrame:
        raise MarketDataError(
            "KisMarketProvider does not implement a full historical ticker master. "
            "Supply a ticker universe separately."
        )

    def _fetch_chunk(self, ticker: str, start, end) -> pd.DataFrame:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": _yyyymmdd(start),
            "FID_INPUT_DATE_2": _yyyymmdd(end),
            "FID_PERIOD_DIV_CODE": "D",
            # Official KIS convention: 0=adjusted price, 1=raw/original price.
            "FID_ORG_ADJ_PRC": "0" if self.adjusted else "1",
        }
        try:
            body, _ = self.client.get(API_PATH, TR_ID, params)
        except KisApiError as exc:
            raise MarketDataError(f"KIS daily-price request failed for {ticker}: {exc}") from exc

        rows = body.get("output2") or []
        if not isinstance(rows, list):
            raise MarketDataError(
                f"KIS daily-price output2 has unexpected type for {ticker}: {type(rows)!r}"
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    @staticmethod
    def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        mapping = {
            "stck_bsop_date": "date",
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_clpr": "close",
            "acml_vol": "volume",
            "acml_tr_pbmn": "trading_value",
        }
        missing = set(mapping) - set(raw.columns)
        if missing:
            raise MarketDataError(
                f"KIS daily-price payload missing columns {sorted(missing)}; "
                f"raw columns={list(raw.columns)}"
            )
        out = raw[list(mapping)].rename(columns=mapping).copy()
        out["date"] = pd.to_datetime(out["date"], format="%Y%m%d", errors="coerce").dt.normalize()
        for col in ["open", "high", "low", "close", "volume", "trading_value"]:
            out[col] = _numeric(out[col])
        out["ticker"] = ticker
        required = ["date", "close", "volume"]
        if out[required].isna().any().any():
            bad = out.loc[out[required].isna().any(axis=1), required].head(5)
            raise MarketDataError(
                f"Incomplete KIS daily market data for {ticker}; examples={bad.to_dict('records')}"
            )
        return out[
            ["ticker", "date", "open", "high", "low", "close", "volume", "trading_value"]
        ]

    def daily_stock_ohlcv(self, ticker: str, start, end) -> pd.DataFrame:
        ticker = _ticker(ticker)
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if end_ts < start_ts:
            raise MarketDataError("end must be on or after start")
        if self.chunk_calendar_days < 1 or self.chunk_calendar_days > 95:
            raise MarketDataError("chunk_calendar_days must be between 1 and 95")

        frames: list[pd.DataFrame] = []
        cursor = start_ts
        # KIS period-price endpoint returns at most 100 rows per request. Using
        # <=95 calendar-day chunks guarantees fewer than 100 trading days.
        while cursor <= end_ts:
            chunk_end = min(
                end_ts,
                cursor + pd.Timedelta(days=self.chunk_calendar_days - 1),
            )
            raw = self._fetch_chunk(ticker, cursor, chunk_end)
            if not raw.empty:
                frames.append(self._normalise(raw, ticker))
            cursor = chunk_end + pd.Timedelta(days=1)

        if not frames:
            raise MarketDataError(
                f"KIS daily market data returned no rows for {ticker} in {start_ts.date()}..{end_ts.date()}."
            )
        out = pd.concat(frames, ignore_index=True)
        out = out.loc[(out["date"] >= start_ts) & (out["date"] <= end_ts)]
        out = (
            out.sort_values("date")
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )
        if out.empty:
            raise MarketDataError(
                f"KIS daily market data had no rows inside requested interval for {ticker}."
            )
        return out
