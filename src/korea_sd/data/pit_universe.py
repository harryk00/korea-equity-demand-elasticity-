from __future__ import annotations

from dataclasses import dataclass, field
import io
import os
import re
import unicodedata
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests


class PitUniverseError(RuntimeError):
    pass


def _ticker(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def _yyyymmdd(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


@dataclass
class PykrxHistoricalUniverseProvider:
    """Historical KOSDAQ membership provider using pykrx date-aware ticker lists.

    The provider intentionally fails loudly if KRX/pykrx is unavailable. A current
    ticker master must never be substituted for historical membership because that
    would reintroduce survivorship bias.
    """

    pause_seconds: float = 0.20
    retries: int = 3
    stock_module: Any | None = None

    @property
    def stock(self):
        if self.stock_module is not None:
            return self.stock_module
        try:
            from pykrx import stock
        except Exception as exc:  # pragma: no cover - environment dependent
            raise PitUniverseError(
                "Could not import pykrx. Install project dependencies first."
            ) from exc
        self.stock_module = stock
        return self.stock_module

    def snapshot(self, date: Any, market: str = "KOSDAQ") -> list[str]:
        d = _yyyymmdd(date)
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                tickers = self.stock.get_market_ticker_list(d, market=market)
                if tickers is None:
                    raise PitUniverseError(
                        f"pykrx returned None for {market} ticker list on {d}."
                    )
                out = sorted({_ticker(t) for t in tickers if str(t).strip()})
                if self.pause_seconds > 0:
                    time.sleep(self.pause_seconds)
                return out
            except Exception as exc:  # pragma: no cover - live service behavior
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(max(self.pause_seconds, 0.5) * attempt)
        raise PitUniverseError(
            f"Historical ticker-list request failed for {market} on {d}: {last_exc}"
        )



def build_membership_from_listing_tables(
    current_listing: pd.DataFrame,
    delisting: pd.DataFrame,
    start: Any,
    end: Any,
    market: str = "KOSDAQ",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct PIT membership from listing/delisting intervals.

    `current_listing` is a descriptive listing snapshot as of (approximately) end.
    `delisting` is the full historical delisting table available as of the same
    cache date. The function never substitutes today's current universe for past
    membership; instead it uses ListingDate/DelistingDate intervals.
    """
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    if end_ts < start_ts:
        raise PitUniverseError("end must be on or after start")

    cur = current_listing.copy()
    de = delisting.copy()

    cur_alias = {
        "Code": "ticker",
        "Symbol": "ticker",
        "Name": "name",
        "Market": "market",
        "ListingDate": "listing_date",
    }
    de_alias = {
        "Code": "ticker",
        "Symbol": "ticker",
        "Name": "name",
        "Market": "market",
        "SecuGroup": "security_group",
        "Kind": "security_kind",
        "ListingDate": "listing_date",
        "DelistingDate": "delisting_date",
    }
    cur = cur.rename(columns={k: v for k, v in cur_alias.items() if k in cur.columns})
    de = de.rename(columns={k: v for k, v in de_alias.items() if k in de.columns})

    required_cur = {"ticker", "market", "listing_date"}
    required_de = {"ticker", "market", "listing_date", "delisting_date"}
    if required_cur - set(cur.columns):
        raise PitUniverseError(
            f"current listing missing columns: {sorted(required_cur - set(cur.columns))}"
        )
    if required_de - set(de.columns):
        raise PitUniverseError(
            f"delisting table missing columns: {sorted(required_de - set(de.columns))}"
        )

    if "name" not in cur.columns:
        cur["name"] = ""
    if "name" not in de.columns:
        de["name"] = ""

    for frame in (cur, de):
        frame["ticker"] = frame["ticker"].astype(str).str.strip().str.zfill(6)
        frame["market"] = frame["market"].astype(str).str.strip().str.upper()
        frame["listing_date"] = pd.to_datetime(
            frame["listing_date"], errors="coerce"
        ).dt.normalize()

    de["delisting_date"] = pd.to_datetime(
        de["delisting_date"], errors="coerce"
    ).dt.normalize()

    # EQUITY-ONLY universe:
    # KRX delisting history contains not only listed shares but also warrants,
    # subscription rights and other temporary securities. Those often have
    # 8-character short codes such as 0676321C / 1099621D and must never enter
    # an equity stock-selection universe.
    #
    # Ordinary KRX stock short codes are six digits. We require that shape for
    # both active and delisted records. If SecuGroup is available, also require
    # '주권' (stock certificate / listed share).
    cur = cur[
        cur["ticker"].astype(str).str.fullmatch(r"\d{6}", na=False)
    ].copy()

    de = de[
        de["ticker"].astype(str).str.fullmatch(r"\d{6}", na=False)
    ].copy()

    if "security_group" in de.columns:
        sec = de["security_group"].fillna("").astype(str).str.strip()
        de = de[sec.eq("주권")].copy()

    market_upper = str(market).upper()

    # Active at the cache snapshot/end: listing interval ends at requested end.
    active = cur.loc[
        cur["market"].eq(market_upper)
        & cur["listing_date"].notna()
        & cur["listing_date"].le(end_ts),
        ["ticker", "name", "listing_date"],
    ].copy()
    active["delisting_date"] = pd.NaT
    active["interval_source"] = "active_listing_cache"

    # Delisted names whose historical membership intersects the research window.
    hist = de.loc[
        de["market"].eq(market_upper)
        & de["listing_date"].notna()
        & de["delisting_date"].notna()
        & de["listing_date"].le(end_ts)
        & de["delisting_date"].ge(start_ts),
        ["ticker", "name", "listing_date", "delisting_date"],
    ].copy()
    hist["interval_source"] = "delisting_cache"

    intervals = pd.concat([active, hist], ignore_index=True)
    intervals = intervals.drop_duplicates(
        ["ticker", "listing_date", "delisting_date", "interval_source"],
        keep="last",
    ).sort_values(["ticker", "listing_date", "delisting_date"])

    # Only retain intervals that actually intersect the requested window.
    intervals["interval_start"] = intervals["listing_date"].clip(lower=start_ts)
    intervals["interval_end"] = intervals["delisting_date"].fillna(end_ts).clip(upper=end_ts)
    intervals = intervals.loc[
        intervals["interval_start"].le(intervals["interval_end"])
    ].copy()

    # Generate business-day membership. Exchange holidays are harmless here because
    # downstream joins to actual OHLCV trading dates; no signal row is created on a
    # holiday without market data.
    frames: list[pd.DataFrame] = []
    for row in intervals.itertuples(index=False):
        dates = pd.date_range(row.interval_start, row.interval_end, freq="B")
        if len(dates) == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": str(row.ticker).zfill(6),
                }
            )
        )

    if not frames:
        raise PitUniverseError(
            f"No {market_upper} membership intervals intersect {start_ts.date()}~{end_ts.date()}."
        )

    membership = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    intervals = intervals[
        [
            "ticker",
            "name",
            "listing_date",
            "delisting_date",
            "interval_start",
            "interval_end",
            "interval_source",
        ]
    ].reset_index(drop=True)

    return membership, intervals


@dataclass
class FdrKrxCacheHistoricalUniverseProvider:
    """KRX historical-universe reconstruction using FinanceDataReader's KRX cache.

    FinanceDataReader maintains daily KRX descriptive-listing and full delisting
    CSV snapshots on GitHub. This avoids the authenticated/blocked pykrx daily
    ticker-list endpoint while preserving listing and delisting dates.

    For reproducibility the provider finds the most recent cache snapshot on or
    before the requested research end date, then reconstructs every membership
    date from ListingDate/DelistingDate intervals.
    """

    base_url: str = (
        "https://raw.githubusercontent.com/FinanceData/"
        "fdr_krx_data_cache/master/data/listing"
    )
    timeout_seconds: float = 30.0
    lookback_days: int = 30
    session: requests.Session = field(default_factory=requests.Session)

    def _download_csv(self, kind: str, date: pd.Timestamp) -> pd.DataFrame | None:
        tag = pd.Timestamp(date).strftime("%Y-%m-%d")
        url = f"{self.base_url.rstrip('/')}/{kind}/{tag}.csv"
        try:
            r = self.session.get(url, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise PitUniverseError(
                f"KRX cache request failed for {url}: {exc}"
            ) from exc

        if r.status_code == 404:
            return None
        if r.status_code != 200:
            raise PitUniverseError(
                f"KRX cache HTTP {r.status_code} for {url}: {r.text[:300]}"
            )
        try:
            return pd.read_csv(
                io.BytesIO(r.content),
                dtype={"Code": str, "Symbol": str, "ToSymbol": str},
            )
        except Exception as exc:
            raise PitUniverseError(
                f"Could not parse KRX cache CSV {url}: {exc}"
            ) from exc

    def snapshot_tables(
        self,
        end: Any,
    ) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
        end_ts = pd.Timestamp(end).normalize()
        errors: list[str] = []

        for offset in range(self.lookback_days + 1):
            date = end_ts - pd.Timedelta(days=offset)
            try:
                desc = self._download_csv("desc", date)
                if desc is None:
                    continue
                dele = self._download_csv("delisting", date)
                if dele is None:
                    continue
                return date, desc, dele
            except PitUniverseError as exc:
                errors.append(str(exc))

        detail = errors[-1] if errors else "no matching cache files"
        raise PitUniverseError(
            f"No paired desc/delisting KRX cache snapshot found on or before "
            f"{end_ts.date()} within {self.lookback_days} days. Last detail: {detail}"
        )

    def build(
        self,
        start: Any,
        end: Any,
        market: str = "KOSDAQ",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
        cache_date, current_listing, delisting = self.snapshot_tables(end=end)

        end_ts = pd.Timestamp(end).normalize()
        staleness = (end_ts - cache_date).days
        if staleness > 7:
            raise PitUniverseError(
                f"Nearest KRX cache snapshot is {cache_date.date()}, "
                f"{staleness} days before requested end {end_ts.date()}. "
                "Refuse stale universe reconstruction; increase cache availability "
                "or choose an earlier end date."
            )

        membership, intervals = build_membership_from_listing_tables(
            current_listing=current_listing,
            delisting=delisting,
            start=start,
            end=end,
            market=market,
        )
        return membership, intervals, cache_date

@dataclass
class OpenDartHistoricalCorpMapProvider:
    """Recover ticker->corp_code mappings from historical KOSDAQ disclosures.

    OpenDART corpCode.xml is excellent for currently listed names but delisted
    securities may no longer carry a stock_code there. Disclosure search responses
    include corp_code and stock_code as they appeared in the filing period, which
    makes them useful for reconstructing historical mappings.
    """

    api_key: str
    base_url: str = "https://opendart.fss.or.kr/api"
    timeout_seconds: float = 30.0
    page_count: int = 100
    session: requests.Session = field(default_factory=requests.Session)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenDartHistoricalCorpMapProvider":
        key = os.environ.get("OPENDART_API_KEY", "").strip()
        if not key:
            raise PitUniverseError(
                "Set OPENDART_API_KEY before building the historical corp map."
            )
        return cls(api_key=key, **kwargs)

    def _request_page(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        page_no: int,
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/list.json"
        params = {
            "crtfc_key": self.api_key,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "corp_cls": "K",
            "pblntf_ty": "A",  # periodic disclosures
            "last_reprt_at": "Y",
            "sort": "date",
            "sort_mth": "asc",
            "page_no": str(page_no),
            "page_count": str(self.page_count),
        }
        try:
            r = self.session.get(url, params=params, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise PitUniverseError(f"OpenDART disclosure search failed: {exc}") from exc
        if r.status_code != 200:
            raise PitUniverseError(
                f"OpenDART disclosure search HTTP {r.status_code}: {r.text[:400]}"
            )
        try:
            body = r.json()
        except ValueError as exc:
            raise PitUniverseError("OpenDART disclosure search returned non-JSON.") from exc

        status = str(body.get("status", ""))
        if status == "013":  # no data
            return {"list": [], "total_page": 0}
        if status != "000":
            raise PitUniverseError(
                f"OpenDART disclosure search status={status}: {body.get('message')}"
            )
        return body

    @staticmethod
    def _windows(start: Any, end: Any, days: int = 80) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
        lo = pd.Timestamp(start).normalize()
        hi = pd.Timestamp(end).normalize()
        cur = lo
        while cur <= hi:
            chunk_end = min(hi, cur + pd.Timedelta(days=days - 1))
            yield cur, chunk_end
            cur = chunk_end + pd.Timedelta(days=1)

    def search(self, start: Any, end: Any) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for lo, hi in self._windows(start, end):
            first = self._request_page(lo, hi, 1)
            total_page = int(first.get("total_page") or 0)
            page_bodies = [first]
            for page in range(2, total_page + 1):
                page_bodies.append(self._request_page(lo, hi, page))

            for body in page_bodies:
                for item in body.get("list") or []:
                    stock_code = str(item.get("stock_code") or "").strip()
                    corp_code = str(item.get("corp_code") or "").strip()
                    if not stock_code or not corp_code:
                        continue
                    rows.append(
                        {
                            "ticker": _ticker(stock_code),
                            "corp_code": corp_code.zfill(8),
                            "corp_name": str(item.get("corp_name") or "").strip(),
                            "rcept_dt": pd.to_datetime(
                                str(item.get("rcept_dt") or ""),
                                format="%Y%m%d",
                                errors="coerce",
                            ),
                            "report_nm": str(item.get("report_nm") or "").strip(),
                            "map_source": "historical_disclosure",
                        }
                    )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "corp_code",
                    "corp_name",
                    "rcept_dt",
                    "report_nm",
                    "map_source",
                ]
            )

        out = pd.DataFrame(rows)
        out = (
            out.sort_values(["ticker", "rcept_dt"])
            .drop_duplicates(["ticker", "corp_code", "rcept_dt", "report_nm"], keep="last")
            .reset_index(drop=True)
        )
        return out


    def _request_page_flexible(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        page_no: int,
        *,
        corp_cls: str | None = None,
        pblntf_ty: str | None = "A",
        last_reprt_at: str | None = "Y",
    ) -> dict[str, Any]:
        """Disclosure search with optional corporation/disclosure filters."""
        url = f"{self.base_url.rstrip('/')}/list.json"
        params: dict[str, str] = {
            "crtfc_key": self.api_key,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "sort": "date",
            "sort_mth": "desc",
            "page_no": str(page_no),
            "page_count": str(self.page_count),
        }
        if corp_cls:
            params["corp_cls"] = str(corp_cls)
        if pblntf_ty:
            params["pblntf_ty"] = str(pblntf_ty)
        if last_reprt_at:
            params["last_reprt_at"] = str(last_reprt_at)

        try:
            r = self.session.get(url, params=params, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise PitUniverseError(
                f"OpenDART deep disclosure search failed: {exc}"
            ) from exc
        if r.status_code != 200:
            raise PitUniverseError(
                f"OpenDART deep disclosure search HTTP {r.status_code}: "
                f"{r.text[:400]}"
            )
        try:
            body = r.json()
        except ValueError as exc:
            raise PitUniverseError(
                "OpenDART deep disclosure search returned non-JSON."
            ) from exc

        status = str(body.get("status", ""))
        if status == "013":
            return {"list": [], "total_page": 0, "total_count": 0}
        if status != "000":
            raise PitUniverseError(
                f"OpenDART deep disclosure search status={status}: "
                f"{body.get('message')}"
            )
        return body

    @staticmethod
    def _windows_reverse(
        start: Any,
        end: Any,
        days: int = 80,
    ) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
        """Yield <=3-month windows newest-first."""
        lo = pd.Timestamp(start).normalize()
        hi = pd.Timestamp(end).normalize()
        cur_hi = hi
        while cur_hi >= lo:
            cur_lo = max(lo, cur_hi - pd.Timedelta(days=days - 1))
            yield cur_lo, cur_hi
            cur_hi = cur_lo - pd.Timedelta(days=1)

    def search_target_tickers(
        self,
        target_tickers: Iterable[str],
        start: Any,
        end: Any,
        *,
        corp_cls: str | None = None,
        pblntf_ty: str | None = "A",
        last_reprt_at: str | None = "Y",
        window_days: int = 80,
        stop_when_all_found: bool = True,
        progress_label: str = "deep",
    ) -> pd.DataFrame:
        """Recover historical ticker->corp_code mappings for specific tickers.

        OpenDART list.json cannot search by stock_code. We therefore scan disclosure
        windows, but retain only requested historical tickers. Windows are searched
        newest-first and the scan can stop as soon as every target ticker is found.

        This is intended for the small unresolved delisted set, not for ordinary
        full-universe collection.
        """
        wanted = {_ticker(x) for x in target_tickers if _ticker(x)}
        if not wanted:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "corp_code",
                    "corp_name",
                    "rcept_dt",
                    "report_nm",
                    "corp_cls",
                    "map_source",
                ]
            )

        found: dict[str, list[dict[str, Any]]] = {}
        windows = list(self._windows_reverse(start, end, days=window_days))
        total_windows = len(windows)

        for wi, (lo, hi) in enumerate(windows, start=1):
            first = self._request_page_flexible(
                lo,
                hi,
                1,
                corp_cls=corp_cls,
                pblntf_ty=pblntf_ty,
                last_reprt_at=last_reprt_at,
            )
            total_page = int(first.get("total_page") or 0)
            bodies = [first]
            for page in range(2, total_page + 1):
                bodies.append(
                    self._request_page_flexible(
                        lo,
                        hi,
                        page,
                        corp_cls=corp_cls,
                        pblntf_ty=pblntf_ty,
                        last_reprt_at=last_reprt_at,
                    )
                )

            for body in bodies:
                for item in body.get("list") or []:
                    stock_code = _ticker(item.get("stock_code"))
                    corp_code = str(item.get("corp_code") or "").strip()
                    if stock_code not in wanted or not corp_code:
                        continue

                    found.setdefault(stock_code, []).append(
                        {
                            "ticker": stock_code,
                            "corp_code": corp_code.zfill(8),
                            "corp_name": str(
                                item.get("corp_name") or ""
                            ).strip(),
                            "rcept_dt": pd.to_datetime(
                                str(item.get("rcept_dt") or ""),
                                format="%Y%m%d",
                                errors="coerce",
                            ),
                            "report_nm": str(
                                item.get("report_nm") or ""
                            ).strip(),
                            "corp_cls": str(
                                item.get("corp_cls") or ""
                            ).strip(),
                            "map_source": (
                                f"historical_disclosure_{progress_label}"
                            ),
                        }
                    )

            if wi == 1 or wi % 5 == 0 or len(found) == len(wanted):
                print(
                    f"[DART {progress_label}] windows {wi}/{total_windows}, "
                    f"recovered {len(found)}/{len(wanted)} target tickers"
                )

            if stop_when_all_found and len(found) == len(wanted):
                break

        rows = [row for rows_ in found.values() for row in rows_]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "corp_code",
                    "corp_name",
                    "rcept_dt",
                    "report_nm",
                    "corp_cls",
                    "map_source",
                ]
            )

        out = pd.DataFrame(rows)
        return (
            out.sort_values(["ticker", "rcept_dt"])
            .drop_duplicates(
                ["ticker", "corp_code", "rcept_dt", "report_nm"],
                keep="last",
            )
            .reset_index(drop=True)
        )


@dataclass
class PykrxOhlcvOnlyProvider:
    """OHLCV-only pykrx fallback for historical/delisted names.

    Unlike the project's strict PykrxMarketProvider, this fallback does not request
    historical market-cap/listed-share data. That is deliberate: point-in-time
    shares and free float come from DART in this research design.
    """

    pause_seconds: float = 0.5
    stock_module: Any | None = None

    @property
    def stock(self):
        if self.stock_module is not None:
            return self.stock_module
        try:
            from pykrx import stock
        except Exception as exc:  # pragma: no cover
            raise PitUniverseError("Could not import pykrx for market fallback.") from exc
        self.stock_module = stock
        return self.stock_module

    def daily_stock_ohlcv(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        ticker = _ticker(ticker)
        try:
            raw = self.stock.get_market_ohlcv_by_date(
                _yyyymmdd(start), _yyyymmdd(end), ticker=ticker
            )
            if self.pause_seconds > 0:
                time.sleep(self.pause_seconds)
        except Exception as exc:
            raise PitUniverseError(f"pykrx OHLCV fallback failed for {ticker}: {exc}") from exc

        if raw is None or raw.empty:
            raise PitUniverseError(f"pykrx OHLCV fallback returned no rows for {ticker}.")

        df = raw.reset_index().copy()
        colmap: dict[str, str] = {}
        aliases = {
            "date": ["날짜", "date", "Date", "index"],
            "open": ["시가", "open", "Open"],
            "high": ["고가", "high", "High"],
            "low": ["저가", "low", "Low"],
            "close": ["종가", "close", "Close"],
            "volume": ["거래량", "volume", "Volume"],
            "trading_value": ["거래대금", "trading_value", "value"],
        }
        cols = [str(c) for c in df.columns]
        for dest, candidates in aliases.items():
            for cand in candidates:
                if cand in cols:
                    colmap[cand] = dest
                    break
        df = df.rename(columns=colmap)

        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise PitUniverseError(
                f"pykrx OHLCV fallback missing {sorted(missing)} for {ticker}; columns={cols}"
            )

        out = df[[c for c in ["date", "open", "high", "low", "close", "volume", "trading_value"] if c in df.columns]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        for c in ["open", "high", "low", "close", "volume", "trading_value"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        if "trading_value" not in out.columns:
            out["trading_value"] = out["close"] * out["volume"]
        else:
            miss = out["trading_value"].isna()
            out.loc[miss, "trading_value"] = out.loc[miss, "close"] * out.loc[miss, "volume"]
        out["ticker"] = ticker
        out = out[["ticker", "date", "open", "high", "low", "close", "volume", "trading_value"]]
        out = out.dropna(subset=["date", "close", "volume"])
        return (
            out.sort_values("date")
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )


def build_union_summary(membership: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker"}
    missing = required - set(membership.columns)
    if missing:
        raise PitUniverseError(f"membership missing columns: {sorted(missing)}")

    m = membership.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    m["ticker"] = m["ticker"].astype(str).str.zfill(6)
    m = m.dropna(subset=["date"]).drop_duplicates(["date", "ticker"])
    if m.empty:
        raise PitUniverseError("Historical membership is empty.")

    last_snapshot = m["date"].max()
    grouped = (
        m.groupby("ticker", as_index=False)
        .agg(
            first_seen=("date", "min"),
            last_seen=("date", "max"),
            membership_days=("date", "nunique"),
        )
    )
    active = set(m.loc[m["date"] == last_snapshot, "ticker"])
    grouped["active_on_last_snapshot"] = grouped["ticker"].isin(active)
    grouped["exited_before_end"] = ~grouped["active_on_last_snapshot"]
    grouped["last_snapshot_date"] = last_snapshot
    return grouped.sort_values("ticker").reset_index(drop=True)


def resolve_corp_map(
    union: pd.DataFrame,
    current_map: pd.DataFrame,
    historical_map: pd.DataFrame,
) -> pd.DataFrame:
    u = union.copy()
    u["ticker"] = u["ticker"].astype(str).str.zfill(6)

    current = current_map.copy()
    if not current.empty:
        current["ticker"] = current["ticker"].astype(str).str.zfill(6)
        current["corp_code"] = current["corp_code"].astype(str).str.zfill(8)

    hist = historical_map.copy()
    if not hist.empty:
        hist["ticker"] = hist["ticker"].astype(str).str.zfill(6)
        hist["corp_code"] = hist["corp_code"].astype(str).str.zfill(8)

    rows: list[dict[str, Any]] = []
    current_by_ticker = {
        t: g.iloc[-1].to_dict() for t, g in current.groupby("ticker", sort=False)
    } if not current.empty else {}

    hist_groups = {
        t: g.copy() for t, g in hist.groupby("ticker", sort=False)
    } if not hist.empty else {}

    for rec in u.to_dict("records"):
        ticker = rec["ticker"]
        candidates: set[str] = set()
        name = ""
        sources: list[str] = []

        hg = hist_groups.get(ticker)
        if hg is not None and not hg.empty:
            codes = set(hg["corp_code"].dropna().astype(str).str.zfill(8))
            candidates |= codes
            latest = hg.sort_values("rcept_dt").iloc[-1]
            name = str(latest.get("corp_name") or "").strip()
            sources.append("historical_disclosure")

        cur = current_by_ticker.get(ticker)
        if cur is not None:
            code = str(cur.get("corp_code") or "").strip()
            if code:
                candidates.add(code.zfill(8))
            if not name:
                name = str(cur.get("corp_name") or "").strip()
            sources.append("current_corpCode")

        row = dict(rec)
        row["corp_name"] = name
        row["corp_code_candidates"] = ";".join(sorted(candidates))
        row["corp_code_count"] = len(candidates)
        row["corp_code"] = next(iter(candidates)) if len(candidates) == 1 else ""
        if len(candidates) == 0:
            row["corp_map_status"] = "missing"
        elif len(candidates) == 1:
            row["corp_map_status"] = "ok"
        else:
            row["corp_map_status"] = "ambiguous"
        row["corp_map_source"] = "+".join(sorted(set(sources)))
        upper_name = name.upper()
        row["is_spac"] = bool(
            ("스팩" in name)
            or ("SPAC" in upper_name)
            or ("기업인수목적" in name)
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)



_LEGAL_NAME_TOKENS = (
    "주식회사",
    "유한회사",
    "유한책임회사",
    "합자회사",
    "합명회사",
)


def normalize_corp_name(value: Any) -> str:
    """Conservative normalization for KRX-name <-> DART formal-name matching."""
    s = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if not s:
        return ""

    # Common Korean corporate-form representations.
    s = s.replace("㈜", "")
    s = re.sub(r"\(\s*주\s*\)", "", s)
    s = re.sub(r"\(\s*유\s*\)", "", s)
    for token in _LEGAL_NAME_TOKENS:
        s = s.replace(token, "")

    # English corporate-form suffixes; only remove explicit legal-form wording.
    s = re.sub(r"\bCO\.?\s*,?\s*LTD\.?\b", "", s)
    s = re.sub(r"\bCORPORATION\b", "", s)
    s = re.sub(r"\bINCORPORATED\b", "", s)

    # Remove spacing/punctuation only after legal-form stripping.
    s = re.sub(r"[\s\-\._,&·/()'\"`]+", "", s)
    return s


def recover_missing_corp_map_by_name(
    resolved: pd.DataFrame,
    all_corp_codes: pd.DataFrame,
    source_name_col: str = "interval_name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover delisted ticker->corp_code mappings using unique exact company names.

    Safety rules:
    - only rows currently marked `missing` are considered;
    - KRX interval name is normalized conservatively;
    - DART corpCode.xml *all-company* names are normalized the same way;
    - a mapping is accepted only when the normalized name maps to exactly one
      unique DART corp_code;
    - multiple candidates are never guessed.

    This is specifically useful because OpenDART stock_code is populated for listed
    companies, while a delisted filing company can remain in corpCode.xml with a
    corp_code/corp_name but blank stock_code.
    """
    out = resolved.copy()
    allc = all_corp_codes.copy()

    if source_name_col not in out.columns:
        raise PitUniverseError(
            f"resolved corp map missing name column: {source_name_col}"
        )
    required = {"corp_code", "corp_name"}
    missing = required - set(allc.columns)
    if missing:
        raise PitUniverseError(
            f"all_corp_codes missing columns: {sorted(missing)}"
        )

    allc["corp_code"] = allc["corp_code"].astype(str).str.strip().str.zfill(8)
    allc["corp_name"] = allc["corp_name"].fillna("").astype(str).str.strip()
    allc["name_key"] = allc["corp_name"].map(normalize_corp_name)
    allc = allc[
        allc["name_key"].ne("")
        & allc["corp_code"].str.fullmatch(r"\d{8}", na=False)
    ].copy()

    # Build only unique name->corp_code relationships.
    grouped = (
        allc.groupby("name_key", as_index=False)
        .agg(
            candidate_count=("corp_code", "nunique"),
            candidate_codes=(
                "corp_code",
                lambda s: ";".join(sorted(set(s.astype(str)))),
            ),
            dart_names=(
                "corp_name",
                lambda s: ";".join(sorted(set(s.astype(str)))),
            ),
        )
    )
    lookup = grouped.set_index("name_key").to_dict("index")

    audit_rows: list[dict[str, Any]] = []

    for idx, row in out.iterrows():
        if str(row.get("corp_map_status") or "") != "missing":
            continue

        source_name = str(row.get(source_name_col) or "").strip()
        key = normalize_corp_name(source_name)
        hit = lookup.get(key)

        audit = {
            "ticker": str(row.get("ticker") or "").zfill(6),
            "source_name": source_name,
            "normalized_name": key,
            "candidate_count": 0,
            "candidate_codes": "",
            "dart_names": "",
            "recovery_status": "no_name_match",
        }

        if not key or hit is None:
            audit_rows.append(audit)
            continue

        count = int(hit["candidate_count"])
        audit["candidate_count"] = count
        audit["candidate_codes"] = str(hit["candidate_codes"])
        audit["dart_names"] = str(hit["dart_names"])

        if count == 1:
            corp_code = str(hit["candidate_codes"]).zfill(8)
            out.at[idx, "corp_code"] = corp_code
            out.at[idx, "corp_code_candidates"] = corp_code
            out.at[idx, "corp_code_count"] = 1
            out.at[idx, "corp_map_status"] = "ok"

            old_source = str(row.get("corp_map_source") or "").strip()
            sources = [x for x in [old_source, "corpCode_name_exact"] if x]
            out.at[idx, "corp_map_source"] = "+".join(
                sorted(set("+".join(sources).split("+")))
            )

            # Preserve official DART formal name for mapped row if available.
            dart_name = str(hit["dart_names"]).split(";")[0].strip()
            if dart_name:
                out.at[idx, "corp_name"] = dart_name

            audit["recovery_status"] = "recovered_unique_exact_name"
        else:
            # Do not guess among multiple legal entities with the same normalized name.
            audit["recovery_status"] = "ambiguous_name_match"

        audit_rows.append(audit)

    audit_df = pd.DataFrame(audit_rows)
    return out.sort_values("ticker").reset_index(drop=True), audit_df

def trim_market_to_observable_free_float(
    market: pd.DataFrame,
    dart: pd.DataFrame,
    distributed_shares_col: str = "distributed_shares",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep each ticker only from its first *observable* positive free-float snapshot.

    This is less wasteful and more PIT-correct than excluding an entire newly listed
    ticker merely because no free-float snapshot existed on its first trading day.
    """

    m = market.copy()
    d = dart.copy()
    m["ticker"] = m["ticker"].astype(str).str.zfill(6)
    d["ticker"] = d["ticker"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()
    d["asof_date"] = pd.to_datetime(d["asof_date"], errors="coerce").dt.normalize()
    d[distributed_shares_col] = pd.to_numeric(
        d[distributed_shares_col], errors="coerce"
    )

    valid = d.loc[
        d["asof_date"].notna()
        & d[distributed_shares_col].gt(0),
        ["ticker", "asof_date"],
    ].copy()
    first_ff = valid.groupby("ticker")["asof_date"].min().to_dict()

    kept: list[pd.DataFrame] = []
    report: list[dict[str, Any]] = []

    for ticker, g in m.groupby("ticker", sort=False):
        g = g.sort_values("date")
        first_market = g["date"].min()
        last_market = g["date"].max()
        ff_start = first_ff.get(ticker, pd.NaT)
        before = len(g)

        if pd.isna(ff_start):
            after_g = g.iloc[0:0].copy()
            status = "no_valid_free_float"
        else:
            after_g = g[g["date"] >= pd.Timestamp(ff_start)].copy()
            status = "ok" if len(after_g) else "free_float_after_market_end"

        if len(after_g):
            kept.append(after_g)

        report.append(
            {
                "ticker": ticker,
                "first_market_date": first_market,
                "last_market_date": last_market,
                "first_valid_free_float_asof": ff_start,
                "rows_before_trim": before,
                "rows_after_trim": len(after_g),
                "rows_trimmed": before - len(after_g),
                "coverage_ratio": len(after_g) / before if before else np.nan,
                "status": status,
            }
        )

    out = (
        pd.concat(kept, ignore_index=True)
        if kept
        else m.iloc[0:0].copy()
    )
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out, pd.DataFrame(report)
