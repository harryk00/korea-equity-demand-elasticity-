from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .kis_client import KisApiError, KisOpenApiClient


class MicrostructureDataError(RuntimeError):
    """Raised when a requested research panel cannot be built completely."""


def _ticker(value: Any) -> str:
    return str(value).strip().zfill(6)


def _yyyymmdd(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _date_series(values: pd.Series) -> pd.Series:
    # KIS date fields are normally YYYYMMDD strings.
    return pd.to_datetime(values.astype(str).str.strip(), format="%Y%m%d", errors="coerce")


def _numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "None": np.nan, "nan": np.nan})
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce")
    return df


def _records(body: dict[str, Any], key: str) -> pd.DataFrame:
    obj = body.get(key, [])
    if obj is None:
        return pd.DataFrame()
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise MicrostructureDataError(f"Unexpected KIS payload at {key}: {type(obj)!r}")
    return pd.DataFrame(obj)


def _clip(df: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    if df.empty:
        return df
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    return df.loc[df["date"].between(lo, hi)].copy()


def _dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.dropna(subset=["date", "ticker"])
        .sort_values(["ticker", "date"])
        .drop_duplicates(["ticker", "date"], keep="last")
        .reset_index(drop=True)
    )


def _calendar_windows(start: Any, end: Any, days: int = 80) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a range into small windows, safely under KIS 100-row daily limits."""
    lo = pd.Timestamp(start).normalize()
    hi = pd.Timestamp(end).normalize()
    if lo > hi:
        raise ValueError("start must be <= end")

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = lo
    while cur <= hi:
        nxt = min(cur + pd.Timedelta(days=days - 1), hi)
        windows.append((cur, nxt))
        cur = nxt + pd.Timedelta(days=1)
    return windows


@dataclass
class KisMicrostructureProvider:
    """Build canonical daily microstructure panels from KIS Open API.

    Canonical keys are always (ticker, date).  Raw provider field names are
    normalized here so downstream research code is independent of KIS naming.
    """

    client: KisOpenApiClient
    max_anchor_calls: int = 50

    # ------------------------------------------------------------------
    # 1) Stock lending / securities borrowing and lending
    # ------------------------------------------------------------------
    def lending(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        ticker = _ticker(ticker)
        frames: list[pd.DataFrame] = []
        for lo, hi in _calendar_windows(start, end):
            body, _ = self.client.get(
                "/uapi/domestic-stock/v1/quotations/daily-loan-trans",
                "HHPST074500C0",
                {
                    # KIS docs: 1 KOSPI, 2 KOSDAQ, 3 individual issue.
                    "MRKT_DIV_CLS_CODE": "3",
                    "MKSC_SHRN_ISCD": ticker,
                    "START_DATE": _yyyymmdd(lo),
                    "END_DATE": _yyyymmdd(hi),
                    "CTS": "",
                },
            )
            raw = _records(body, "output1")
            if raw.empty:
                continue
            frames.append(self._normalise_lending(raw, ticker))

        if not frames:
            raise MicrostructureDataError(f"Lending returned no rows for {ticker}.")
        return _dedupe_sort(_clip(pd.concat(frames, ignore_index=True), start, end))

    @staticmethod
    def _normalise_lending(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        required = {"bsop_date", "new_stcn", "rdmp_stcn", "rmnd_stcn"}
        missing = required - set(raw.columns)
        if missing:
            raise MicrostructureDataError(f"Lending payload missing columns: {sorted(missing)}")

        raw = _numeric(
            raw.copy(),
            ["new_stcn", "rdmp_stcn", "prdy_rmnd_vrss", "rmnd_stcn", "rmnd_amt", "acml_vol"],
        )
        out = pd.DataFrame(
            {
                "date": _date_series(raw["bsop_date"]),
                "ticker": ticker,
                "lending_new_shares": raw["new_stcn"],
                "lending_return_shares": raw["rdmp_stcn"],
                "lending_net_change_shares": raw.get("prdy_rmnd_vrss"),
                "lending_balance_shares": raw["rmnd_stcn"],
                "lending_balance_value": raw.get("rmnd_amt"),
            }
        )
        # If the provider omits net change, reconstruct from new - return.
        if out["lending_net_change_shares"].isna().all():
            out["lending_net_change_shares"] = (
                out["lending_new_shares"] - out["lending_return_shares"]
            )
        return out

    # ------------------------------------------------------------------
    # 2) Short selling
    # ------------------------------------------------------------------
    def short_selling(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        ticker = _ticker(ticker)
        frames: list[pd.DataFrame] = []
        for lo, hi in _calendar_windows(start, end):
            body, _ = self.client.get(
                "/uapi/domestic-stock/v1/quotations/daily-short-sale",
                "FHPST04830000",
                {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker,
                    "FID_INPUT_DATE_1": _yyyymmdd(lo),
                    "FID_INPUT_DATE_2": _yyyymmdd(hi),
                },
            )
            raw = _records(body, "output2")
            if raw.empty:
                continue
            frames.append(self._normalise_short(raw, ticker))

        if not frames:
            raise MicrostructureDataError(f"Short-selling returned no rows for {ticker}.")
        return _dedupe_sort(_clip(pd.concat(frames, ignore_index=True), start, end))

    @staticmethod
    def _normalise_short(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        required = {"stck_bsop_date", "ssts_cntg_qty", "ssts_tr_pbmn"}
        missing = required - set(raw.columns)
        if missing:
            raise MicrostructureDataError(f"Short payload missing columns: {sorted(missing)}")

        raw = _numeric(
            raw.copy(),
            [
                "stnd_vol_smtn", "ssts_cntg_qty", "ssts_vol_rlim",
                "acml_ssts_cntg_qty", "acml_ssts_cntg_qty_rlim",
                "stnd_tr_pbmn_smtn", "ssts_tr_pbmn", "ssts_tr_pbmn_rlim",
                "acml_ssts_tr_pbmn", "acml_ssts_tr_pbmn_rlim", "avrg_prc",
            ],
        )
        return pd.DataFrame(
            {
                "date": _date_series(raw["stck_bsop_date"]),
                "ticker": ticker,
                "short_volume": raw["ssts_cntg_qty"],
                "short_volume_ratio_pct": raw.get("ssts_vol_rlim"),
                "short_value": raw["ssts_tr_pbmn"],
                "short_value_ratio_pct": raw.get("ssts_tr_pbmn_rlim"),
                "short_avg_price": raw.get("avrg_prc"),
                "short_reference_volume": raw.get("stnd_vol_smtn"),
                "short_reference_value": raw.get("stnd_tr_pbmn_smtn"),
            }
        )

    # ------------------------------------------------------------------
    # 3) Margin credit balance
    # ------------------------------------------------------------------
    def credit(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        ticker = _ticker(ticker)
        frames = self._anchor_backfill(
            ticker=ticker,
            start=start,
            end=end,
            api_path="/uapi/domestic-stock/v1/quotations/daily-credit-balance",
            tr_id="FHPST04760000",
            output_key="output",
            date_field="deal_date",
            params_factory=lambda anchor: {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_COND_SCR_DIV_CODE": "20476",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": _yyyymmdd(anchor),
            },
            normaliser=self._normalise_credit,
        )
        if not frames:
            raise MicrostructureDataError(f"Credit balance returned no rows for {ticker}.")
        return _dedupe_sort(_clip(pd.concat(frames, ignore_index=True), start, end))

    @staticmethod
    def _normalise_credit(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        required = {"deal_date", "whol_loan_new_stcn", "whol_loan_rdmp_stcn", "whol_loan_rmnd_stcn"}
        missing = required - set(raw.columns)
        if missing:
            raise MicrostructureDataError(f"Credit payload missing columns: {sorted(missing)}")

        numeric_cols = [
            "whol_loan_new_stcn", "whol_loan_rdmp_stcn", "whol_loan_rmnd_stcn",
            "whol_loan_new_amt", "whol_loan_rdmp_amt", "whol_loan_rmnd_amt",
            "whol_loan_rmnd_rate", "whol_loan_gvrt",
            "whol_stln_new_stcn", "whol_stln_rdmp_stcn", "whol_stln_rmnd_stcn",
            "whol_stln_new_amt", "whol_stln_rdmp_amt", "whol_stln_rmnd_amt",
            "whol_stln_rmnd_rate", "whol_stln_gvrt",
        ]
        raw = _numeric(raw.copy(), numeric_cols)
        out = pd.DataFrame(
            {
                "date": _date_series(raw["deal_date"]),
                "ticker": ticker,
                "credit_new_shares": raw["whol_loan_new_stcn"],
                "credit_repay_shares": raw["whol_loan_rdmp_stcn"],
                "credit_balance_shares": raw["whol_loan_rmnd_stcn"],
                "credit_new_value": raw.get("whol_loan_new_amt"),
                "credit_repay_value": raw.get("whol_loan_rdmp_amt"),
                "credit_balance_value": raw.get("whol_loan_rmnd_amt"),
                "credit_balance_ratio_pct": raw.get("whol_loan_rmnd_rate"),
                "credit_grant_rate_pct": raw.get("whol_loan_gvrt"),
                # Securities-loan (대주) is different from securities lending (대차),
                # but retaining it is useful as a separate retail short-side control.
                "stock_loan_new_shares": raw.get("whol_stln_new_stcn"),
                "stock_loan_repay_shares": raw.get("whol_stln_rdmp_stcn"),
                "stock_loan_balance_shares": raw.get("whol_stln_rmnd_stcn"),
            }
        )
        out["credit_net_change_shares"] = out["credit_new_shares"] - out["credit_repay_shares"]
        return out

    # ------------------------------------------------------------------
    # 4) Program trading by stock, daily
    # ------------------------------------------------------------------
    def program(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        ticker = _ticker(ticker)
        frames = self._anchor_backfill(
            ticker=ticker,
            start=start,
            end=end,
            api_path="/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily",
            tr_id="FHPPG04650201",
            output_key="output",
            date_field="stck_bsop_date",
            params_factory=lambda anchor: {
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": _yyyymmdd(anchor),
            },
            normaliser=self._normalise_program,
        )
        if not frames:
            raise MicrostructureDataError(f"Program trading returned no rows for {ticker}.")
        return _dedupe_sort(_clip(pd.concat(frames, ignore_index=True), start, end))

    @staticmethod
    def _normalise_program(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        required = {"stck_bsop_date", "whol_smtn_ntby_qty", "whol_smtn_ntby_tr_pbmn"}
        missing = required - set(raw.columns)
        if missing:
            raise MicrostructureDataError(f"Program payload missing columns: {sorted(missing)}")

        numeric_cols = [
            "whol_smtn_seln_vol", "whol_smtn_shnu_vol", "whol_smtn_ntby_qty",
            "whol_smtn_seln_tr_pbmn", "whol_smtn_shnu_tr_pbmn", "whol_smtn_ntby_tr_pbmn",
            "whol_ntby_vol_icdc", "whol_ntby_tr_pbmn_icdc2", "acml_vol", "acml_tr_pbmn",
        ]
        raw = _numeric(raw.copy(), numeric_cols)
        return pd.DataFrame(
            {
                "date": _date_series(raw["stck_bsop_date"]),
                "ticker": ticker,
                "program_sell_volume": raw.get("whol_smtn_seln_vol"),
                "program_buy_volume": raw.get("whol_smtn_shnu_vol"),
                "program_netbuy_volume": raw["whol_smtn_ntby_qty"],
                "program_sell_value": raw.get("whol_smtn_seln_tr_pbmn"),
                "program_buy_value": raw.get("whol_smtn_shnu_tr_pbmn"),
                "program_netbuy_value": raw["whol_smtn_ntby_tr_pbmn"],
                "program_netbuy_volume_change": raw.get("whol_ntby_vol_icdc"),
                "program_netbuy_value_change": raw.get("whol_ntby_tr_pbmn_icdc2"),
            }
        )

    # ------------------------------------------------------------------
    # 5) Historical execution-strength proxy
    # ------------------------------------------------------------------
    def execution_strength(self, ticker: str, start: Any, end: Any) -> pd.DataFrame:
        """Build a backtestable daily execution-strength panel.

        KIS exposes a real-time/current `tday_rltv` execution-strength field, but
        that is not a historical daily series.  For historical research we use
        the official daily buy/sell execution-volume endpoint and compute:

            execution_strength = 100 * total_buy_execution_qty / total_sell_execution_qty

        This avoids look-ahead from reconstructing old values using today's
        real-time endpoint.
        """
        ticker = _ticker(ticker)
        frames: list[pd.DataFrame] = []
        for lo, hi in _calendar_windows(start, end):
            body, _ = self.client.get(
                "/uapi/domestic-stock/v1/quotations/inquire-daily-trade-volume",
                "FHKST03010800",
                {
                    "FID_COND_MRKT_DIV_CODE": "J",
                    "FID_INPUT_ISCD": ticker,
                    "FID_PERIOD_DIV_CODE": "D",
                    "FID_INPUT_DATE_1": _yyyymmdd(lo),
                    "FID_INPUT_DATE_2": _yyyymmdd(hi),
                },
            )
            raw = _records(body, "output2")
            if raw.empty:
                continue
            frames.append(self._normalise_execution(raw, ticker))

        if not frames:
            raise MicrostructureDataError(f"Execution-strength returned no rows for {ticker}.")
        return _dedupe_sort(_clip(pd.concat(frames, ignore_index=True), start, end))

    @staticmethod
    def _normalise_execution(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        required = {"stck_bsop_date", "total_seln_qty", "total_shnu_qty"}
        missing = required - set(raw.columns)
        if missing:
            raise MicrostructureDataError(f"Execution payload missing columns: {sorted(missing)}")

        raw = _numeric(raw.copy(), ["total_seln_qty", "total_shnu_qty"])
        sell = raw["total_seln_qty"].astype(float)
        buy = raw["total_shnu_qty"].astype(float)
        strength = np.where(sell > 0, 100.0 * buy / sell, np.nan)
        return pd.DataFrame(
            {
                "date": _date_series(raw["stck_bsop_date"]),
                "ticker": ticker,
                "buy_execution_volume": buy,
                "sell_execution_volume": sell,
                "execution_strength": strength,
            }
        )

    # ------------------------------------------------------------------
    # Generic anchor-based historical backfill for 30-ish-row APIs
    # ------------------------------------------------------------------
    def _anchor_backfill(
        self,
        *,
        ticker: str,
        start: Any,
        end: Any,
        api_path: str,
        tr_id: str,
        output_key: str,
        date_field: str,
        params_factory: Callable[[pd.Timestamp], dict[str, str]],
        normaliser: Callable[[pd.DataFrame, str], pd.DataFrame],
    ) -> list[pd.DataFrame]:
        lo = pd.Timestamp(start).normalize()
        anchor = pd.Timestamp(end).normalize()
        frames: list[pd.DataFrame] = []
        seen_min_dates: set[pd.Timestamp] = set()

        for _ in range(self.max_anchor_calls):
            if anchor < lo:
                break
            body, _ = self.client.get(api_path, tr_id, params_factory(anchor))
            raw = _records(body, output_key)
            if raw.empty or date_field not in raw.columns:
                break

            parsed_dates = _date_series(raw[date_field]).dropna()
            if parsed_dates.empty:
                raise MicrostructureDataError(
                    f"{api_path} returned rows without parseable {date_field} for {ticker}."
                )
            min_date = parsed_dates.min().normalize()
            if min_date in seen_min_dates:
                break
            seen_min_dates.add(min_date)

            frames.append(normaliser(raw, ticker))
            if min_date <= lo:
                break
            anchor = min_date - pd.Timedelta(days=1)

        return frames
