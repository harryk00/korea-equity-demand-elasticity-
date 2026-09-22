from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

import numpy as np
import pandas as pd
import requests


REPORT_CODES = {
    "FY": "11011",
    "ANNUAL": "11011",
    "Q1": "11013",
    "HY": "11012",
    "H1": "11012",
    "Q3": "11014",
}

REPORT_LABELS = {
    "11011": "FY",
    "11013": "Q1",
    "11012": "H1",
    "11014": "Q3",
}


class OpenDartError(RuntimeError):
    """Raised when OpenDART returns an error or unusable share-status data."""


def _num(value: Any) -> float:
    if value is None:
        return np.nan
    text = str(value).replace(",", "").strip()
    if text in {"", "-", "None", "nan"}:
        return np.nan
    return float(pd.to_numeric(text, errors="coerce"))


def _ticker(value: Any) -> str:
    return str(value).strip().zfill(6)


def _receipt_date(rows: pd.DataFrame) -> pd.Timestamp:
    if "rcept_no" not in rows.columns:
        return pd.NaT
    vals = rows["rcept_no"].astype(str).str.extract(r"(\d{8})", expand=False)
    return pd.to_datetime(vals, format="%Y%m%d", errors="coerce").max()


def _period_end(rows: pd.DataFrame) -> pd.Timestamp:
    if "stlm_dt" not in rows.columns:
        return pd.NaT
    return pd.to_datetime(rows["stlm_dt"], errors="coerce").max()


def _normalised_text(value: Any) -> str:
    return str(value or "").replace(" ", "").strip().lower()


@dataclass
class OpenDartClient:
    api_key: str
    base_url: str = "https://opendart.fss.or.kr/api"
    timeout_seconds: float = 20.0
    session: requests.Session = field(default_factory=requests.Session)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenDartClient":
        key = os.environ.get("OPENDART_API_KEY", "").strip()
        if not key:
            raise OpenDartError("Set OPENDART_API_KEY before calling OpenDART.")
        return cls(api_key=key, **kwargs)

    def _get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload = {"crtfc_key": self.api_key, **params}
        try:
            response = self.session.get(url, params=payload, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise OpenDartError(f"OpenDART request failed: {exc}") from exc
        if response.status_code != 200:
            raise OpenDartError(
                f"OpenDART HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise OpenDartError(
                f"OpenDART returned non-JSON: {response.text[:500]}"
            ) from exc
        status = str(body.get("status", "000"))
        if status != "000":
            raise OpenDartError(
                f"OpenDART error status={status}, message={body.get('message')}"
            )
        return body

    @staticmethod
    def report_code(report: str) -> str:
        key = str(report).strip().upper()
        if key in REPORT_CODES:
            return REPORT_CODES[key]
        if key in REPORT_CODES.values():
            return key
        raise ValueError(
            f"Unknown report {report!r}; use one of {sorted(REPORT_CODES)} or a 5-digit code."
        )

    def _stock_total_raw(self, corp_code: str, year: int | str, report_code: str) -> pd.DataFrame:
        body = self._get(
            "stockTotqySttus.json",
            {
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
            },
        )
        rows = body.get("list") or []
        if not isinstance(rows, list) or not rows:
            raise OpenDartError(
                f"OpenDART stock total status returned no rows for corp_code={corp_code}, "
                f"year={year}, report={report_code}."
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_direct_stock_total(raw: pd.DataFrame) -> dict[str, Any] | None:
        needed = {"istc_totqy", "tesstk_co", "distb_stock_co", "stlm_dt"}
        missing = needed - set(raw.columns)
        if missing:
            raise OpenDartError(
                f"OpenDART share-status payload missing columns {sorted(missing)}."
            )

        def numeric(col: str) -> pd.Series:
            if col not in raw.columns:
                return pd.Series(np.nan, index=raw.index, dtype="float64")
            return pd.to_numeric(
                raw[col].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )

        issued_s = numeric("istc_totqy")
        treasury_s = numeric("tesstk_co")
        distributed_s = numeric("distb_stock_co")
        now_issued_s = numeric("now_to_isu_stock_totqy")
        decreased_s = numeric("now_to_dcrs_stock_totqy")

        total_mask = pd.Series(False, index=raw.index)
        if "se" in raw.columns:
            labels = raw["se"].astype(str).str.replace(" ", "", regex=False).str.lower()
            total_mask = labels.str.contains("합계|총계|total", regex=True, na=False)

        issued = treasury = distributed = np.nan
        period_end = _period_end(raw)

        valid_total = total_mask & issued_s.gt(0)
        if valid_total.any():
            idx = valid_total[valid_total].index[-1]
            issued = issued_s.loc[idx]
            treasury = treasury_s.loc[idx]
            distributed = distributed_s.loc[idx]
        else:
            class_mask = ~total_mask
            class_issued = issued_s.loc[class_mask].dropna()
            if not class_issued.empty:
                issued = class_issued.sum()
                tvals = treasury_s.loc[class_mask].dropna()
                dvals = distributed_s.loc[class_mask].dropna()
                treasury = tvals.sum() if not tvals.empty else np.nan
                distributed = dvals.sum() if not dvals.empty else np.nan

        if pd.isna(issued) or issued <= 0:
            candidates = now_issued_s - decreased_s
            candidates = candidates[candidates.gt(0)]
            if not candidates.empty:
                total_candidates = candidates.loc[
                    candidates.index.intersection(raw.index[total_mask])
                ]
                issued = (
                    total_candidates.iloc[-1]
                    if not total_candidates.empty
                    else candidates.max()
                )

        if pd.isna(issued) or issued <= 0:
            return None

        if pd.isna(distributed) and pd.notna(treasury):
            distributed = issued - treasury
        if pd.isna(treasury) and pd.notna(distributed):
            treasury = issued - distributed
        if pd.isna(distributed):
            return None
        if distributed < 0 or distributed > issued:
            raise OpenDartError(
                f"Invalid direct DART share-status arithmetic: issued={issued}, "
                f"distributed={distributed}."
            )

        return {
            "shares_outstanding": float(issued),
            "treasury_shares": float(treasury) if pd.notna(treasury) else np.nan,
            "distributed_shares": float(distributed),
            "report_period_end": pd.Timestamp(period_end).normalize()
            if pd.notna(period_end)
            else pd.NaT,
            "available_date": _receipt_date(raw),
            "share_status_source": "stockTotqySttus",
        }

    def _latest_valid_baseline(
        self, corp_code: str, year: int, target_report_code: str
    ) -> tuple[dict[str, Any], int, str]:
        order = {
            "11013": [(year - 1, "11011")],
            "11012": [(year, "11013"), (year - 1, "11011")],
            "11014": [
                (year, "11012"),
                (year, "11013"),
                (year - 1, "11011"),
            ],
            "11011": [
                (year, "11014"),
                (year, "11012"),
                (year, "11013"),
                (year - 1, "11011"),
            ],
        }
        for candidate_year, candidate_code in order.get(target_report_code, []):
            try:
                raw = self._stock_total_raw(corp_code, candidate_year, candidate_code)
            except OpenDartError:
                continue
            parsed = self._parse_direct_stock_total(raw)
            if parsed is not None:
                return parsed, candidate_year, candidate_code
        raise OpenDartError(
            f"No earlier valid stockTotqySttus baseline found for corp_code={corp_code}, "
            f"year={year}, report={target_report_code}."
        )

    @staticmethod
    def _issuance_event_sign(style: Any) -> int:
        text = _normalised_text(style)
        negative = ("감자", "소각", "상환", "감소", "병합")
        positive = (
            "증자",
            "주식배당",
            "전환권",
            "전환청구",
            "신주인수권",
            "주식매수선택권",
            "분할",
            "합병",
            "교환",
            "증가",
        )
        if any(k in text for k in negative):
            return -1
        if any(k in text for k in positive):
            return 1
        raise OpenDartError(
            f"Cannot infer issuance/decrease sign for DART style={style!r}. "
            "Refusing to guess."
        )

    def _issued_share_delta(
        self,
        corp_code: str,
        year: int,
        report_code: str,
        baseline_period_end: pd.Timestamp,
        target_period_end: pd.Timestamp,
    ) -> tuple[float, list[str]]:
        try:
            body = self._get(
                "irdsSttus.json",
                {
                    "corp_code": corp_code,
                    "bsns_year": str(year),
                    "reprt_code": report_code,
                },
            )
        except OpenDartError as exc:
            raise OpenDartError(
                f"Cannot reconstruct issued shares because irdsSttus failed: {exc}"
            ) from exc

        rows = pd.DataFrame(body.get("list") or [])
        if rows.empty:
            return 0.0, []
        required = {"isu_dcrs_de", "isu_dcrs_stle", "isu_dcrs_qy"}
        missing = required - set(rows.columns)
        if missing:
            raise OpenDartError(f"irdsSttus missing columns {sorted(missing)}.")

        event_dates = pd.to_datetime(rows["isu_dcrs_de"], errors="coerce")
        qty = pd.to_numeric(
            rows["isu_dcrs_qy"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        mask = (
            event_dates.gt(pd.Timestamp(baseline_period_end))
            & event_dates.le(pd.Timestamp(target_period_end))
            & qty.notna()
            & qty.gt(0)
        )
        delta = 0.0
        notes: list[str] = []
        for idx in rows.index[mask]:
            style = rows.at[idx, "isu_dcrs_stle"]
            sign = self._issuance_event_sign(style)
            q = float(qty.loc[idx])
            delta += sign * q
            notes.append(f"{event_dates.loc[idx].date()}:{style}:{sign*q:+.0f}")
        return delta, notes

    def _treasury_at_report_end(
        self, corp_code: str, year: int, report_code: str
    ) -> tuple[float, pd.Timestamp, pd.Timestamp]:
        body = self._get(
            "tesstkAcqsDspsSttus.json",
            {
                "corp_code": corp_code,
                "bsns_year": str(year),
                "reprt_code": report_code,
            },
        )
        rows = pd.DataFrame(body.get("list") or [])
        if rows.empty or "trmend_qy" not in rows.columns:
            raise OpenDartError(
                f"No usable tesstkAcqsDspsSttus rows for corp_code={corp_code}."
            )
        qty = pd.to_numeric(
            rows["trmend_qy"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )

        def _norm_label(series: pd.Series) -> pd.Series:
            return (
                series.fillna("")
                .astype(str)
                .str.replace(" ", "", regex=False)
                .str.lower()
            )

        work = rows.loc[qty.notna()].copy()
        work["_qty"] = qty.loc[work.index]
        if work.empty:
            raise OpenDartError(
                f"tesstkAcqsDspsSttus has no numeric trmend_qy rows for corp_code={corp_code}."
            )

        for col in ("acqs_mth1", "acqs_mth2", "acqs_mth3", "stock_knd"):
            if col not in work.columns:
                work[col] = ""
            work[f"_{col}"] = _norm_label(work[col])

        total_re = r"총계|합계|total|grandtotal"
        subtotal_re = r"소계|subtotal"

        def _class_treasury(group: pd.DataFrame) -> float:
            # 1) Prefer an explicit total at any acquisition-method level.
            total_mask = pd.Series(False, index=group.index)
            for col in ("_acqs_mth3", "_acqs_mth2", "_acqs_mth1"):
                total_mask |= group[col].str.fullmatch(total_re, na=False)
            totals = group.loc[total_mask]
            if not totals.empty:
                return float(totals.sort_index()["_qty"].iloc[-1])

            # 2) If no grand total exists, mth3='소계' rows are non-overlapping
            #    subtotals for their (mth1, mth2) branches. Sum one per branch.
            subtotals = group.loc[group["_acqs_mth3"].str.fullmatch(subtotal_re, na=False)].copy()
            if not subtotals.empty:
                branch_cols = ["_acqs_mth1", "_acqs_mth2"]
                subtotals = (
                    subtotals.sort_index()
                    .drop_duplicates(branch_cols, keep="last")
                )
                return float(subtotals["_qty"].sum())

            # 3) Last fallback: sum unique leaf rows only. Exclude every aggregate
            #    label so parent/subtotal rows cannot be counted with their children.
            aggregate_re = rf"(?:{total_re}|{subtotal_re})"
            leaf = group.loc[
                ~group["_acqs_mth3"].str.fullmatch(aggregate_re, na=False)
                & group["_acqs_mth3"].ne("")
            ].copy()
            if not leaf.empty:
                leaf = (
                    leaf.sort_index()
                    .drop_duplicates(
                        ["_acqs_mth1", "_acqs_mth2", "_acqs_mth3"],
                        keep="last",
                    )
                )
                return float(leaf["_qty"].sum())

            preview_cols = [
                c for c in (
                    "acqs_mth1", "acqs_mth2", "acqs_mth3",
                    "stock_knd", "trmend_qy"
                ) if c in group.columns
            ]
            preview = group[preview_cols].head(12).to_dict("records")
            raise OpenDartError(
                "tesstkAcqsDspsSttus has numeric rows but no safely aggregatable "
                f"total/subtotal/leaf hierarchy for corp_code={corp_code}. "
                f"Payload preview={preview}"
            )

        # If DART supplies a grand-total stock class, use it alone. Otherwise
        # aggregate each actual stock class separately (e.g. common + preferred).
        grand_stock = work.loc[work["_stock_knd"].str.fullmatch(total_re, na=False)]
        if not grand_stock.empty:
            treasury = _class_treasury(grand_stock)
        else:
            treasury = 0.0
            grouped = work.groupby("_stock_knd", dropna=False, sort=False)
            for stock_kind, group in grouped:
                if stock_kind in {"", "비고", "note"}:
                    continue
                treasury += _class_treasury(group)

        return treasury, _period_end(rows), _receipt_date(rows)

    def _reconstruct_stock_total(
        self,
        corp_code: str,
        year: int,
        report_code: str,
        target_raw: pd.DataFrame,
    ) -> dict[str, Any]:
        target_period_end = _period_end(target_raw)
        if pd.isna(target_period_end):
            raise OpenDartError(
                f"Target report has no settlement date for corp_code={corp_code}."
            )

        baseline, baseline_year, baseline_code = self._latest_valid_baseline(
            corp_code, year, report_code
        )
        baseline_period_end = baseline["report_period_end"]
        if pd.isna(baseline_period_end):
            raise OpenDartError("Baseline report period end is unavailable.")

        delta, delta_notes = self._issued_share_delta(
            corp_code,
            year,
            report_code,
            baseline_period_end,
            target_period_end,
        )
        issued = float(baseline["shares_outstanding"]) + float(delta)
        if issued <= 0:
            raise OpenDartError(
                f"Reconstructed issued shares are invalid: baseline={baseline['shares_outstanding']}, "
                f"delta={delta}."
            )

        treasury, treasury_period_end, treasury_available_date = self._treasury_at_report_end(
            corp_code, year, report_code
        )
        if pd.notna(treasury_period_end) and pd.Timestamp(treasury_period_end) != pd.Timestamp(target_period_end):
            raise OpenDartError(
                f"Treasury report period mismatch: stock_total={target_period_end.date()}, "
                f"treasury={pd.Timestamp(treasury_period_end).date()}."
            )
        distributed = issued - treasury
        if distributed <= 0 or distributed > issued:
            raise OpenDartError(
                f"Reconstructed distributed shares invalid: issued={issued}, treasury={treasury}."
            )

        available_candidates = [
            _receipt_date(target_raw),
            treasury_available_date,
        ]
        available_candidates = [x for x in available_candidates if pd.notna(x)]
        available_date = max(available_candidates) if available_candidates else pd.NaT

        return {
            "shares_outstanding": issued,
            "treasury_shares": treasury,
            "distributed_shares": distributed,
            "report_period_end": pd.Timestamp(target_period_end).normalize(),
            "available_date": pd.Timestamp(available_date).normalize()
            if pd.notna(available_date)
            else pd.NaT,
            "share_status_source": "reconstructed:baseline+irds+tesstk",
            "baseline_report_year": baseline_year,
            "baseline_report_code": baseline_code,
            "issued_share_delta": float(delta),
            "reconstruction_events": " | ".join(delta_notes),
        }

    def stock_total_status(
        self,
        corp_code: str,
        year: int | str,
        report: str = "FY",
        *,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        """Return point-in-time issued/treasury/distributed-share status.

        Primary source is OpenDART ``stockTotqySttus``. Some filings return only
        ``-`` values (Samsung Electronics 2025 Q3 is one real example). In that
        case this client reconstructs the snapshot from:

        1. the latest earlier valid ``stockTotqySttus`` baseline,
        2. ``irdsSttus`` issuance/decrease events through the target report end,
        3. ``tesstkAcqsDspsSttus`` target-period treasury-share ending balance.

        ``asof_date`` is the disclosure-availability date (receipt date), not the
        report-period end. This avoids leaking quarter-end information into dates
        before the filing became public. ``report_period_end`` is retained separately.
        """
        corp_code = str(corp_code).strip().zfill(8)
        year_i = int(year)
        report_code = self.report_code(report)
        raw = self._stock_total_raw(corp_code, year_i, report_code)
        parsed = self._parse_direct_stock_total(raw)
        if parsed is None:
            parsed = self._reconstruct_stock_total(corp_code, year_i, report_code, raw)

        available_date = parsed.get("available_date")
        if pd.isna(available_date):
            raise OpenDartError(
                f"Disclosure receipt/availability date unavailable for corp_code={corp_code}; "
                "refusing to use report-period end as point-in-time availability."
            )

        record = {
            "corp_code": corp_code,
            "asof_date": pd.Timestamp(available_date).normalize(),
            "report_period_end": pd.Timestamp(parsed["report_period_end"]).normalize(),
            "shares_outstanding": float(parsed["shares_outstanding"]),
            "treasury_shares": float(parsed["treasury_shares"])
            if pd.notna(parsed["treasury_shares"])
            else np.nan,
            "distributed_shares": float(parsed["distributed_shares"]),
            "dart_report_year": year_i,
            "dart_report_code": report_code,
            "dart_report_label": REPORT_LABELS.get(report_code, report_code),
            "share_status_source": parsed.get("share_status_source", "stockTotqySttus"),
        }
        for key in (
            "baseline_report_year",
            "baseline_report_code",
            "issued_share_delta",
            "reconstruction_events",
        ):
            if key in parsed:
                record[key] = parsed[key]
        if ticker is not None:
            record["ticker"] = _ticker(ticker)
        return pd.DataFrame([record])

    def build_share_status(
        self,
        corp_map: pd.DataFrame,
        year: int | str,
        report: str = "FY",
    ) -> pd.DataFrame:
        required = {"ticker", "corp_code"}
        missing = required - set(corp_map.columns)
        if missing:
            raise OpenDartError(f"corp_map missing columns: {sorted(missing)}")
        frames = []
        for row in corp_map.loc[:, ["ticker", "corp_code"]].drop_duplicates().itertuples(index=False):
            frames.append(
                self.stock_total_status(
                    row.corp_code,
                    year,
                    report,
                    ticker=row.ticker,
                )
            )
        if not frames:
            raise OpenDartError("corp_map produced zero share-status rows.")
        out = pd.concat(frames, ignore_index=True)
        if out.duplicated(["ticker", "asof_date"]).any():
            raise OpenDartError("Duplicate ticker-asof_date rows in DART share-status output.")
        return out.sort_values(["ticker", "asof_date"]).reset_index(drop=True)
