#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Recover unresolved delisted KOSDAQ ticker -> DART corp_code mappings "
            "from historical disclosure company names."
        )
    )
    p.add_argument(
        "--universe",
        default="data/pit_kosdaq/universe/pit_union_all.csv",
    )
    p.add_argument(
        "--start",
        default="2022-01-01",
        help="Historical disclosure scan start. 2022 is sufficient for 2025-26 exits in most cases.",
    )
    p.add_argument(
        "--end",
        default="2026-08-29",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/universe/historical_name_recovery",
    )
    p.add_argument("--window-days", type=int, default=80)
    p.add_argument("--page-count", type=int, default=100)
    p.add_argument("--pause-seconds", type=float, default=0.08)
    p.add_argument(
        "--fuzzy-topn",
        type=int,
        default=5,
        help="Diagnostic fuzzy candidates only; never auto-applied.",
    )
    return p.parse_args()


LEGAL_TOKENS = (
    "주식회사",
    "유한회사",
    "유한책임회사",
    "합자회사",
    "합명회사",
)


def normalize_name(value) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip().upper()
    if not s:
        return ""

    s = s.replace("㈜", "")
    s = re.sub(r"\(\s*주\s*\)", "", s)
    s = re.sub(r"\(\s*유\s*\)", "", s)

    for token in LEGAL_TOKENS:
        s = s.replace(token, "")

    s = re.sub(r"\bCO\.?\s*,?\s*LTD\.?\b", "", s)
    s = re.sub(r"\bCORPORATION\b", "", s)
    s = re.sub(r"\bINCORPORATED\b", "", s)

    s = re.sub(r"[\s\-\._,&·/()'\"`]+", "", s)
    return s


class DartListScanner:
    def __init__(
        self,
        api_key: str,
        *,
        page_count: int = 100,
        pause_seconds: float = 0.08,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.page_count = int(page_count)
        self.pause_seconds = float(pause_seconds)
        self.timeout = float(timeout)
        self.session = requests.Session()
        self.url = "https://opendart.fss.or.kr/api/list.json"

    def request_page(self, start, end, page_no):
        params = {
            "crtfc_key": self.api_key,
            "bgn_de": pd.Timestamp(start).strftime("%Y%m%d"),
            "end_de": pd.Timestamp(end).strftime("%Y%m%d"),
            "pblntf_ty": "A",  # periodic disclosures
            "sort": "date",
            "sort_mth": "desc",
            "page_no": str(page_no),
            "page_count": str(self.page_count),
        }

        r = self.session.get(self.url, params=params, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(
                f"OpenDART HTTP {r.status_code}: {r.text[:300]}"
            )

        body = r.json()
        status = str(body.get("status", ""))

        if status == "013":
            return {
                "list": [],
                "total_page": 0,
                "total_count": 0,
            }

        if status != "000":
            raise RuntimeError(
                f"OpenDART status={status}: {body.get('message')}"
            )

        return body

    def windows(self, start, end, window_days):
        lo = pd.Timestamp(start).normalize()
        hi = pd.Timestamp(end).normalize()
        cur_hi = hi

        while cur_hi >= lo:
            cur_lo = max(
                lo,
                cur_hi - pd.Timedelta(days=window_days - 1),
            )
            yield cur_lo, cur_hi
            cur_hi = cur_lo - pd.Timedelta(days=1)

    def scan(self, start, end, window_days=80):
        rows = []
        windows = list(self.windows(start, end, window_days))
        total_windows = len(windows)

        for wi, (lo, hi) in enumerate(windows, start=1):
            first = self.request_page(lo, hi, 1)
            total_page = int(first.get("total_page") or 0)

            bodies = [first]

            for page in range(2, total_page + 1):
                time.sleep(self.pause_seconds)
                bodies.append(self.request_page(lo, hi, page))

            for body in bodies:
                for item in body.get("list") or []:
                    corp_code = str(item.get("corp_code") or "").strip()
                    corp_name = str(item.get("corp_name") or "").strip()

                    if not corp_code or not corp_name:
                        continue

                    rows.append(
                        {
                            "corp_code": corp_code.zfill(8),
                            "corp_name_hist": corp_name,
                            "name_key": normalize_name(corp_name),
                            "stock_code_api": str(
                                item.get("stock_code") or ""
                            ).strip(),
                            "corp_cls": str(
                                item.get("corp_cls") or ""
                            ).strip(),
                            "report_nm": str(
                                item.get("report_nm") or ""
                            ).strip(),
                            "rcept_dt": pd.to_datetime(
                                str(item.get("rcept_dt") or ""),
                                format="%Y%m%d",
                                errors="coerce",
                            ),
                        }
                    )

            print(
                f"[historical-name] windows {wi}/{total_windows} "
                f"({lo.date()} ~ {hi.date()}), "
                f"pages={total_page}, accumulated rows={len(rows):,}"
            )

            time.sleep(self.pause_seconds)

        if not rows:
            return pd.DataFrame(
                columns=[
                    "corp_code",
                    "corp_name_hist",
                    "name_key",
                    "stock_code_api",
                    "corp_cls",
                    "report_nm",
                    "rcept_dt",
                ]
            )

        return (
            pd.DataFrame(rows)
            .drop_duplicates(
                [
                    "corp_code",
                    "corp_name_hist",
                    "report_nm",
                    "rcept_dt",
                ]
            )
            .sort_values(["corp_code", "rcept_dt"])
            .reset_index(drop=True)
        )


def build_exact_recovery(universe, history):
    unresolved = universe[
        (universe["exited_before_end"] == True)
        & (universe["corp_map_status"] == "missing")
    ].copy()

    unresolved["source_name"] = (
        unresolved["interval_name"]
        .fillna(unresolved.get("corp_name", ""))
        .astype(str)
    )
    unresolved["name_key"] = unresolved["source_name"].map(normalize_name)

    hist = history.copy()
    hist = hist[hist["name_key"].ne("")].copy()

    grouped = (
        hist.groupby("name_key", as_index=False)
        .agg(
            candidate_count=("corp_code", "nunique"),
            candidate_codes=(
                "corp_code",
                lambda s: ";".join(sorted(set(s.astype(str)))),
            ),
            historical_names=(
                "corp_name_hist",
                lambda s: ";".join(sorted(set(s.astype(str)))),
            ),
            first_rcept_dt=("rcept_dt", "min"),
            last_rcept_dt=("rcept_dt", "max"),
            corp_classes=(
                "corp_cls",
                lambda s: ";".join(
                    sorted(set(x for x in s.astype(str) if x))
                ),
            ),
        )
    )

    # Keep diagnostics tolerant to universe-schema differences.
    # Some PIT-universe builds expose listing interval columns such as
    # first_listing_date / last_delisting_date rather than first_seen_date /
    # last_seen_date. Only select columns that actually exist.
    preferred_cols = [
        "ticker",
        "source_name",
        "name_key",
        "first_seen_date",
        "last_seen_date",
        "first_listing_date",
        "last_delisting_date",
        "first_membership_date",
        "last_membership_date",
        "exited_before_end",
    ]
    audit_cols = [c for c in preferred_cols if c in unresolved.columns]

    # These are created immediately above and therefore must always be present.
    for required_col in ["ticker", "source_name", "name_key", "exited_before_end"]:
        if required_col not in audit_cols:
            audit_cols.append(required_col)

    audit = unresolved[audit_cols].merge(
        grouped,
        on="name_key",
        how="left",
    )

    audit["candidate_count"] = (
        audit["candidate_count"].fillna(0).astype(int)
    )
    audit["recovery_status"] = np.where(
        audit["candidate_count"].eq(1),
        "recovered_unique_historical_name",
        np.where(
            audit["candidate_count"].gt(1),
            "ambiguous_historical_name",
            "no_historical_name_match",
        ),
    )

    recovered = audit[
        audit["recovery_status"].eq(
            "recovered_unique_historical_name"
        )
    ].copy()

    recovered["corp_code"] = (
        recovered["candidate_codes"].astype(str).str.zfill(8)
    )

    return audit, recovered


def fuzzy_candidates(universe, history, exact_audit, topn=5):
    residual = exact_audit[
        ~exact_audit["recovery_status"].eq(
            "recovered_unique_historical_name"
        )
    ].copy()

    name_table = (
        history[["name_key", "corp_code", "corp_name_hist"]]
        .dropna()
        .drop_duplicates()
    )

    # Only one representative display name per (name_key, corp_code).
    candidate_rows = name_table.to_dict("records")
    out = []

    for row in residual.itertuples(index=False):
        source_key = str(row.name_key or "")
        source_name = str(row.source_name or "")

        scores = []
        for cand in candidate_rows:
            key = str(cand["name_key"] or "")
            if not source_key or not key:
                continue

            score = difflib.SequenceMatcher(
                None,
                source_key,
                key,
            ).ratio()

            scores.append(
                (
                    score,
                    str(cand["corp_code"]),
                    str(cand["corp_name_hist"]),
                    key,
                )
            )

        scores.sort(reverse=True, key=lambda x: x[0])

        seen = set()
        rank = 0
        for score, corp_code, corp_name, key in scores:
            pair = (corp_code, key)
            if pair in seen:
                continue
            seen.add(pair)
            rank += 1

            out.append(
                {
                    "ticker": str(row.ticker).zfill(6),
                    "source_name": source_name,
                    "source_name_key": source_key,
                    "rank": rank,
                    "similarity": score,
                    "candidate_corp_code": corp_code,
                    "candidate_corp_name": corp_name,
                    "candidate_name_key": key,
                }
            )

            if rank >= topn:
                break

    return pd.DataFrame(out)


def apply_recovery(universe, recovered):
    out = universe.copy()

    recovery_map = dict(
        zip(
            recovered["ticker"].astype(str).str.zfill(6),
            recovered["corp_code"].astype(str).str.zfill(8),
        )
    )

    for idx, row in out.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        corp_code = recovery_map.get(ticker)

        if not corp_code:
            continue

        out.at[idx, "corp_code"] = corp_code
        out.at[idx, "corp_code_candidates"] = corp_code
        out.at[idx, "corp_code_count"] = 1
        out.at[idx, "corp_map_status"] = "ok"

        old_source = str(row.get("corp_map_source") or "").strip()
        sources = [
            x
            for x in [
                old_source,
                "historical_disclosure_name_exact",
            ]
            if x
        ]
        out.at[idx, "corp_map_source"] = "+".join(
            sorted(set("+".join(sources).split("+")))
        )

    out["eligible_for_pipeline"] = out["corp_map_status"].eq("ok")
    if "is_spac" in out.columns:
        out["eligible_for_pipeline"] &= ~out["is_spac"].fillna(False)

    return out


def print_coverage(label, df):
    ex = df[df["exited_before_end"] == True]
    mapped = int(ex["corp_map_status"].eq("ok").sum())
    total = len(ex)
    missing = int(ex["corp_map_status"].eq("missing").sum())
    coverage = mapped / total if total else np.nan

    print(f"\n=== {label} ===")
    print(f"delisted total : {total}")
    print(f"mapped         : {mapped}")
    print(f"missing        : {missing}")
    print(f"coverage       : {coverage:.2%}")


def main():
    args = parse_args()

    key = os.environ.get("OPENDART_API_KEY", "").strip()
    if not key:
        raise SystemExit("Set OPENDART_API_KEY first.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    universe = pd.read_csv(
        args.universe,
        dtype={
            "ticker": str,
            "corp_code": str,
            "corp_code_candidates": str,
        },
    )
    universe["ticker"] = universe["ticker"].astype(str).str.zfill(6)

    if "interval_name" not in universe.columns:
        raise SystemExit(
            "pit_union_all.csv must contain interval_name."
        )

    unresolved = universe[
        (universe["exited_before_end"] == True)
        & (universe["corp_map_status"] == "missing")
    ]

    print_coverage("Before historical-name recovery", universe)
    print(f"target unresolved delisted tickers: {len(unresolved)}")

    scanner = DartListScanner(
        key,
        page_count=args.page_count,
        pause_seconds=args.pause_seconds,
    )

    history = scanner.scan(
        start=args.start,
        end=args.end,
        window_days=args.window_days,
    )

    history.to_parquet(
        out_dir / "historical_periodic_disclosure_names.parquet",
        index=False,
    )

    exact_audit, recovered = build_exact_recovery(
        universe,
        history,
    )

    exact_audit.to_csv(
        out_dir / "exact_name_recovery_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    recovered.to_csv(
        out_dir / "recovered_unique_mappings.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fuzzy = fuzzy_candidates(
        universe,
        history,
        exact_audit,
        topn=args.fuzzy_topn,
    )

    fuzzy.to_csv(
        out_dir / "residual_fuzzy_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    patched = apply_recovery(universe, recovered)

    patched.to_csv(
        out_dir / "pit_union_all_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pipeline = patched[
        patched["eligible_for_pipeline"] == True
    ].copy()

    pipeline.to_csv(
        out_dir / "pit_pipeline_universe_v4.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Historical exact-name recovery audit ===")
    print(
        exact_audit["recovery_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print_coverage("After historical-name recovery", patched)

    ex = patched[patched["exited_before_end"] == True]
    coverage = float(ex["corp_map_status"].eq("ok").mean())

    print("\n=== Output ===")
    print(
        f"recovered unique mappings: {len(recovered)}"
    )
    print(
        f"pipeline tickers after recovery: {len(pipeline)}"
    )
    print(
        f"fuzzy diagnostics rows: {len(fuzzy)} "
        "(diagnostic only; NOT auto-applied)"
    )
    print(f"wrote -> {out_dir}")

    print("\n=== Decision ===")
    if coverage >= 0.90:
        print(
            "PASS >=90% delisted DART mapping coverage. "
            "Review residuals, then full core can proceed."
        )
    elif coverage >= 0.80:
        print(
            "BORDERLINE 80-90%. Review residual fuzzy candidates "
            "before the full core run."
        )
    else:
        print(
            "FAIL <80%. Do not run full 1,700+ ticker core yet."
        )


if __name__ == "__main__":
    main()
