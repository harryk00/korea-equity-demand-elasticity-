#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit survivorship/mapping gap by membership-day, not just ticker count."
    )
    p.add_argument(
        "--universe",
        default="data/pit_kosdaq/universe/pit_union_all.csv",
    )
    p.add_argument(
        "--membership",
        default="data/pit_kosdaq/universe/membership_daily.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/universe/survivorship_gap_audit",
    )
    p.add_argument(
        "--max-missing-row-share",
        type=float,
        default=0.05,
        help="Strict-backtest pass threshold for missing non-SPAC membership rows.",
    )
    p.add_argument(
        "--min-daily-coverage",
        type=float,
        default=0.95,
        help="Pass threshold for average daily mapped non-SPAC membership coverage.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    u = pd.read_csv(args.universe, dtype={"ticker": str})
    u["ticker"] = u["ticker"].astype(str).str.zfill(6)

    m = pd.read_parquet(args.membership)
    m["ticker"] = m["ticker"].astype(str).str.zfill(6)
    m["date"] = pd.to_datetime(m["date"])

    cols = ["ticker", "corp_map_status", "exited_before_end"]
    if "is_spac" in u.columns:
        cols.append("is_spac")
    else:
        u["is_spac"] = False
        cols.append("is_spac")

    meta = u[cols].drop_duplicates("ticker")
    x = m.merge(meta, on="ticker", how="left", validate="many_to_one")

    x["is_spac"] = x["is_spac"].fillna(False).astype(bool)
    x["mapped"] = x["corp_map_status"].eq("ok")
    x["missing"] = x["corp_map_status"].eq("missing")
    x["non_spac"] = ~x["is_spac"]

    ns = x[x["non_spac"]].copy()

    total_rows = len(ns)
    mapped_rows = int(ns["mapped"].sum())
    missing_rows = int(ns["missing"].sum())
    missing_row_share = missing_rows / total_rows if total_rows else np.nan

    # Focus specifically on delisted names.
    de = ns[ns["exited_before_end"] == True].copy()
    de_rows = len(de)
    de_mapped_rows = int(de["mapped"].sum())
    de_missing_rows = int(de["missing"].sum())
    de_missing_row_share = de_missing_rows / de_rows if de_rows else np.nan

    daily = (
        ns.groupby("date", as_index=False)
        .agg(
            total_non_spac=("ticker", "nunique"),
            mapped_non_spac=("mapped", "sum"),
            missing_non_spac=("missing", "sum"),
        )
    )
    daily["mapped_coverage"] = (
        daily["mapped_non_spac"] / daily["total_non_spac"]
    )
    daily["missing_share"] = (
        daily["missing_non_spac"] / daily["total_non_spac"]
    )

    yearly = (
        ns.assign(year=ns["date"].dt.year)
        .groupby("year", as_index=False)
        .agg(
            membership_rows=("ticker", "size"),
            mapped_rows=("mapped", "sum"),
            missing_rows=("missing", "sum"),
            unique_tickers=("ticker", "nunique"),
        )
    )
    yearly["missing_row_share"] = (
        yearly["missing_rows"] / yearly["membership_rows"]
    )
    yearly["mapped_row_coverage"] = (
        yearly["mapped_rows"] / yearly["membership_rows"]
    )

    # Ticker-level membership duration.
    ticker_days = (
        ns.groupby(
            ["ticker", "corp_map_status", "exited_before_end"],
            as_index=False,
        )
        .agg(
            first_membership=("date", "min"),
            last_membership=("date", "max"),
            membership_days=("date", "nunique"),
        )
    )

    missing_delisted = ticker_days[
        (ticker_days["exited_before_end"] == True)
        & (ticker_days["corp_map_status"] == "missing")
    ].copy()

    missing_delisted["exit_month"] = (
        missing_delisted["last_membership"].dt.to_period("M").astype(str)
    )

    exit_month = (
        missing_delisted.groupby("exit_month", as_index=False)
        .agg(
            missing_tickers=("ticker", "nunique"),
            missing_membership_days=("membership_days", "sum"),
        )
        .sort_values("exit_month")
    )

    # Concentration: how much of the missing exposure comes from the longest-lived names?
    md = missing_delisted.sort_values("membership_days", ascending=False).copy()
    total_missing_days = md["membership_days"].sum()
    md["share_of_missing_days"] = (
        md["membership_days"] / total_missing_days
        if total_missing_days
        else np.nan
    )
    md["cum_share_of_missing_days"] = md["share_of_missing_days"].cumsum()

    summary = {
        "non_spac_union_tickers": int(
            u.loc[~u["is_spac"].fillna(False), "ticker"].nunique()
        ),
        "non_spac_membership_rows": int(total_rows),
        "mapped_membership_rows": int(mapped_rows),
        "missing_membership_rows": int(missing_rows),
        "missing_membership_row_share": float(missing_row_share),
        "mapped_membership_row_coverage": float(mapped_rows / total_rows) if total_rows else np.nan,
        "delisted_membership_rows": int(de_rows),
        "delisted_mapped_membership_rows": int(de_mapped_rows),
        "delisted_missing_membership_rows": int(de_missing_rows),
        "delisted_missing_membership_row_share": float(de_missing_row_share),
        "avg_daily_mapped_coverage": float(daily["mapped_coverage"].mean()),
        "median_daily_mapped_coverage": float(daily["mapped_coverage"].median()),
        "min_daily_mapped_coverage": float(daily["mapped_coverage"].min()),
        "max_daily_missing_share": float(daily["missing_share"].max()),
        "dates_below_95pct_coverage": int((daily["mapped_coverage"] < 0.95).sum()),
        "trading_dates": int(daily["date"].nunique()),
        "missing_delisted_tickers": int(len(missing_delisted)),
    }

    passes = (
        summary["missing_membership_row_share"] <= args.max_missing_row_share
        and summary["avg_daily_mapped_coverage"] >= args.min_daily_coverage
    )

    if passes:
        decision = "PASS_STRICT_MAPPED_SUBSET"
    elif summary["missing_membership_row_share"] <= 0.10:
        decision = "CAUTION_RUN_WITH_BIAS_DISCLOSURE"
    else:
        decision = "FAIL_MAPPING_GAP_TOO_LARGE"

    summary["decision"] = decision

    pd.DataFrame([summary]).to_csv(
        out_dir / "summary.csv", index=False
    )
    daily.to_csv(out_dir / "daily_coverage.csv", index=False)
    yearly.to_csv(out_dir / "yearly_coverage.csv", index=False)
    md.to_csv(out_dir / "missing_delisted_tickers.csv", index=False)
    exit_month.to_csv(out_dir / "missing_by_exit_month.csv", index=False)

    print("\n=== Survivorship Gap Audit ===")
    for k, v in summary.items():
        if isinstance(v, float):
            if "share" in k or "coverage" in k:
                print(f"{k:38s}: {v:.2%}")
            else:
                print(f"{k:38s}: {v:.6f}")
        else:
            print(f"{k:38s}: {v}")

    print("\n=== Yearly Coverage ===")
    print(yearly.to_string(index=False))

    print("\n=== Missing Delisted: Top 20 by Membership Days ===")
    print(
        md[
            [
                "ticker",
                "first_membership",
                "last_membership",
                "membership_days",
                "share_of_missing_days",
                "cum_share_of_missing_days",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nInterpretation:")
    if decision == "PASS_STRICT_MAPPED_SUBSET":
        print(
            "- Missing names are <=5% of non-SPAC membership rows and average daily "
            "coverage is >=95%."
        )
        print(
            "- Proceed with the mapped-subset full core, but label results as a "
            "strict PIT mapped-universe test, not a perfect full-universe test."
        )
    elif decision == "CAUTION_RUN_WITH_BIAS_DISCLOSURE":
        print(
            "- Mapping gap is material but not dominant. A full mapped-subset run "
            "can be informative, but survivorship bias must be treated as a major limitation."
        )
    else:
        print(
            "- Missing membership exposure is too large. Do not treat the mapped subset "
            "as a credible full-market PIT validation."
        )

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
