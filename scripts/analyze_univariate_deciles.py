#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "free_float_market_cap",
    "float_days",
    "float_turnover_1d",
    "lending_balance_float",
    "lending_acceleration",
    "short_volume_float",
    "short_acceleration",
    "credit_balance_float",
    "credit_acceleration",
    "program_netbuy_float_mcap",
    "program_acceleration",
    "execution_strength",
    "execution_strength_acceleration",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Cross-sectional decile analysis for +30% forward-return targets."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
        help="Input model parquet.",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/univariate",
        help="Output directory.",
    )
    p.add_argument(
        "--target",
        default="target_20d_30pct",
        help="Daily forward target column.",
    )
    p.add_argument(
        "--event",
        default="surge_event_30pct",
        help="Independent surge-event column.",
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=20,
        help="Minimum non-null names on a date to form deciles.",
    )
    p.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional feature list. Defaults to project core features.",
    )
    return p.parse_args()


def cross_sectional_decile(
    df: pd.DataFrame,
    feature: str,
    min_cross_section: int = 20,
) -> pd.Series:
    """
    Form deciles separately on each date to avoid mixing market regimes.
    Returns nullable Int64 values 1..10.
    """
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for _, idx in df.groupby("date", sort=False).groups.items():
        s = pd.to_numeric(df.loc[idx, feature], errors="coerce")
        valid = s.dropna()

        if len(valid) < min_cross_section:
            continue

        # percentile ranks handle duplicate values more robustly than qcut
        pct = valid.rank(method="average", pct=True)
        decile = np.ceil(pct * 10).clip(1, 10).astype(int)
        out.loc[valid.index] = decile.astype("Int64")

    return out


def analyse_feature(
    df: pd.DataFrame,
    feature: str,
    target: str,
    event: str,
    min_cross_section: int,
) -> tuple[pd.DataFrame, dict]:
    work = df[["ticker", "date", feature, target, event]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[target] = pd.to_numeric(work[target], errors="coerce")
    work[event] = pd.to_numeric(work[event], errors="coerce")

    work["decile"] = cross_sectional_decile(
        work,
        feature=feature,
        min_cross_section=min_cross_section,
    )

    eligible = work.dropna(subset=["decile", target]).copy()
    if eligible.empty:
        return pd.DataFrame(), {
            "feature": feature,
            "usable_rows": 0,
            "overall_target_rate": np.nan,
            "top_decile_target_rate": np.nan,
            "bottom_decile_target_rate": np.nan,
            "top_vs_overall": np.nan,
            "bottom_vs_overall": np.nan,
            "direction": "insufficient_data",
        }

    overall = eligible[target].mean()

    grouped = (
        eligible.groupby("decile", observed=True)
        .agg(
            observations=(target, "size"),
            target_ones=(target, "sum"),
            target_rate=(target, "mean"),
            independent_events=(event, "sum"),
            feature_median=(feature, "median"),
            feature_mean=(feature, "mean"),
        )
        .reset_index()
        .sort_values("decile")
    )

    grouped["feature"] = feature
    grouped["overall_target_rate"] = overall
    grouped["relative_risk_vs_overall"] = grouped["target_rate"] / overall if overall > 0 else np.nan
    grouped["event_rate_per_1000_obs"] = (
        grouped["independent_events"] / grouped["observations"] * 1000
    )

    d1 = grouped.loc[grouped["decile"] == 1, "target_rate"]
    d10 = grouped.loc[grouped["decile"] == 10, "target_rate"]

    bottom = float(d1.iloc[0]) if len(d1) else np.nan
    top = float(d10.iloc[0]) if len(d10) else np.nan

    if pd.notna(top) and pd.notna(bottom):
        direction = "higher_feature_stronger" if top > bottom else (
            "lower_feature_stronger" if bottom > top else "flat"
        )
    else:
        direction = "insufficient_data"

    summary = {
        "feature": feature,
        "usable_rows": int(len(eligible)),
        "overall_target_rate": float(overall),
        "bottom_decile_target_rate": bottom,
        "top_decile_target_rate": top,
        "bottom_vs_overall": bottom / overall if overall > 0 and pd.notna(bottom) else np.nan,
        "top_vs_overall": top / overall if overall > 0 and pd.notna(top) else np.nan,
        "direction": direction,
    }

    return grouped, summary


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {"ticker", "date", args.target, args.event}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    features = args.features or DEFAULT_FEATURES
    features = [f for f in features if f in df.columns]

    if not features:
        raise SystemExit("None of the requested analysis features exist in the input dataset.")

    all_tables = []
    summaries = []

    for feature in features:
        table, summary = analyse_feature(
            df=df,
            feature=feature,
            target=args.target,
            event=args.event,
            min_cross_section=args.min_cross_section,
        )
        summaries.append(summary)

        if not table.empty:
            all_tables.append(table)
            table.to_csv(out_dir / f"{feature}_deciles.csv", index=False)

    summary_df = pd.DataFrame(summaries)

    if not summary_df.empty:
        summary_df["extreme_decile_spread_pp"] = (
            summary_df["top_decile_target_rate"]
            - summary_df["bottom_decile_target_rate"]
        ) * 100

        summary_df["stronger_extreme_vs_overall"] = summary_df[
            ["top_vs_overall", "bottom_vs_overall"]
        ].max(axis=1)

        summary_df = summary_df.sort_values(
            ["stronger_extreme_vs_overall", "usable_rows"],
            ascending=[False, False],
        )

    summary_df.to_csv(out_dir / "feature_summary.csv", index=False)

    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(
            out_dir / "all_deciles.csv",
            index=False,
        )

    print(f"tickers: {df['ticker'].nunique()}")
    print(f"rows: {len(df):,}")
    print(f"features analysed: {len(features)}")
    print(f"output: {out_dir}")

    print("\n=== Overall +30% target ===")
    print(df[args.target].value_counts(dropna=False))

    print("\n=== Feature ranking ===")
    show_cols = [
        "feature",
        "usable_rows",
        "bottom_decile_target_rate",
        "top_decile_target_rate",
        "extreme_decile_spread_pp",
        "stronger_extreme_vs_overall",
        "direction",
    ]
    print(summary_df[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
