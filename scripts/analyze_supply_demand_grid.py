#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="2D cross-sectional grid analysis for low float × demand pressure."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/supply_demand_grid",
    )
    p.add_argument(
        "--target",
        default="target_20d_30pct",
    )
    p.add_argument(
        "--event",
        default="surge_event_30pct",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Cross-sectional bins per axis. 5 is suitable for the ~46-name pilot.",
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=20,
    )
    return p.parse_args()


def cs_bin(df: pd.DataFrame, col: str, bins: int, higher_is_stronger: bool = True,
           min_cross_section: int = 20) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for _, idx in df.groupby("date", sort=False).groups.items():
        s = pd.to_numeric(df.loc[idx, col], errors="coerce")
        valid = s.dropna()
        if len(valid) < min_cross_section:
            continue

        pct = valid.rank(method="average", pct=True)

        if higher_is_stronger:
            bucket = np.ceil(pct * bins)
        else:
            # Low raw values should map to the strongest/highest bucket.
            bucket = np.ceil((1.000000001 - pct) * bins)

        bucket = np.clip(bucket, 1, bins).astype(int)
        out.loc[valid.index] = pd.Series(bucket, index=valid.index, dtype="Int64")

    return out


def make_grid(
    df: pd.DataFrame,
    x_col: str,
    x_higher_stronger: bool,
    y_col: str,
    y_higher_stronger: bool,
    target: str,
    event: str,
    bins: int,
    min_cross_section: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["ticker", "date", x_col, y_col, target, event]
    w = df[cols].copy()

    w["x_bucket"] = cs_bin(
        w, x_col, bins, x_higher_stronger, min_cross_section
    )
    w["y_bucket"] = cs_bin(
        w, y_col, bins, y_higher_stronger, min_cross_section
    )

    eligible = w.dropna(subset=["x_bucket", "y_bucket", target]).copy()

    g = (
        eligible.groupby(["x_bucket", "y_bucket"], observed=True)
        .agg(
            observations=(target, "size"),
            target_ones=(target, "sum"),
            target_rate=(target, "mean"),
            independent_events=(event, "sum"),
            x_median=(x_col, "median"),
            y_median=(y_col, "median"),
        )
        .reset_index()
    )

    overall = eligible[target].mean()
    g["overall_target_rate"] = overall
    g["relative_risk_vs_overall"] = (
        g["target_rate"] / overall if overall > 0 else np.nan
    )
    g["event_rate_per_1000_obs"] = (
        g["independent_events"] / g["observations"] * 1000
    )
    g["grid"] = label

    pivot = g.pivot(index="y_bucket", columns="x_bucket", values="target_rate")
    pivot = pivot.sort_index(ascending=False)
    pivot.index.name = "y_strength_bucket"
    pivot.columns.name = "x_strength_bucket"

    return g, pivot


def main():
    args = parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {
        "ticker", "date", args.target, args.event,
        "free_float_market_cap",
        "float_turnover_1d",
        "program_netbuy_float_mcap",
        "program_acceleration",
        "execution_strength",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    specs = [
        # Core thesis: scarce tradable supply × active demand.
        (
            "float_mcap_x_turnover",
            "free_float_market_cap", False,
            "float_turnover_1d", True,
        ),
        # Incremental order-flow test.
        (
            "float_mcap_x_program_netbuy",
            "free_float_market_cap", False,
            "program_netbuy_float_mcap", True,
        ),
        # Program acceleration as a change-in-demand signal.
        (
            "float_mcap_x_program_acceleration",
            "free_float_market_cap", False,
            "program_acceleration", True,
        ),
        # Execution strength as a secondary confirmation signal.
        (
            "float_mcap_x_execution_strength",
            "free_float_market_cap", False,
            "execution_strength", True,
        ),
    ]

    summaries = []

    for label, x_col, x_dir, y_col, y_dir in specs:
        grid, pivot = make_grid(
            df=df,
            x_col=x_col,
            x_higher_stronger=x_dir,
            y_col=y_col,
            y_higher_stronger=y_dir,
            target=args.target,
            event=args.event,
            bins=args.bins,
            min_cross_section=args.min_cross_section,
            label=label,
        )

        grid.to_csv(out_dir / f"{label}.csv", index=False)
        pivot.to_csv(out_dir / f"{label}_pivot.csv")

        strongest = grid.sort_values(
            ["relative_risk_vs_overall", "observations"],
            ascending=[False, False],
        ).iloc[0]

        # Bucket `bins` is defined as the "strongest" side on both axes:
        # low free-float market cap and high demand.
        thesis_cell = grid[
            (grid["x_bucket"] == args.bins) &
            (grid["y_bucket"] == args.bins)
        ]

        if len(thesis_cell):
            t = thesis_cell.iloc[0]
            thesis_rate = t["target_rate"]
            thesis_rr = t["relative_risk_vs_overall"]
            thesis_obs = int(t["observations"])
            thesis_events = int(t["independent_events"])
        else:
            thesis_rate = np.nan
            thesis_rr = np.nan
            thesis_obs = 0
            thesis_events = 0

        summaries.append({
            "grid": label,
            "overall_target_rate": float(grid["overall_target_rate"].iloc[0]),
            "thesis_cell_target_rate": thesis_rate,
            "thesis_cell_relative_risk": thesis_rr,
            "thesis_cell_observations": thesis_obs,
            "thesis_cell_independent_events": thesis_events,
            "best_x_bucket": int(strongest["x_bucket"]),
            "best_y_bucket": int(strongest["y_bucket"]),
            "best_cell_target_rate": float(strongest["target_rate"]),
            "best_cell_relative_risk": float(strongest["relative_risk_vs_overall"]),
            "best_cell_observations": int(strongest["observations"]),
        })

        print(f"\n=== {label} ===")
        print("Rows = y strength bucket, columns = x strength bucket")
        print("5 means stronger: smaller free-float mcap / higher demand.")
        print((pivot * 100).round(2).to_string())
        print("\nThesis cell (5,5):")
        print({
            "target_rate": thesis_rate,
            "relative_risk": thesis_rr,
            "observations": thesis_obs,
            "independent_events": thesis_events,
        })

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / "grid_summary.csv", index=False)

    print("\n=== Grid summary ===")
    print(summary.to_string(index=False))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
