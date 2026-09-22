#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet",
    )
    return p.parse_args()


def qbucket(pct):
    return np.ceil(pct * 5).clip(1, 5).astype("Int64")


def main():
    args = parse_args()
    df = pd.read_parquet(args.input)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).copy()

    df["prev_close"] = df.groupby("ticker")["close"].shift(1)
    df["ret_1d"] = df["close"] / df["prev_close"] - 1

    df["float_mcap_pct"] = (
        df.groupby("date")["free_float_market_cap"]
        .rank(method="average", pct=True, ascending=True)
    )
    df["turnover_pct"] = (
        df.groupby("date")["float_turnover_1d"]
        .rank(method="average", pct=True, ascending=True)
    )
    df["base"] = (
        (df["float_mcap_pct"] <= 0.20)
        & (df["turnover_pct"] >= 0.80)
    )

    sig = df["base"].fillna(False)
    for col, out in [
        ("float_turnover_1d", "turn_pct"),
        ("free_float_market_cap", "float_pct"),
        ("ret_1d", "ret_pct"),
    ]:
        df[out] = np.nan
        r = (
            df.loc[sig]
            .groupby("date")[col]
            .rank(method="average", pct=True, ascending=True)
        )
        df.loc[r.index, out] = r

    df["c5"] = (
        df["base"]
        & (qbucket(df["turn_pct"]) <= 2)
        & (qbucket(df["float_pct"]) <= 2)
        & (qbucket(df["ret_pct"]) < 5)
    )

    c5 = df[df["c5"]].copy()
    print("=== C5 ENRICHED COVERAGE ===")
    print("rows", len(c5))
    print("tickers", c5["ticker"].nunique())

    cols = [
        "program_net_buy_value",
        "program_net_buy_accel_1d",
        "execution_strength",
        "execution_strength_accel_1d",
    ]

    rows = []
    for year, g in c5.groupby(c5["date"].dt.year):
        row = {"year": int(year), "c5_rows": len(g)}
        for c in cols:
            if c in g.columns:
                row[f"{c}_coverage"] = float(g[c].notna().mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
