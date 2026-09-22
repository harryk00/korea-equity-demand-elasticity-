#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="data/pit_kosdaq/core/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--micro-dir",
        default="data/pit_kosdaq/demand_microstructure",
    )
    p.add_argument(
        "--output",
        default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet",
    )
    return p.parse_args()


def main():
    args = parse_args()

    model = pd.read_parquet(args.model)
    model["ticker"] = model["ticker"].astype(str).str.zfill(6)
    model["date"] = pd.to_datetime(model["date"])

    micro_dir = Path(args.micro_dir)

    program_path = micro_dir / "program_daily.parquet"
    execution_path = micro_dir / "execution_strength_daily.parquet"

    if not program_path.exists():
        raise SystemExit(f"Missing: {program_path}")
    if not execution_path.exists():
        raise SystemExit(f"Missing: {execution_path}")

    program = pd.read_parquet(program_path)
    execution = pd.read_parquet(execution_path)

    for x in [program, execution]:
        x["ticker"] = x["ticker"].astype(str).str.zfill(6)
        x["date"] = pd.to_datetime(x["date"])

    # Keep standardized fields that the existing demand v3 analyzer recognizes.
    program_cols = [
        "ticker",
        "date",
        "program_sell_qty",
        "program_buy_qty",
        "program_net_buy_qty",
        "program_sell_value",
        "program_buy_value",
        "program_net_buy_value",
        "program_net_buy_value_lag1",
        "program_net_buy_accel_1d",
    ]
    program_cols = [c for c in program_cols if c in program.columns]

    execution_cols = [
        "ticker",
        "date",
        "sell_execution_qty",
        "buy_execution_qty",
        "execution_strength",
        "execution_imbalance",
        "execution_strength_lag1",
        "execution_strength_accel_1d",
    ]
    execution_cols = [c for c in execution_cols if c in execution.columns]

    # Remove any same-name old columns before adding refreshed panels.
    add_cols = set(program_cols + execution_cols) - {"ticker", "date"}
    old = [c for c in add_cols if c in model.columns]
    if old:
        model = model.drop(columns=old)

    out = model.merge(
        program[program_cols],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        execution[execution_cols],
        on=["ticker", "date"],
        how="left",
        validate="one_to_one",
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("=== MERGE SUMMARY ===")
    print("rows", len(out))
    for c in [
        "program_net_buy_value",
        "program_net_buy_accel_1d",
        "execution_strength",
        "execution_strength_accel_1d",
    ]:
        if c in out.columns:
            print(c, "non_null", int(out[c].notna().sum()))

    print(f"\nwrote -> {out_path}")


if __name__ == "__main__":
    main()
