#!/usr/bin/env python3
from __future__ import annotations

import sys

# Allow direct execution from a source checkout without relying on pytest
# pythonpath or an editable install.
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import argparse
from pathlib import Path

import pandas as pd

from korea_sd.panels import build_stock_master
from korea_sd.utils import atomic_write_parquet


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, dtype={"ticker": str})
    return pd.read_parquet(p)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Point-in-time Stock Master builder.")
    p.add_argument("--market", required=True)
    p.add_argument("--free-float", required=True, dest="free_float")
    p.add_argument("--ff-ratio-col")
    p.add_argument("--ff-shares-col")
    p.add_argument("--distributed-shares-col")
    p.add_argument("--allow-missing-free-float", action="store_true")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    market = read_table(args.market)
    ff = read_table(args.free_float)
    master = build_stock_master(
        market,
        ff,
        ff_ratio_col=args.ff_ratio_col,
        ff_shares_col=args.ff_shares_col,
        distributed_shares_col=args.distributed_shares_col,
        require_free_float=not args.allow_missing_free_float,
    )
    atomic_write_parquet(master, args.out)
    print(f"wrote {len(master):,} rows x {len(master.columns):,} columns -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
