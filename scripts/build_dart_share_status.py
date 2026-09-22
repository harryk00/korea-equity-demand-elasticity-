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

from korea_sd.data import OpenDartClient
from korea_sd.utils import atomic_write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build OpenDART issued/treasury/distributed-share snapshots.")
    p.add_argument("--corp-map", required=True, help="CSV with ticker,corp_code")
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--report", default="FY", help="FY, Q1, HY/H1, Q3 or DART report code")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    corp_map = pd.read_csv(args.corp_map, dtype={"ticker": str, "corp_code": str})
    client = OpenDartClient.from_env()
    out = client.build_share_status(corp_map, args.year, args.report)
    atomic_write_parquet(out, args.out)
    print(f"wrote {len(out):,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
