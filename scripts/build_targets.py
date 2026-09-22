#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from korea_sd.model import add_cooldown_events, add_forward_targets
from korea_sd.utils import atomic_write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build forward-return targets and independent surge events.")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cooldown", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input)
    out = add_forward_targets(df)
    out = add_cooldown_events(out, cooldown=args.cooldown)
    atomic_write_parquet(out, args.out)
    print(f"wrote {len(out):,} rows x {len(out.columns):,} columns -> {args.out}")
    print("\nTarget counts:")
    print(out["target_20d_30pct"].value_counts(dropna=False))
    print("\nIndependent +30% events:", int(out["surge_event_30pct"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
