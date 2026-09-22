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

from korea_sd.panels.microstructure import (
    add_microstructure_features,
    merge_microstructure_panels,
)
from korea_sd.utils import atomic_write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge independent microstructure panels onto Stock Master by ticker-date."
    )
    p.add_argument("--master", required=True, help="Existing Stock Master parquet")
    p.add_argument("--lending")
    p.add_argument("--short-selling", dest="short_selling")
    p.add_argument("--credit")
    p.add_argument("--program")
    p.add_argument("--execution-strength", dest="execution_strength")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--no-features",
        action="store_true",
        help="Merge raw canonical columns only; do not create ratios/rolling features.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    master = pd.read_parquet(args.master)

    panels: dict[str, pd.DataFrame] = {}
    for name in ["lending", "short_selling", "credit", "program", "execution_strength"]:
        path = getattr(args, name)
        if path:
            panels[name] = pd.read_parquet(path)

    if not panels:
        raise SystemExit("At least one microstructure panel path is required.")

    merged = merge_microstructure_panels(master, panels)
    if not args.no_features:
        merged = add_microstructure_features(merged)

    atomic_write_parquet(merged, args.out)
    print(f"wrote {len(merged):,} rows x {len(merged.columns):,} columns -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
