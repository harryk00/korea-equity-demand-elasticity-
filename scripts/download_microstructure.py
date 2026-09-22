#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution from a source checkout without relying on pytest
# pythonpath or an editable install.
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from korea_sd.data.kis_client import KisOpenApiClient
from korea_sd.data.kis_microstructure import KisMicrostructureProvider, MicrostructureDataError
from korea_sd.utils import atomic_write_parquet


PANEL_METHODS = {
    "lending": "lending",
    "short_selling": "short_selling",
    "credit": "credit",
    "program": "program",
    "execution_strength": "execution_strength",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download five independent KIS microstructure panels."
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--tickers", nargs="+", required=True, help="6-digit stock codes")
    p.add_argument(
        "--panels",
        nargs="+",
        choices=["all", *PANEL_METHODS],
        default=["all"],
    )
    p.add_argument("--out-dir", default="data/raw/microstructure")
    p.add_argument(
        "--pause-seconds",
        type=float,
        default=None,
        help="Override KIS_PAUSE_SECONDS. Default is env or 0.8s.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = {}
    if args.pause_seconds is not None:
        kwargs["pause_seconds"] = args.pause_seconds

    client = KisOpenApiClient.from_env(**kwargs)
    provider = KisMicrostructureProvider(client)

    requested = list(PANEL_METHODS) if "all" in args.panels else list(dict.fromkeys(args.panels))
    tickers = [str(t).strip().zfill(6) for t in args.tickers]
    out_dir = Path(args.out_dir)

    for panel_name in requested:
        method = getattr(provider, PANEL_METHODS[panel_name])
        frames: list[pd.DataFrame] = []

        # Strict panel-level atomicity: do not write until every requested ticker succeeded.
        for idx, ticker in enumerate(tickers, start=1):
            print(f"[{panel_name}] {idx}/{len(tickers)} {ticker}", file=sys.stderr)
            frame = method(ticker, args.start, args.end)
            if frame.empty:
                raise MicrostructureDataError(
                    f"{panel_name} returned zero rows for {ticker}; no file was written."
                )
            frames.append(frame)

        panel = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )
        out = out_dir / f"{panel_name}.parquet"
        atomic_write_parquet(panel, out)
        print(f"wrote {len(panel):,} rows -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
