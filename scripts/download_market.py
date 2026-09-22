#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from korea_sd.data import KisMarketProvider, MarketDataError, PykrxMarketProvider
from korea_sd.utils import atomic_write_parquet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download Korean-equity daily market data. KIS is the recommended provider."
    )
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", choices=["kis", "pykrx"], default="kis")
    p.add_argument("--pause-seconds", type=float, default=1.0)
    p.add_argument(
        "--raw-price",
        action="store_true",
        help="KIS only: use unadjusted/original prices instead of adjusted prices.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.provider == "kis":
        provider = KisMarketProvider.from_env(
            adjusted=not args.raw_price,
            client_kwargs={"pause_seconds": args.pause_seconds},
        )
        label = "KIS"
    else:
        provider = PykrxMarketProvider(pause_seconds=args.pause_seconds)
        label = "pykrx"

    tickers = [str(t).strip().zfill(6) for t in args.tickers]
    frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{label}] {i}/{len(tickers)} {ticker}", file=sys.stderr)
        frame = provider.daily_stock_ohlcv(ticker, args.start, args.end)
        if frame.empty:
            raise MarketDataError(f"No rows for {ticker}; no output file written.")
        frames.append(frame)

    out = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    if set(out["ticker"].unique()) != set(tickers):
        raise MarketDataError("Ticker batch incomplete; no output file written.")
    atomic_write_parquet(out, args.out)
    print(f"wrote {len(out):,} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
