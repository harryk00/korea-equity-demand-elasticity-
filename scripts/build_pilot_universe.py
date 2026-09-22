#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from korea_sd.data.dart_corp_codes import OpenDartCorpCodeProvider
from korea_sd.data.kis_universe import KisKosdaqMasterProvider, UniverseDataError


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a deterministic KOSDAQ mid/small-cap pilot universe from the official KIS master file."
    )
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--rank-start", type=int, default=31)
    p.add_argument("--rank-end", type=int, default=400)
    p.add_argument("--min-volume", type=int, default=30000)
    p.add_argument("--master-url", default=None, help="Optional KIS master ZIP override")
    p.add_argument("--out", default="data/universe/pilot_50.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    provider = KisKosdaqMasterProvider(**({"url": args.master_url} if args.master_url else {}))
    master = provider.download()
    print(f"parsed {len(master):,} KOSDAQ master rows")

    pilot = provider.select_pilot(
        master,
        count=args.count,
        rank_start=args.rank_start,
        rank_end=args.rank_end,
        min_volume=args.min_volume,
    )
    print(f"selected {len(pilot):,} pilot rows before DART mapping")

    dart_codes = OpenDartCorpCodeProvider.from_env().download()
    merged = pilot.merge(dart_codes[["ticker", "corp_code", "corp_name"]], on="ticker", how="left")
    missing = merged["corp_code"].isna()
    if missing.any():
        names = merged.loc[missing, ["ticker", "name"]].to_dict("records")
        raise UniverseDataError(f"DART corp_code missing for selected tickers: {names}")

    merged["selection_asof"] = date.today().isoformat()
    merged["selection_note"] = "current-survivor pilot from KIS KOSDAQ master; not final inference universe"
    cols = [
        "pilot_order", "ticker", "corp_code", "name", "corp_name",
        "market_cap_rank", "current_market_cap", "current_price", "current_volume",
        "listed_shares_current", "selection_asof", "selection_note",
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged[cols].to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {len(merged):,} rows -> {out}")
    print(merged[cols[:9]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
