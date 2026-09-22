#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd

from korea_sd.data.dart_client import OpenDartClient, OpenDartError
from korea_sd.data.kis_client import KisOpenApiClient
from korea_sd.data.kis_market import KisMarketProvider
from korea_sd.data.kis_microstructure import KisMicrostructureProvider
from korea_sd.model import add_cooldown_events, add_forward_targets
from korea_sd.panels import build_stock_master
from korea_sd.panels.pit_eligibility import filter_full_period_free_float_eligible
from korea_sd.panels.microstructure import add_microstructure_features, merge_microstructure_panels
from korea_sd.utils import atomic_write_parquet


PANELS = ("lending", "short_selling", "credit", "program", "execution_strength")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume-safe end-to-end pilot pipeline for market, DART, 5 microstructure panels and targets."
    )
    p.add_argument("--universe", default="data/universe/pilot_50.csv")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-08-29")
    p.add_argument("--out-root", default="data/pilot50")
    p.add_argument("--pause-seconds", type=float, default=1.2)
    p.add_argument("--refresh", action="store_true", help="Ignore per-ticker cache files and redownload.")
    p.add_argument("--skip-market", action="store_true")
    p.add_argument("--skip-dart", action="store_true")
    p.add_argument("--skip-microstructure", action="store_true")
    p.add_argument("--cooldown", type=int, default=20)
    return p.parse_args()


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str, "corp_code": str})
    for c in ("ticker", "corp_code"):
        if c not in df.columns:
            raise SystemExit(f"Universe must contain {c!r} column.")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(6)
    df["corp_code"] = df["corp_code"].astype(str).str.strip().str.zfill(8)
    return df.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _derived_dart_years(start: str, end: str) -> list[int]:
    start_year = pd.Timestamp(start).year
    end_year = pd.Timestamp(end).year
    # Example 2025-01..2026-08 -> FY2023, FY2024, FY2025.
    return list(range(start_year - 2, end_year))


def _load_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    universe = _read_csv(args.universe)
    root = Path(args.out_root)
    cache = root / "cache"
    processed = root / "processed"
    quality = root / "quality"
    for p in (cache, processed, quality):
        p.mkdir(parents=True, exist_ok=True)

    kis_client = KisOpenApiClient.from_env(pause_seconds=args.pause_seconds)
    market_provider = KisMarketProvider(client=kis_client)
    micro_provider = KisMicrostructureProvider(kis_client)
    dart_client = OpenDartClient.from_env()
    dart_years = _derived_dart_years(args.start, args.end)

    status_rows: list[dict[str, object]] = []

    # --------------------------------------------------------------
    # Per-ticker checkpointed downloads
    # --------------------------------------------------------------
    for n, row in enumerate(universe.itertuples(index=False), start=1):
        ticker = str(row.ticker).zfill(6)
        corp_code = str(row.corp_code).zfill(8)
        print(f"\n=== {n}/{len(universe)} {ticker} ===", file=sys.stderr)
        ticker_ok = True

        # Market
        market_path = cache / "market" / f"{ticker}.parquet"
        market_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not args.skip_market and (args.refresh or not market_path.exists()):
                frame = market_provider.daily_stock_ohlcv(ticker, args.start, args.end)
                atomic_write_parquet(frame, market_path)
            if _load_if_exists(market_path) is None:
                raise RuntimeError("market cache missing")
            status_rows.append({"ticker": ticker, "stage": "market", "status": "ok", "detail": ""})
        except Exception as exc:
            ticker_ok = False
            status_rows.append({"ticker": ticker, "stage": "market", "status": "fail", "detail": str(exc)})
            print(f"[FAIL market] {ticker}: {exc}", file=sys.stderr)

        # DART annual history
        if ticker_ok:
            dart_success = 0
            for year in dart_years:
                dart_path = cache / "dart" / f"{ticker}_{year}_FY.parquet"
                dart_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if not args.skip_dart and (args.refresh or not dart_path.exists()):
                        frame = dart_client.stock_total_status(corp_code, year, "FY", ticker=ticker)
                        atomic_write_parquet(frame, dart_path)
                    if _load_if_exists(dart_path) is not None:
                        dart_success += 1
                        status_rows.append({"ticker": ticker, "stage": f"dart_{year}", "status": "ok", "detail": ""})
                except OpenDartError as exc:
                    status_rows.append({"ticker": ticker, "stage": f"dart_{year}", "status": "fail", "detail": str(exc)})
                    print(f"[WARN dart {year}] {ticker}: {exc}", file=sys.stderr)
            if dart_success == 0:
                ticker_ok = False
                print(f"[FAIL dart] {ticker}: no annual snapshot succeeded", file=sys.stderr)

        # Microstructure panels
        if ticker_ok:
            for panel in PANELS:
                panel_path = cache / "microstructure" / panel / f"{ticker}.parquet"
                panel_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    if not args.skip_microstructure and (args.refresh or not panel_path.exists()):
                        method = getattr(micro_provider, panel)
                        frame = method(ticker, args.start, args.end)
                        atomic_write_parquet(frame, panel_path)
                    if _load_if_exists(panel_path) is None:
                        raise RuntimeError(f"{panel} cache missing")
                    status_rows.append({"ticker": ticker, "stage": panel, "status": "ok", "detail": ""})
                except Exception as exc:
                    ticker_ok = False
                    status_rows.append({"ticker": ticker, "stage": panel, "status": "fail", "detail": str(exc)})
                    print(f"[FAIL {panel}] {ticker}: {exc}", file=sys.stderr)
                    # Continue other panels so one rerun can fill as much cache as possible.

    status = pd.DataFrame(status_rows)
    status.to_csv(quality / "download_status.csv", index=False, encoding="utf-8-sig")

    # --------------------------------------------------------------
    # Determine complete tickers from caches, independent of this run.
    # --------------------------------------------------------------
    complete: list[str] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        market_ok = (cache / "market" / f"{ticker}.parquet").exists()
        dart_ok = any((cache / "dart" / f"{ticker}_{year}_FY.parquet").exists() for year in dart_years)
        micro_ok = all((cache / "microstructure" / p / f"{ticker}.parquet").exists() for p in PANELS)
        if market_ok and dart_ok and micro_ok:
            complete.append(ticker)

    if not complete:
        raise SystemExit(
            f"No fully complete tickers. See {quality / 'download_status.csv'} and rerun; caches are preserved."
        )

    eligible = universe.loc[universe["ticker"].isin(complete)].copy()
    eligible.to_csv(quality / "eligible_universe.csv", index=False, encoding="utf-8-sig")
    print(f"\nComplete tickers: {len(complete)}/{len(universe)}")

    # --------------------------------------------------------------
    # Consolidate market and DART
    # --------------------------------------------------------------
    market = pd.concat(
        [pd.read_parquet(cache / "market" / f"{t}.parquet") for t in complete],
        ignore_index=True,
    ).sort_values(["ticker", "date"]).reset_index(drop=True)
    atomic_write_parquet(market, processed / "market.parquet")

    dart_frames: list[pd.DataFrame] = []
    for t in complete:
        for year in dart_years:
            p = cache / "dart" / f"{t}_{year}_FY.parquet"
            if p.exists():
                dart_frames.append(pd.read_parquet(p))
    dart = (
        pd.concat(dart_frames, ignore_index=True)
        .sort_values(["ticker", "asof_date"])
        .drop_duplicates(["ticker", "asof_date"], keep="last")
        .reset_index(drop=True)
    )
    atomic_write_parquet(dart, processed / "dart_share_status_history.parquet")

    # Full-period PIT free-float eligibility. A ticker must already have a
    # positive distributed-shares snapshot by its first market date. This
    # avoids failing one ticker at a time and prevents both look-ahead and
    # missing-free-float rows in the research panel.
    pit_eligible, pit_report = filter_full_period_free_float_eligible(
        market,
        dart,
        tickers=complete,
        distributed_shares_col="distributed_shares",
    )
    pit_report.to_csv(
        quality / "free_float_eligibility.csv", index=False, encoding="utf-8-sig"
    )

    excluded_pit = [t for t in complete if t not in set(pit_eligible)]
    if excluded_pit:
        print(
            f"PIT free-float exclusions: {len(excluded_pit)} -> "
            + ", ".join(excluded_pit),
            file=sys.stderr,
        )
    print(f"PIT-eligible tickers: {len(pit_eligible)}/{len(complete)}")

    if not pit_eligible:
        raise SystemExit(
            f"No tickers have full-period PIT free-float coverage. "
            f"See {quality / 'free_float_eligibility.csv'}."
        )

    # From here onward use only full-period PIT-eligible names.
    complete = pit_eligible
    eligible = universe.loc[universe["ticker"].isin(complete)].copy()
    eligible.to_csv(quality / "eligible_universe.csv", index=False, encoding="utf-8-sig")

    market = market.loc[market["ticker"].isin(complete)].copy()
    dart = dart.loc[dart["ticker"].isin(complete)].copy()
    atomic_write_parquet(market, processed / "market.parquet")
    atomic_write_parquet(dart, processed / "dart_share_status_history.parquet")

    master = build_stock_master(
        market,
        dart,
        distributed_shares_col="distributed_shares",
        require_free_float=True,
    )
    atomic_write_parquet(master, processed / "stock_master.parquet")

    # --------------------------------------------------------------
    # Consolidate and merge five microstructure panels
    # --------------------------------------------------------------
    panels: dict[str, pd.DataFrame] = {}
    for panel in PANELS:
        frame = pd.concat(
            [pd.read_parquet(cache / "microstructure" / panel / f"{t}.parquet") for t in complete],
            ignore_index=True,
        )
        frame = (
            frame.sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
            .reset_index(drop=True)
        )
        panels[panel] = frame
        atomic_write_parquet(frame, processed / f"{panel}.parquet")

    merged = merge_microstructure_panels(master, panels)
    merged = add_microstructure_features(merged)
    atomic_write_parquet(merged, processed / "stock_master_microstructure.parquet")

    # Targets + cooldown events
    model = add_forward_targets(merged)
    model = add_cooldown_events(model, cooldown=args.cooldown)
    atomic_write_parquet(model, processed / "stock_master_model.parquet")

    summary = pd.DataFrame(
        [{
            "requested_tickers": len(universe),
            "complete_tickers": len(complete),
            "pit_excluded_tickers": len(excluded_pit),
            "rows": len(model),
            "target_20d_30pct_ones": int((model["target_20d_30pct"] == 1).sum()),
            "surge_event_30pct_count": int(model["surge_event_30pct"].sum()),
            "start": args.start,
            "end": args.end,
        }]
    )
    summary.to_csv(quality / "pipeline_summary.csv", index=False)

    print(f"wrote final model -> {processed / 'stock_master_model.parquet'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Per-ticker cache files were preserved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
