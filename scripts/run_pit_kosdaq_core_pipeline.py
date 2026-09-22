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

from korea_sd.data.dart_client import OpenDartClient, OpenDartError
from korea_sd.data.kis_client import KisOpenApiClient
from korea_sd.data.kis_market import KisMarketProvider
from korea_sd.data.pit_universe import PykrxOhlcvOnlyProvider, trim_market_to_observable_free_float
from korea_sd.model import add_cooldown_events, add_forward_targets
from korea_sd.panels import build_stock_master
from korea_sd.utils import atomic_write_parquet


def parse_args():
    p = argparse.ArgumentParser(
        description="Resume-safe whole-KOSDAQ PIT core pipeline: market + DART + supply features + targets."
    )
    p.add_argument(
        "--universe",
        default="data/pit_kosdaq/universe/pit_pipeline_universe.csv",
    )
    p.add_argument(
        "--membership",
        default="data/pit_kosdaq/universe/membership_daily.parquet",
    )
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-08-29")
    p.add_argument("--out-root", default="data/pit_kosdaq/core")
    p.add_argument("--pause-seconds", type=float, default=1.2)
    p.add_argument("--pykrx-fallback-pause", type=float, default=0.5)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--skip-market", action="store_true")
    p.add_argument("--skip-dart", action="store_true")
    p.add_argument("--cooldown", type=int, default=20)
    p.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full historical union.",
    )
    return p.parse_args()


def _read_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"ticker": str, "corp_code": str})
    required = {"ticker", "corp_code"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Universe missing columns: {sorted(missing)}")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["corp_code"] = df["corp_code"].astype(str).str.zfill(8)
    return df.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _read_membership(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, dtype={"ticker": str})
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    return df.dropna(subset=["date"]).drop_duplicates(["date", "ticker"])


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _dart_years_for_window(start: pd.Timestamp, end: pd.Timestamp) -> list[int]:
    return list(range(start.year - 2, end.year))


def main() -> int:
    args = parse_args()
    global_start = pd.Timestamp(args.start).normalize()
    global_end = pd.Timestamp(args.end).normalize()

    universe = _read_universe(args.universe)
    membership = _read_membership(args.membership)
    membership = membership.loc[
        (membership["date"] >= global_start) & (membership["date"] <= global_end)
    ].copy()

    universe = universe[universe["ticker"].isin(set(membership["ticker"]))].copy()
    if args.max_tickers is not None:
        universe = universe.head(args.max_tickers).copy()
        membership = membership[membership["ticker"].isin(set(universe["ticker"]))].copy()

    root = Path(args.out_root)
    cache = root / "cache"
    processed = root / "processed"
    quality = root / "quality"
    for p in (cache, processed, quality):
        p.mkdir(parents=True, exist_ok=True)

    kis = KisOpenApiClient.from_env(pause_seconds=args.pause_seconds)
    kis_market = KisMarketProvider(client=kis)
    pykrx_market = PykrxOhlcvOnlyProvider(pause_seconds=args.pykrx_fallback_pause)
    dart_client = OpenDartClient.from_env()

    mem_groups = {
        t: g.sort_values("date").copy()
        for t, g in membership.groupby("ticker", sort=False)
    }

    status_rows: list[dict[str, object]] = []

    for n, row in enumerate(universe.itertuples(index=False), start=1):
        ticker = str(row.ticker).zfill(6)
        corp_code = str(row.corp_code).zfill(8)
        mg = mem_groups.get(ticker)
        if mg is None or mg.empty:
            continue

        t_start = max(global_start, mg["date"].min())
        t_end = min(global_end, mg["date"].max())
        print(f"\n=== {n}/{len(universe)} {ticker} {t_start.date()}..{t_end.date()} ===", file=sys.stderr)

        market_path = cache / "market" / f"{ticker}.parquet"
        market_path.parent.mkdir(parents=True, exist_ok=True)

        if not args.skip_market and (args.refresh or not market_path.exists()):
            try:
                frame = kis_market.daily_stock_ohlcv(ticker, t_start, t_end)
                frame["market_provider"] = "KIS"
                atomic_write_parquet(frame, market_path)
                status_rows.append({"ticker": ticker, "stage": "market", "status": "ok", "detail": "KIS"})
            except Exception as kis_exc:
                print(f"[WARN KIS market] {ticker}: {kis_exc}", file=sys.stderr)
                try:
                    frame = pykrx_market.daily_stock_ohlcv(ticker, t_start, t_end)
                    frame["market_provider"] = "pykrx_ohlcv_fallback"
                    atomic_write_parquet(frame, market_path)
                    status_rows.append({"ticker": ticker, "stage": "market", "status": "ok", "detail": "pykrx_fallback"})
                except Exception as px_exc:
                    status_rows.append({
                        "ticker": ticker,
                        "stage": "market",
                        "status": "fail",
                        "detail": f"KIS={kis_exc}; pykrx={px_exc}",
                    })
                    print(f"[FAIL market] {ticker}: {px_exc}", file=sys.stderr)

        if _load(market_path) is None:
            status_rows.append({"ticker": ticker, "stage": "market_cache", "status": "fail", "detail": "missing"})
            continue

        dart_years = _dart_years_for_window(t_start, t_end)
        dart_success = 0
        for year in dart_years:
            dart_path = cache / "dart" / f"{ticker}_{year}_FY.parquet"
            dart_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not args.skip_dart and (args.refresh or not dart_path.exists()):
                    frame = dart_client.stock_total_status(corp_code, year, "FY", ticker=ticker)
                    atomic_write_parquet(frame, dart_path)
                if _load(dart_path) is not None:
                    dart_success += 1
                    status_rows.append({"ticker": ticker, "stage": f"dart_{year}", "status": "ok", "detail": ""})
            except OpenDartError as exc:
                status_rows.append({"ticker": ticker, "stage": f"dart_{year}", "status": "fail", "detail": str(exc)})
                print(f"[WARN dart {year}] {ticker}: {exc}", file=sys.stderr)

        if dart_success == 0:
            status_rows.append({"ticker": ticker, "stage": "dart_all", "status": "fail", "detail": "no annual snapshot"})

    status = pd.DataFrame(status_rows)
    status.to_csv(quality / "download_status.csv", index=False, encoding="utf-8-sig")

    complete: list[str] = []
    for row in universe.itertuples(index=False):
        ticker = str(row.ticker).zfill(6)
        mg = mem_groups.get(ticker)
        if mg is None or mg.empty:
            continue
        t_start = max(global_start, mg["date"].min())
        t_end = min(global_end, mg["date"].max())
        years = _dart_years_for_window(t_start, t_end)
        market_ok = (cache / "market" / f"{ticker}.parquet").exists()
        dart_ok = any((cache / "dart" / f"{ticker}_{y}_FY.parquet").exists() for y in years)
        if market_ok and dart_ok:
            complete.append(ticker)

    if not complete:
        raise SystemExit(f"No complete core tickers. See {quality / 'download_status.csv'}")

    market_frames = [pd.read_parquet(cache / "market" / f"{t}.parquet") for t in complete]
    market = pd.concat(market_frames, ignore_index=True)
    market["ticker"] = market["ticker"].astype(str).str.zfill(6)
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    # Exact KOSDAQ membership filter. This prevents rows outside the dates on which
    # the security actually belonged to KOSDAQ.
    member_keys = membership.loc[membership["ticker"].isin(complete), ["ticker", "date"]].copy()
    market = market.merge(member_keys, on=["ticker", "date"], how="inner", validate="many_to_one")
    market = market.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)

    dart_frames: list[pd.DataFrame] = []
    for t in complete:
        mg = mem_groups[t]
        years = _dart_years_for_window(max(global_start, mg["date"].min()), min(global_end, mg["date"].max()))
        for y in years:
            p = cache / "dart" / f"{t}_{y}_FY.parquet"
            if p.exists():
                dart_frames.append(pd.read_parquet(p))

    dart = pd.concat(dart_frames, ignore_index=True)
    dart = (
        dart.sort_values(["ticker", "asof_date"])
        .drop_duplicates(["ticker", "asof_date"], keep="last")
        .reset_index(drop=True)
    )

    market_trimmed, ff_coverage = trim_market_to_observable_free_float(
        market,
        dart,
        distributed_shares_col="distributed_shares",
    )
    ff_coverage.to_csv(
        quality / "free_float_coverage.csv", index=False, encoding="utf-8-sig"
    )

    observable = sorted(market_trimmed["ticker"].unique().tolist())
    if not observable:
        raise SystemExit("No tickers remained after point-in-time free-float trimming.")

    dart = dart[dart["ticker"].isin(observable)].copy()
    atomic_write_parquet(market_trimmed, processed / "market.parquet")
    atomic_write_parquet(dart, processed / "dart_share_status_history.parquet")

    master = build_stock_master(
        market_trimmed,
        dart,
        distributed_shares_col="distributed_shares",
        require_free_float=True,
    )
    atomic_write_parquet(master, processed / "stock_master.parquet")

    model = add_forward_targets(master)
    model = add_cooldown_events(model, cooldown=args.cooldown)
    atomic_write_parquet(model, processed / "stock_master_model.parquet")

    eligible = universe[universe["ticker"].isin(observable)].copy()
    eligible.to_csv(quality / "eligible_universe.csv", index=False, encoding="utf-8-sig")

    requested_exited = int(universe.get("exited_before_end", pd.Series(False, index=universe.index)).fillna(False).sum())
    complete_set = set(complete)
    observable_set = set(observable)
    exited_tickers = set(
        universe.loc[
            universe.get("exited_before_end", pd.Series(False, index=universe.index)).fillna(False),
            "ticker",
        ]
    )
    complete_exited = len(exited_tickers & complete_set)
    observable_exited = len(exited_tickers & observable_set)

    summary = pd.DataFrame(
        [{
            "requested_tickers": len(universe),
            "complete_market_plus_dart": len(complete),
            "observable_free_float_tickers": len(observable),
            "requested_exited_before_end": requested_exited,
            "complete_exited_before_end": complete_exited,
            "observable_exited_before_end": observable_exited,
            "exited_core_completion_rate": (
                complete_exited / requested_exited if requested_exited else float("nan")
            ),
            "exited_observable_ff_rate": (
                observable_exited / requested_exited if requested_exited else float("nan")
            ),
            "rows": len(model),
            "target_20d_30pct_ones": int((model["target_20d_30pct"] == 1).sum()),
            "surge_event_30pct_count": int(model["surge_event_30pct"].fillna(0).sum()),
            "start": global_start,
            "end": global_end,
        }]
    )
    summary.to_csv(quality / "pipeline_summary.csv", index=False)

    print("\n=== PIT KOSDAQ Core Pipeline Summary ===")
    print(summary.to_string(index=False))
    print(f"model -> {processed / 'stock_master_model.parquet'}")

    if requested_exited and observable_exited / requested_exited < 0.80:
        print(
            "\n[WARNING] Coverage of names that exited KOSDAQ before the end date is below 80%. "
            "Do not interpret the resulting strategy test as survivorship-bias-clean yet. "
            "Inspect download_status.csv and free_float_coverage.csv.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Per-ticker caches were preserved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
