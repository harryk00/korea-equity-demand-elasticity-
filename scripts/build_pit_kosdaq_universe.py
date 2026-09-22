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

from korea_sd.data.dart_corp_codes import OpenDartCorpCodeProvider
from korea_sd.data.pit_universe import (
    FdrKrxCacheHistoricalUniverseProvider,
    OpenDartHistoricalCorpMapProvider,
    PitUniverseError,
    build_union_summary,
    recover_missing_corp_map_by_name,
    resolve_corp_map,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build PIT KOSDAQ membership from listing/delisting intervals using "
            "FinanceDataReader's public KRX cache; no pykrx/KRX login required."
        )
    )
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-08-29")
    p.add_argument("--out-root", default="data/pit_kosdaq")
    p.add_argument(
        "--cache-lookback-days",
        type=int,
        default=30,
        help="How many days before --end to search for a paired KRX cache snapshot.",
    )
    p.add_argument(
        "--dart-map-lookback-days",
        type=int,
        default=420,
        help="Search periodic disclosures this many days before start to recover delisted corp codes.",
    )
    p.add_argument(
        "--include-spac",
        action="store_true",
        help="Keep SPAC names in pipeline universe. Default excludes obvious SPACs.",
    )
    p.add_argument(
        "--deep-dart-start",
        default="2015-01-01",
        help=(
            "Earliest date for targeted historical DART recovery of unresolved "
            "delisted tickers."
        ),
    )
    p.add_argument(
        "--skip-deep-dart-recovery",
        action="store_true",
        help="Skip the targeted historical disclosure recovery pass.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.out_root)
    universe_dir = root / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise SystemExit("--end must be on or after --start")

    print("Building KOSDAQ PIT membership from listing/delisting intervals...", file=sys.stderr)

    provider = FdrKrxCacheHistoricalUniverseProvider(
        lookback_days=args.cache_lookback_days,
    )
    membership, intervals, cache_date = provider.build(
        start=start,
        end=end,
        market="KOSDAQ",
    )

    membership.to_parquet(
        universe_dir / "membership_daily.parquet",
        index=False,
    )
    intervals.to_csv(
        universe_dir / "membership_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [{
            "requested_start": start,
            "requested_end": end,
            "cache_snapshot_date": cache_date,
            "cache_staleness_days": int((end - cache_date).days),
            "membership_dates": membership["date"].nunique(),
            "membership_rows": len(membership),
            "interval_rows": len(intervals),
        }]
    ).to_csv(
        universe_dir / "cache_source_status.csv",
        index=False,
        encoding="utf-8-sig",
    )

    union = build_union_summary(membership)

    # Attach listing metadata/name from the reconstructed interval table.
    meta = (
        intervals.sort_values(["ticker", "listing_date"])
        .groupby("ticker", as_index=False)
        .agg(
            interval_name=("name", "last"),
            first_listing_date=("listing_date", "min"),
            last_delisting_date=("delisting_date", "max"),
        )
    )
    union = union.merge(meta, on="ticker", how="left")

    # OpenDART corpCode.xml contains every filing corporation's corp_code/corp_name,
    # but stock_code is populated only for listed companies. Keep both views:
    # - current_map: ticker-linked listed companies
    # - all_corp_map: includes delisted companies with blank stock_code, usable by name
    corp_provider = OpenDartCorpCodeProvider.from_env()

    # Keep the complete all-company table for delisted-name recovery.
    all_corp_map = corp_provider.download_all()

    # IMPORTANT: use the provider's backward-compatible listed-company view.
    # Do not reimplement the ticker regex here; the previous patch accidentally
    # used r"\\d{6}" and silently emptied this map.
    current_map = corp_provider.download()
    all_corp_map.to_parquet(
        universe_dir / "dart_corp_codes_all.parquet",
        index=False,
    )

    # Historical periodic disclosures recover corp codes for many names that
    # disappeared from the current ticker map after delisting.
    hist_provider = OpenDartHistoricalCorpMapProvider.from_env()
    hist_start = start - pd.Timedelta(days=args.dart_map_lookback_days)
    historical_map = hist_provider.search(hist_start, end)
    historical_map.to_parquet(
        universe_dir / "historical_dart_corp_map.parquet",
        index=False,
    )

    resolved = resolve_corp_map(union, current_map, historical_map)

    # Stage 1A: ordinary current-map + recent KOSDAQ periodic-disclosure map.
    stage1a = (
        resolved.groupby("exited_before_end")["corp_map_status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    stage1a.to_csv(
        universe_dir / "corp_map_stage1a_recent_before_deep.csv",
        encoding="utf-8-sig",
    )

    # Fail fast if the currently listed mapping unexpectedly collapses.
    active = resolved[resolved["exited_before_end"] == False]
    active_ok_rate = (
        float(active["corp_map_status"].eq("ok").mean()) if len(active) else 0.0
    )
    if len(current_map) < 1000 or active_ok_rate < 0.95:
        raise PitUniverseError(
            "DART current listed-company map failed sanity check: "
            f"current_map_rows={len(current_map)}, "
            f"active_ok_rate={active_ok_rate:.3%}. "
            "Refuse to continue because the universe mapping is corrupted."
        )

    # Targeted deep recovery:
    # The initial historical search used corp_cls='K'. A delisted corporation can
    # later be classified as 'E' (other), and a 420-day lookback can also be too
    # short. Scan periodic disclosures without a corp_cls filter, newest-first,
    # but retain ONLY the unresolved delisted tickers.
    deep_map = pd.DataFrame()
    unresolved = resolved[
        (resolved["exited_before_end"] == True)
        & (resolved["corp_map_status"] == "missing")
    ].copy()

    if len(unresolved) and not args.skip_deep_dart_recovery:
        deep_cache = universe_dir / "historical_dart_corp_map_deep.parquet"
        target_tickers = unresolved["ticker"].astype(str).str.zfill(6).tolist()

        deep_map = hist_provider.search_target_tickers(
            target_tickers=target_tickers,
            start=pd.Timestamp(args.deep_dart_start),
            end=end,
            corp_cls=None,
            pblntf_ty="A",
            last_reprt_at="Y",
            window_days=80,
            stop_when_all_found=True,
            progress_label="periodic_all_cls",
        )
        deep_map.to_parquet(deep_cache, index=False)

        if not deep_map.empty:
            combined_hist = pd.concat(
                [historical_map, deep_map],
                ignore_index=True,
                sort=False,
            )
            combined_hist = (
                combined_hist.sort_values(["ticker", "rcept_dt"])
                .drop_duplicates(
                    ["ticker", "corp_code", "rcept_dt", "report_nm"],
                    keep="last",
                )
                .reset_index(drop=True)
            )
            resolved = resolve_corp_map(union, current_map, combined_hist)
            combined_hist.to_parquet(
                universe_dir / "historical_dart_corp_map_combined.parquet",
                index=False,
            )

    stage1b = (
        resolved.groupby("exited_before_end")["corp_map_status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    stage1b.to_csv(
        universe_dir / "corp_map_stage1b_after_deep.csv",
        encoding="utf-8-sig",
    )

    # Delisted companies can remain in corpCode.xml with corp_code/corp_name but
    # blank stock_code. Recover only unique exact normalized KRX-name matches.
    resolved, name_recovery = recover_missing_corp_map_by_name(
        resolved,
        all_corp_map,
        source_name_col="interval_name",
    )
    name_recovery.to_csv(
        universe_dir / "corp_name_recovery_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    stage2 = (
        resolved.groupby("exited_before_end")["corp_map_status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    stage2.to_csv(
        universe_dir / "corp_map_stage2_after_name_recovery.csv",
        encoding="utf-8-sig",
    )

    # If DART mapping did not supply a name, preserve cache interval name so SPAC
    # filtering and diagnostics still work.
    if "interval_name" in resolved.columns:
        blank = resolved["corp_name"].fillna("").astype(str).str.strip().eq("")
        resolved.loc[blank, "corp_name"] = (
            resolved.loc[blank, "interval_name"].fillna("").astype(str)
        )
        names = resolved["corp_name"].fillna("").astype(str)
        resolved["is_spac"] = (
            names.str.contains("스팩", regex=False)
            | names.str.upper().str.contains("SPAC", regex=False)
            | names.str.contains("기업인수목적", regex=False)
        )

    resolved["eligible_for_pipeline"] = resolved["corp_map_status"].eq("ok")
    if not args.include_spac:
        resolved["eligible_for_pipeline"] &= ~resolved["is_spac"]

    resolved.to_csv(
        universe_dir / "pit_union_all.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pipeline = resolved.loc[resolved["eligible_for_pipeline"]].copy()
    pipeline.to_csv(
        universe_dir / "pit_pipeline_universe.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [{
            "start": start,
            "end": end,
            "cache_snapshot_date": cache_date,
            "snapshot_dates_with_membership": membership["date"].nunique(),
            "union_tickers": resolved["ticker"].nunique(),
            "active_on_last_snapshot": int(resolved["active_on_last_snapshot"].sum()),
            "exited_before_end": int(resolved["exited_before_end"].sum()),
            "corp_map_ok": int(resolved["corp_map_status"].eq("ok").sum()),
            "corp_map_missing": int(resolved["corp_map_status"].eq("missing").sum()),
            "corp_map_ambiguous": int(resolved["corp_map_status"].eq("ambiguous").sum()),
            "deep_dart_recovered": int(
                deep_map["ticker"].nunique() if not deep_map.empty else 0
            ),
            "name_recovered": int(
                resolved["corp_map_source"].fillna("").astype(str).str.contains(
                    "corpCode_name_exact", regex=False
                ).sum()
            ),
            "spac_count": int(resolved["is_spac"].sum()),
            "pipeline_tickers": len(pipeline),
        }]
    )
    summary.to_csv(universe_dir / "universe_summary.csv", index=False)

    print("\n=== PIT KOSDAQ Universe Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Corp-map Stage 1A: recent map before deep recovery ===")
    print(stage1a.to_string())

    print("\n=== Corp-map Stage 1B: after targeted historical recovery ===")
    print(stage1b.to_string())

    print("\n=== Corp-map Stage 2: after name recovery ===")
    print(stage2.to_string())

    print("\n=== Name recovery audit ===")
    if name_recovery.empty:
        print("no missing rows were sent to name recovery")
    else:
        print(name_recovery["recovery_status"].value_counts(dropna=False).to_string())

    print("\nSource:")
    print(
        f"- FinanceDataReader KRX cache snapshot: {cache_date.date()} "
        f"(requested end: {end.date()})"
    )
    print("- Membership reconstructed from ListingDate / DelistingDate intervals.")
    print("- No pykrx daily ticker-list calls and no KRX_ID/KRX_PW are used.")
    print(f"\nmembership -> {universe_dir / 'membership_daily.parquet'}")
    print(f"intervals -> {universe_dir / 'membership_intervals.csv'}")
    print(f"pipeline universe -> {universe_dir / 'pit_pipeline_universe.csv'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except PitUniverseError as exc:
        raise SystemExit(str(exc))
