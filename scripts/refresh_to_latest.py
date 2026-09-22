#!/usr/bin/env python3
from __future__ import annotations

import argparse, subprocess, sys
from pathlib import Path
import pandas as pd


def sh(cmd):
    print("\n$", " ".join(map(str,cmd)), flush=True)
    subprocess.run(list(map(str,cmd)), check=True)


def latest_market_date():
    import FinanceDataReader as fdr
    today=pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    start=today-pd.Timedelta(days=14)
    q=fdr.DataReader("KQ11", start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"))
    if q is None or q.empty:
        raise SystemExit("Could not determine latest KOSDAQ trading date from KQ11.")
    return pd.Timestamp(q.index.max()).normalize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--end",default="latest",help="YYYY-MM-DD or latest")
    p.add_argument("--start",default="2025-01-01")
    p.add_argument("--out-root",default="data/pit_kosdaq")
    p.add_argument("--skip-universe",action="store_true")
    a=p.parse_args()

    end=latest_market_date() if a.end=="latest" else pd.Timestamp(a.end).normalize()
    py=sys.executable
    root=Path(a.out_root)

    print("Latest refresh end =",end.date())

    if not a.skip_universe:
        sh([
            py,"scripts/build_pit_kosdaq_universe.py",
            "--start",a.start,
            "--end",end.strftime("%Y-%m-%d"),
            "--out-root",str(root),
        ])

    sh([
        py,"scripts/incremental_update_market_cache.py",
        "--universe",str(root/"universe/pit_pipeline_universe.csv"),
        "--membership",str(root/"universe/membership_daily.parquet"),
        "--core-root",str(root/"core"),
        "--start",a.start,
        "--end",end.strftime("%Y-%m-%d"),
    ])

    # Rebuild processed core using updated market cache.
    # Existing DART caches are reused; only missing DART files are requested.
    sh([
        py,"scripts/run_pit_kosdaq_core_pipeline.py",
        "--universe",str(root/"universe/pit_pipeline_universe.csv"),
        "--membership",str(root/"universe/membership_daily.parquet"),
        "--start",a.start,
        "--end",end.strftime("%Y-%m-%d"),
        "--out-root",str(root/"core"),
        "--skip-market",
    ])

    sh([
        py,"scripts/collect_market_state_kq11.py",
        "--start","2024-10-01",
        "--end",end.strftime("%Y-%m-%d"),
        "--out",str(root/"market_state/kosdaq_index_kq11.parquet"),
    ])

    sh([
        py,"scripts/analyze_market_state_v1.py",
        "--model",str(root/"core/processed/stock_master_model.parquet"),
        "--index",str(root/"market_state/kosdaq_index_kq11.parquet"),
        "--out-dir",str(root/"analysis/market_state_v1"),
    ])

    print("\n=== REFRESH COMPLETE ===")
    print("latest_model_date =",end.date())
    print("\nRun the frozen picker with:")
    print(
        f'{py} scripts/pick_stocks_by_date.py '
        f'--date latest '
        f'--model {root/"core/processed/stock_master_model.parquet"} '
        f'--market-state {root/"analysis/market_state_v1/market_state_daily.parquet"}'
    )
    print("\nNote: KRX short-balance is informational only. If its file is not refreshed,")
    print("the latest candidate may show SHORT=UNKNOWN; this does NOT change candidate selection.")

if __name__=="__main__":
    main()
