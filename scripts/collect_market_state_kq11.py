#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-10-01")
    p.add_argument("--end", default="2026-08-29")
    p.add_argument("--out", default="data/pit_kosdaq/market_state/kosdaq_index_kq11.parquet")
    a = p.parse_args()

    try:
        import FinanceDataReader as fdr
    except ImportError:
        raise SystemExit(
            "FinanceDataReader is not installed. Run: pip install -U finance-datareader"
        )

    df = fdr.DataReader("KQ11", a.start, a.end)
    if df is None or df.empty:
        raise SystemExit("KQ11 download returned no rows.")

    df = df.reset_index()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"])

    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc == "open": rename[c] = "kq_open"
        elif lc == "high": rename[c] = "kq_high"
        elif lc == "low": rename[c] = "kq_low"
        elif lc == "close": rename[c] = "kq_close"
        elif lc == "volume": rename[c] = "kq_volume"
        elif lc == "change": rename[c] = "kq_change"
    df = df.rename(columns=rename)

    need = ["date", "kq_close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns after FDR download: {missing}; got={list(df.columns)}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    print("=== KOSDAQ INDEX COLLECTION ===")
    print("rows", len(df))
    print("start", df["date"].min().date())
    print("end", df["date"].max().date())
    print("columns", list(df.columns))
    print("wrote ->", out)

if __name__ == "__main__":
    main()
