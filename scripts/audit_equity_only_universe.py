#!/usr/bin/env python3
from __future__ import annotations

import io
from pathlib import Path
import pandas as pd
import requests


def main():
    path = Path("data/pit_kosdaq/universe/membership_intervals.csv")
    if not path.exists():
        print("Run build_pit_kosdaq_universe.py first.")
        return

    df = pd.read_csv(path, dtype={"ticker": str})
    df["ticker"] = df["ticker"].astype(str)

    valid = df["ticker"].str.fullmatch(r"\d{6}", na=False)

    print("\n=== Equity-only PIT interval audit ===")
    print("interval rows             :", len(df))
    print("six-digit equity codes    :", int(valid.sum()))
    print("non-six-digit codes       :", int((~valid).sum()))

    if (~valid).any():
        print("\nUnexpected non-six-digit codes:")
        print(
            df.loc[~valid, ["ticker", "name", "interval_source"]]
            .head(30)
            .to_string(index=False)
        )
        raise SystemExit(
            "FAIL: non-six-digit securities remain in the equity PIT universe."
        )

    print("\nPASS: all PIT interval tickers are six-digit equity short codes.")


if __name__ == "__main__":
    main()
