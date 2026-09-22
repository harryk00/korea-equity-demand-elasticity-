#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="2025 discovery -> 2026 out-of-sample validation for Small Float + High Turnover."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
        help="Input model parquet.",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/oos",
        help="Output directory.",
    )
    p.add_argument(
        "--target",
        default="target_20d_30pct",
        help="Daily forward +30% label.",
    )
    p.add_argument(
        "--event",
        default="surge_event_30pct",
        help="Independent +30% event label.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Cross-sectional bins. Default=5.",
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=20,
        help="Minimum names per date to rank.",
    )
    p.add_argument(
        "--train-year",
        type=int,
        default=2025,
    )
    p.add_argument(
        "--test-year",
        type=int,
        default=2026,
    )
    return p.parse_args()


def cs_bucket(
    df: pd.DataFrame,
    col: str,
    bins: int,
    higher_is_stronger: bool,
    min_cross_section: int,
) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for _, idx in df.groupby("date", sort=False).groups.items():
        s = pd.to_numeric(df.loc[idx, col], errors="coerce")
        valid = s.dropna()

        if len(valid) < min_cross_section:
            continue

        pct = valid.rank(method="average", pct=True)

        if higher_is_stronger:
            bucket = np.ceil(pct * bins)
        else:
            bucket = np.ceil((1.000000001 - pct) * bins)

        bucket = np.clip(bucket, 1, bins).astype(int)
        out.loc[valid.index] = pd.Series(bucket, index=valid.index, dtype="Int64")

    return out


def make_signal(df: pd.DataFrame, bins: int, min_cross_section: int) -> pd.DataFrame:
    out = df.copy()

    # strongest bucket = 5
    # free_float_market_cap: smaller is stronger
    # float_turnover_1d: higher is stronger
    out["supply_bucket"] = cs_bucket(
        out,
        "free_float_market_cap",
        bins=bins,
        higher_is_stronger=False,
        min_cross_section=min_cross_section,
    )
    out["demand_bucket"] = cs_bucket(
        out,
        "float_turnover_1d",
        bins=bins,
        higher_is_stronger=True,
        min_cross_section=min_cross_section,
    )

    out["signal_smallfloat_highturnover"] = (
        (out["supply_bucket"] == bins) &
        (out["demand_bucket"] == bins)
    ).astype("Int64")

    return out


def daily_metrics(df: pd.DataFrame, target: str, event: str) -> dict:
    work = df.dropna(subset=[target]).copy()

    overall_rate = float(work[target].mean()) if len(work) else np.nan

    sig = work[work["signal_smallfloat_highturnover"] == 1].copy()
    nonsig = work[work["signal_smallfloat_highturnover"] == 0].copy()

    signal_rate = float(sig[target].mean()) if len(sig) else np.nan
    nonsignal_rate = float(nonsig[target].mean()) if len(nonsig) else np.nan

    rr_vs_overall = (
        signal_rate / overall_rate
        if pd.notna(signal_rate) and pd.notna(overall_rate) and overall_rate > 0
        else np.nan
    )

    rr_vs_nonsignal = (
        signal_rate / nonsignal_rate
        if pd.notna(signal_rate) and pd.notna(nonsignal_rate) and nonsignal_rate > 0
        else np.nan
    )

    return {
        "rows_with_target": int(len(work)),
        "overall_target_rate": overall_rate,
        "signal_observations": int(len(sig)),
        "signal_target_ones": int(sig[target].sum()) if len(sig) else 0,
        "signal_target_rate": signal_rate,
        "nonsignal_observations": int(len(nonsig)),
        "nonsignal_target_rate": nonsignal_rate,
        "relative_risk_vs_overall": rr_vs_overall,
        "relative_risk_vs_nonsignal": rr_vs_nonsignal,
        "signal_share_of_rows": float(len(sig) / len(work)) if len(work) else np.nan,
    }


def event_metrics(df: pd.DataFrame, target: str, event: str) -> dict:
    # Event recall:
    # among independent surge events, how many event dates themselves satisfy signal?
    event_rows = df[df[event] == 1].copy()
    caught = event_rows[event_rows["signal_smallfloat_highturnover"] == 1]

    recall = len(caught) / len(event_rows) if len(event_rows) else np.nan

    # Event precision-like metric:
    # among signal dates, how many are independent event dates?
    # This is intentionally separate from daily +30% target precision because
    # independent event labels are sparse by construction.
    signal_rows = df[df["signal_smallfloat_highturnover"] == 1].copy()
    event_precision = (
        float((signal_rows[event] == 1).mean())
        if len(signal_rows)
        else np.nan
    )

    return {
        "independent_events": int(len(event_rows)),
        "events_caught_on_event_date": int(len(caught)),
        "event_recall_on_event_date": recall,
        "signal_dates": int(len(signal_rows)),
        "event_precision_on_signal_date": event_precision,
    }


def event_window_recall(
    df: pd.DataFrame,
    event: str,
    lookback_days: int = 5,
) -> dict:
    """
    More practical recall:
    Was there at least one signal during t-lookback ... t,
    where t is the independent event date?
    """
    recalls = []

    for ticker, g in df.sort_values("date").groupby("ticker", sort=False):
        g = g.reset_index(drop=True)
        event_pos = np.flatnonzero(g[event].fillna(0).to_numpy() == 1)

        for pos in event_pos:
            start = max(0, pos - lookback_days)
            window = g.iloc[start:pos + 1]
            caught = bool((window["signal_smallfloat_highturnover"] == 1).any())
            recalls.append({
                "ticker": ticker,
                "event_date": g.loc[pos, "date"],
                "caught_within_lookback": int(caught),
            })

    detail = pd.DataFrame(recalls)

    if detail.empty:
        return {
            "detail": detail,
            "event_count": 0,
            "caught_count": 0,
            "recall": np.nan,
        }

    return {
        "detail": detail,
        "event_count": int(len(detail)),
        "caught_count": int(detail["caught_within_lookback"].sum()),
        "recall": float(detail["caught_within_lookback"].mean()),
    }


def yearly_summary(df: pd.DataFrame, target: str, event: str) -> pd.DataFrame:
    rows = []

    for year, g in df.groupby(df["date"].dt.year):
        d = daily_metrics(g, target, event)
        e = event_metrics(g, target, event)

        rows.append({
            "year": int(year),
            **d,
            **e,
        })

    return pd.DataFrame(rows).sort_values("year")


def main():
    args = parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {
        "ticker",
        "date",
        "free_float_market_cap",
        "float_turnover_1d",
        args.target,
        args.event,
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = make_signal(
        df,
        bins=args.bins,
        min_cross_section=args.min_cross_section,
    )

    train = df[df["date"].dt.year == args.train_year].copy()
    test = df[df["date"].dt.year == args.test_year].copy()

    train_daily = daily_metrics(train, args.target, args.event)
    test_daily = daily_metrics(test, args.target, args.event)

    train_event = event_metrics(train, args.target, args.event)
    test_event = event_metrics(test, args.target, args.event)

    train_window = event_window_recall(train, args.event, lookback_days=5)
    test_window = event_window_recall(test, args.event, lookback_days=5)

    summary = pd.DataFrame([
        {
            "sample": f"train_{args.train_year}",
            **train_daily,
            **train_event,
            "event_recall_tminus5_to_t": train_window["recall"],
            "events_caught_tminus5_to_t": train_window["caught_count"],
        },
        {
            "sample": f"test_{args.test_year}",
            **test_daily,
            **test_event,
            "event_recall_tminus5_to_t": test_window["recall"],
            "events_caught_tminus5_to_t": test_window["caught_count"],
        },
    ])

    summary.to_csv(out_dir / "oos_summary.csv", index=False)

    yearly = yearly_summary(df, args.target, args.event)
    yearly.to_csv(out_dir / "yearly_summary.csv", index=False)

    cols_to_save = [
        "ticker",
        "date",
        "free_float_market_cap",
        "float_turnover_1d",
        "supply_bucket",
        "demand_bucket",
        "signal_smallfloat_highturnover",
        args.target,
        args.event,
    ]

    df[cols_to_save].to_parquet(
        out_dir / "signal_panel.parquet",
        index=False,
    )

    if not train_window["detail"].empty:
        train_window["detail"].to_csv(
            out_dir / f"event_recall_detail_{args.train_year}.csv",
            index=False,
        )

    if not test_window["detail"].empty:
        test_window["detail"].to_csv(
            out_dir / f"event_recall_detail_{args.test_year}.csv",
            index=False,
        )

    print("\n=== OOS Summary ===")
    display_cols = [
        "sample",
        "rows_with_target",
        "overall_target_rate",
        "signal_observations",
        "signal_target_rate",
        "relative_risk_vs_overall",
        "relative_risk_vs_nonsignal",
        "independent_events",
        "events_caught_on_event_date",
        "event_recall_on_event_date",
        "event_recall_tminus5_to_t",
        "events_caught_tminus5_to_t",
    ]
    print(summary[display_cols].to_string(index=False))

    print("\n=== Interpretation helpers ===")
    if len(test):
        print(
            f"{args.test_year} signal +30% rate: "
            f"{test_daily['signal_target_rate']:.4f}"
            if pd.notna(test_daily["signal_target_rate"])
            else f"{args.test_year} signal +30% rate: NaN"
        )
        print(
            f"{args.test_year} overall +30% rate: "
            f"{test_daily['overall_target_rate']:.4f}"
            if pd.notna(test_daily["overall_target_rate"])
            else f"{args.test_year} overall +30% rate: NaN"
        )
        print(
            f"{args.test_year} relative risk vs overall: "
            f"{test_daily['relative_risk_vs_overall']:.3f}x"
            if pd.notna(test_daily["relative_risk_vs_overall"])
            else f"{args.test_year} relative risk vs overall: NaN"
        )
        print(
            f"{args.test_year} event recall (t-5..t): "
            f"{test_window['recall']:.3%}"
            if pd.notna(test_window["recall"])
            else f"{args.test_year} event recall (t-5..t): NaN"
        )

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
