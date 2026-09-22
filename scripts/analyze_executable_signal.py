#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/pit_kosdaq/core/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/analysis/executable_signal",
    )
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--commission-bps", type=float, default=7.5)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--sell-tax-bps", type=float, default=0.0)
    return p.parse_args()


def require(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def quintile(s):
    return np.ceil(
        s.rank(method="average", pct=True) * 5
    ).clip(1, 5).astype("Int64")


def add_fixed_signal(df):
    df = df.copy()

    df["float_mcap_pct"] = (
        df.groupby("date")["free_float_market_cap"]
        .rank(method="average", pct=True, ascending=True)
    )
    df["turnover_pct"] = (
        df.groupby("date")["float_turnover_1d"]
        .rank(method="average", pct=True, ascending=True)
    )

    df["signal"] = (
        (df["float_mcap_pct"] <= 0.20)
        & (df["turnover_pct"] >= 0.80)
    )
    return df


def scan_trade(g, pos, tp, sl, max_hold, buy_cost, sell_cost):
    entry_pos = pos + 1
    if entry_pos >= len(g):
        return None

    s = g.iloc[pos]
    e = g.iloc[entry_pos]

    raw_open = float(e["open"])
    signal_close = float(s["close"])

    if not np.isfinite(raw_open) or raw_open <= 0:
        return None

    entry = raw_open * (1 + buy_cost)
    tp_px = entry * (1 + tp)
    sl_px = entry * (1 + sl)

    end_pos = min(entry_pos + max_hold - 1, len(g) - 1)
    w = g.iloc[entry_pos : end_pos + 1]

    exit_reason = "TIME"
    exit_raw = float(w.iloc[-1]["close"])
    exit_date = pd.Timestamp(w.iloc[-1]["date"])
    holding_bars = len(w)

    mfe = -np.inf
    mae = np.inf

    for i, r in enumerate(w.itertuples(index=False), start=1):
        o = float(r.open)
        h = float(r.high)
        l = float(r.low)

        if np.isfinite(h):
            mfe = max(mfe, h / entry - 1)
        if np.isfinite(l):
            mae = min(mae, l / entry - 1)

        # gap through stop
        if np.isfinite(o) and o <= sl_px:
            exit_reason = "SL_GAP"
            exit_raw = o
            exit_date = pd.Timestamp(r.date)
            holding_bars = i
            break

        hit_sl = np.isfinite(l) and l <= sl_px
        hit_tp = np.isfinite(h) and h >= tp_px

        # conservative same-bar assumption
        if hit_sl and hit_tp:
            exit_reason = "SL_SAME_BAR"
            exit_raw = sl_px
            exit_date = pd.Timestamp(r.date)
            holding_bars = i
            break
        if hit_sl:
            exit_reason = "SL"
            exit_raw = sl_px
            exit_date = pd.Timestamp(r.date)
            holding_bars = i
            break
        if hit_tp:
            exit_reason = "TP"
            exit_raw = tp_px
            exit_date = pd.Timestamp(r.date)
            holding_bars = i
            break

    exit_px = exit_raw * (1 - sell_cost)
    net_ret = exit_px / entry - 1

    gap = (
        raw_open / signal_close - 1
        if np.isfinite(signal_close) and signal_close > 0
        else np.nan
    )

    return {
        "ticker": s["ticker"],
        "signal_date": pd.Timestamp(s["date"]),
        "entry_date": pd.Timestamp(e["date"]),
        "exit_date": exit_date,
        "signal_close": signal_close,
        "entry_open_raw": raw_open,
        "entry_price": entry,
        "exit_price": exit_px,
        "exit_reason": exit_reason,
        "holding_bars": holding_bars,
        "next_open_gap": gap,
        "mfe_20d": mfe if np.isfinite(mfe) else np.nan,
        "mae_20d": mae if np.isfinite(mae) else np.nan,
        "trade_net_return": net_ret,
        "tp_before_sl": int(exit_reason == "TP"),
        "sl_before_tp": int(
            exit_reason in {"SL", "SL_GAP", "SL_SAME_BAR"}
        ),
        "time_exit": int(exit_reason == "TIME"),
        "target_20d_30pct": s.get("target_20d_30pct", np.nan),
        "free_float_market_cap": s["free_float_market_cap"],
        "float_turnover_1d": s["float_turnover_1d"],
        "float_mcap_pct": s["float_mcap_pct"],
        "turnover_pct": s["turnover_pct"],
    }


def summarize(g, name):
    if len(g) == 0:
        return {"sample": name, "signals": 0}

    pos = g.loc[g["trade_net_return"] > 0, "trade_net_return"].sum()
    neg = -g.loc[g["trade_net_return"] < 0, "trade_net_return"].sum()

    target1 = g[g["target_20d_30pct"] == 1]

    return {
        "sample": name,
        "signals": len(g),
        "tp_before_sl_rate": g["tp_before_sl"].mean(),
        "sl_before_tp_rate": g["sl_before_tp"].mean(),
        "time_exit_rate": g["time_exit"].mean(),
        "avg_trade_net_return": g["trade_net_return"].mean(),
        "median_trade_net_return": g["trade_net_return"].median(),
        "profit_factor_trade_sum": pos / neg if neg > 0 else np.nan,
        "avg_next_open_gap": g["next_open_gap"].mean(),
        "median_next_open_gap": g["next_open_gap"].median(),
        "avg_mfe_20d": g["mfe_20d"].mean(),
        "avg_mae_20d": g["mae_20d"].mean(),
        "original_target30_rate": g["target_20d_30pct"].mean(),
        "target1_but_no_tp_before_sl_rate": (
            1 - target1["tp_before_sl"].mean()
            if len(target1)
            else np.nan
        ),
    }


def grouped_summary(df, bucket_col):
    rows = []
    for b, g in df.groupby(bucket_col, dropna=False):
        r = summarize(g, str(b))
        r[bucket_col] = b
        rows.append(r)

    out = pd.DataFrame(rows)
    cols = [bucket_col] + [
        c for c in out.columns
        if c not in {bucket_col, "sample"}
    ]
    return out[cols].sort_values(bucket_col)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)

    require(
        df,
        [
            "ticker",
            "date",
            "open",
            "high",
            "low",
            "close",
            "free_float_market_cap",
            "float_turnover_1d",
        ],
    )

    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    df = add_fixed_signal(df)

    buy_cost = (
        args.commission_bps + args.slippage_bps
    ) / 10000.0

    sell_cost = (
        args.commission_bps
        + args.slippage_bps
        + args.sell_tax_bps
    ) / 10000.0

    rows = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.reset_index(drop=True)

        signal_positions = np.flatnonzero(
            g["signal"].fillna(False).to_numpy()
        )

        for pos in signal_positions:
            result = scan_trade(
                g,
                int(pos),
                args.tp,
                args.sl,
                args.max_hold,
                buy_cost,
                sell_cost,
            )
            if result is not None:
                rows.append(result)

    trades = pd.DataFrame(rows)

    if trades.empty:
        raise SystemExit("No executable signal rows.")

    trades["year"] = trades["signal_date"].dt.year

    # within-signal daily quintiles
    trades["turnover_signal_quintile"] = (
        trades.groupby("signal_date")["float_turnover_1d"]
        .transform(quintile)
    )

    trades["float_mcap_signal_quintile"] = (
        trades.groupby("signal_date")["free_float_market_cap"]
        .transform(quintile)
    )

    # global gap quintile for diagnostic use only
    trades["gap_quintile"] = quintile(
        trades["next_open_gap"]
    )

    summary = pd.DataFrame(
        [summarize(trades, "full")]
        + [
            summarize(g, str(year))
            for year, g in trades.groupby("year")
        ]
    )

    valid = trades["target_20d_30pct"].notna()
    cross = pd.crosstab(
        trades.loc[valid, "target_20d_30pct"].astype(int),
        trades.loc[valid, "tp_before_sl"].astype(int),
        margins=True,
    )

    by_turnover = grouped_summary(
        trades,
        "turnover_signal_quintile",
    )

    by_float = grouped_summary(
        trades,
        "float_mcap_signal_quintile",
    )

    by_gap = grouped_summary(
        trades,
        "gap_quintile",
    )

    grid = (
        trades.groupby(
            [
                "float_mcap_signal_quintile",
                "turnover_signal_quintile",
            ],
            dropna=False,
        )
        .agg(
            signals=("ticker", "size"),
            tp_before_sl_rate=("tp_before_sl", "mean"),
            avg_trade_net_return=("trade_net_return", "mean"),
            median_trade_net_return=("trade_net_return", "median"),
            avg_next_open_gap=("next_open_gap", "mean"),
            avg_mfe_20d=("mfe_20d", "mean"),
            avg_mae_20d=("mae_20d", "mean"),
        )
        .reset_index()
    )

    ranked = trades.copy()

    ranked["rank_turnover_desc"] = (
        ranked.groupby("signal_date")["float_turnover_1d"]
        .rank(method="first", ascending=False)
    )

    ranked["rank_float_small"] = (
        ranked.groupby("signal_date")["free_float_market_cap"]
        .rank(method="first", ascending=True)
    )

    topn_rows = []

    for col, label in [
        ("rank_turnover_desc", "turnover_desc"),
        ("rank_float_small", "float_mcap_small"),
    ]:
        for n in [1, 3, 5, 10]:
            g = ranked[ranked[col] <= n]
            r = summarize(g, f"{label}_top{n}")
            r["ranking"] = label
            r["top_n"] = n
            topn_rows.append(r)

    topn = pd.DataFrame(topn_rows)

    trades.to_parquet(
        out_dir / "signal_executable_events.parquet",
        index=False,
    )
    trades.to_csv(
        out_dir / "signal_executable_events.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    cross.to_csv(
        out_dir / "original_vs_executable_target.csv"
    )
    by_turnover.to_csv(
        out_dir / "by_turnover_signal_quintile.csv",
        index=False,
    )
    by_float.to_csv(
        out_dir / "by_float_mcap_signal_quintile.csv",
        index=False,
    )
    by_gap.to_csv(
        out_dir / "by_next_open_gap_quintile.csv",
        index=False,
    )
    grid.to_csv(
        out_dir / "signal_internal_5x5_grid.csv",
        index=False,
    )
    topn.to_csv(
        out_dir / "topn_candidate_quality.csv",
        index=False,
    )

    print("\n=== Executable Signal Summary ===")
    print(summary.to_string(index=False))

    print("\n=== Original Target vs TP-before-SL ===")
    print(cross.to_string())

    print("\n=== By Turnover Quintile Within Signal ===")
    print(by_turnover.to_string(index=False))

    print("\n=== By Free-Float-MCap Quintile Within Signal ===")
    print(by_float.to_string(index=False))

    print("\n=== By Next-Open Gap Quintile ===")
    print(by_gap.to_string(index=False))

    print("\n=== Simple Top-N Candidate Quality ===")
    print(topn.to_string(index=False))

    print("\nAssumptions:")
    print("- signal = free-float market cap bottom 20% AND turnover top 20%")
    print("- signal known after t close; enter at next available open")
    print(f"- TP +{args.tp:.0%}, SL {args.sl:.0%}, hold {args.max_hold} bars")
    print("- same-bar TP+SL -> stop first")
    print("- gap through stop -> exit at gap open")
    print(
        f"- commission {args.commission_bps:.1f} bps/side, "
        f"slippage {args.slippage_bps:.1f} bps/side, "
        f"sell tax {args.sell_tax_bps:.1f} bps"
    )
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
