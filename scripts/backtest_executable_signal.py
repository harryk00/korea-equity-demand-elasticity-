#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Executable backtest for Small Float + High Turnover signal."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/backtest",
    )
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--min-cross-section", type=int, default=20)
    p.add_argument("--take-profit", type=float, default=0.30)
    p.add_argument(
        "--stop-losses",
        nargs="*",
        type=float,
        default=[0.10, 0.15],
        help="Positive fractions, e.g. 0.10 0.15",
    )
    p.add_argument("--max-hold", type=int, default=20)
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


def add_signal(df, bins, min_cross_section):
    out = df.copy()

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

    out["signal"] = (
        (out["supply_bucket"] == bins)
        & (out["demand_bucket"] == bins)
    ).astype("Int64")

    return out


def resolve_intraday_exit(
    high: float,
    low: float,
    tp_price: float,
    sl_price: float,
):
    """
    Conservative rule if TP and SL are both touched on the same daily bar:
    assume stop-loss happened first because intraday path is unavailable.
    """
    hit_tp = pd.notna(high) and high >= tp_price
    hit_sl = pd.notna(low) and low <= sl_price

    if hit_tp and hit_sl:
        return "stop_same_bar", sl_price
    if hit_sl:
        return "stop", sl_price
    if hit_tp:
        return "take_profit", tp_price
    return None, None


def build_trades_for_stop(
    df: pd.DataFrame,
    stop_loss: float,
    take_profit: float,
    max_hold: int,
) -> pd.DataFrame:
    trades = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        # Prevent overlapping positions in the same ticker.
        next_allowed_signal_pos = 0

        for sig_pos in range(len(g)):
            if sig_pos < next_allowed_signal_pos:
                continue
            if g.loc[sig_pos, "signal"] != 1:
                continue

            entry_pos = sig_pos + 1
            if entry_pos >= len(g):
                continue

            entry_price = pd.to_numeric(
                pd.Series([g.loc[entry_pos, "open"]]), errors="coerce"
            ).iloc[0]

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            entry_date = g.loc[entry_pos, "date"]
            signal_date = g.loc[sig_pos, "date"]

            tp_price = entry_price * (1 + take_profit)
            sl_price = entry_price * (1 - stop_loss)

            last_pos = min(entry_pos + max_hold - 1, len(g) - 1)

            exit_reason = "time_exit"
            exit_pos = last_pos
            exit_price = float(g.loc[last_pos, "close"])

            mfe = -np.inf
            mae = np.inf

            for pos in range(entry_pos, last_pos + 1):
                high = float(g.loc[pos, "high"])
                low = float(g.loc[pos, "low"])

                mfe = max(mfe, high / entry_price - 1)
                mae = min(mae, low / entry_price - 1)

                reason, px = resolve_intraday_exit(
                    high=high,
                    low=low,
                    tp_price=tp_price,
                    sl_price=sl_price,
                )

                if reason is not None:
                    exit_reason = reason
                    exit_pos = pos
                    exit_price = float(px)
                    break

            exit_date = g.loc[exit_pos, "date"]
            trade_return = exit_price / entry_price - 1
            hold_days = exit_pos - entry_pos + 1

            trades.append({
                "ticker": ticker,
                "signal_date": signal_date,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "holding_days": hold_days,
                "trade_return": trade_return,
                "mfe": mfe,
                "mae": mae,
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "signal_year": int(pd.Timestamp(signal_date).year),
                "entry_year": int(pd.Timestamp(entry_date).year),
                "supply_bucket": int(g.loc[sig_pos, "supply_bucket"]),
                "demand_bucket": int(g.loc[sig_pos, "demand_bucket"]),
                "free_float_market_cap": g.loc[sig_pos, "free_float_market_cap"],
                "float_turnover_1d": g.loc[sig_pos, "float_turnover_1d"],
            })

            # No second position until the current one is exited.
            # A new signal can be considered only after exit day.
            next_allowed_signal_pos = exit_pos + 1

    return pd.DataFrame(trades)


def max_drawdown_from_trade_sequence(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def summarize(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {
            "sample": label,
            "trades": 0,
        }

    r = trades["trade_return"]
    wins = r > 0
    losses = r < 0

    gross_profit = r[wins].sum()
    gross_loss = -r[losses].sum()

    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else np.nan
    )

    return {
        "sample": label,
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()),
        "avg_return": float(r.mean()),
        "median_return": float(r.median()),
        "profit_factor": float(profit_factor) if pd.notna(profit_factor) else np.nan,
        "avg_holding_days": float(trades["holding_days"].mean()),
        "median_holding_days": float(trades["holding_days"].median()),
        "avg_mfe": float(trades["mfe"].mean()),
        "median_mfe": float(trades["mfe"].median()),
        "avg_mae": float(trades["mae"].mean()),
        "median_mae": float(trades["mae"].median()),
        "tp_exit_rate": float(
            trades["exit_reason"].eq("take_profit").mean()
        ),
        "stop_exit_rate": float(
            trades["exit_reason"].isin(["stop", "stop_same_bar"]).mean()
        ),
        "time_exit_rate": float(
            trades["exit_reason"].eq("time_exit").mean()
        ),
        "max_drawdown_trade_sequence": max_drawdown_from_trade_sequence(r),
    }


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
        "ticker", "date", "open", "high", "low", "close",
        "free_float_market_cap", "float_turnover_1d"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"Missing required columns: {missing}\n"
            "The executable backtest needs OHLC + signal features."
        )

    df = add_signal(
        df,
        bins=args.bins,
        min_cross_section=args.min_cross_section,
    )

    all_trades = []
    summaries = []

    for sl in args.stop_losses:
        trades = build_trades_for_stop(
            df,
            stop_loss=sl,
            take_profit=args.take_profit,
            max_hold=args.max_hold,
        )

        if trades.empty:
            continue

        sl_tag = f"{int(round(sl * 100))}pct"
        trades.to_csv(
            out_dir / f"trades_sl_{sl_tag}.csv",
            index=False,
        )

        all_trades.append(trades)

        summaries.append(
            summarize(
                trades,
                label=f"all_SL{sl:.0%}",
            )
        )

        for year in sorted(trades["entry_year"].unique()):
            y = trades[trades["entry_year"] == year].copy()
            summaries.append(
                summarize(
                    y,
                    label=f"{year}_SL{sl:.0%}",
                )
            )

    summary = pd.DataFrame(summaries)

    if not summary.empty:
        summary.to_csv(out_dir / "backtest_summary.csv", index=False)

    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            out_dir / "all_trades.csv",
            index=False,
        )

    signal_panel_cols = [
        "ticker", "date", "open", "high", "low", "close",
        "free_float_market_cap", "float_turnover_1d",
        "supply_bucket", "demand_bucket", "signal",
    ]
    df[signal_panel_cols].to_parquet(
        out_dir / "backtest_signal_panel.parquet",
        index=False,
    )

    print("\n=== Executable Backtest Summary ===")
    if summary.empty:
        print("No trades generated.")
    else:
        show = [
            "sample",
            "trades",
            "win_rate",
            "avg_return",
            "median_return",
            "profit_factor",
            "avg_holding_days",
            "avg_mfe",
            "avg_mae",
            "tp_exit_rate",
            "stop_exit_rate",
            "time_exit_rate",
            "max_drawdown_trade_sequence",
        ]
        print(summary[show].to_string(index=False))

    print("\nExecution assumptions:")
    print("- Signal is known after day t closes.")
    print("- Entry is next trading day's OPEN.")
    print(f"- Take profit: +{args.take_profit:.0%}.")
    print(f"- Stop losses tested: {[f'-{x:.0%}' for x in args.stop_losses]}.")
    print(f"- Max holding period: {args.max_hold} trading days.")
    print("- Same ticker cannot open overlapping positions.")
    print("- If TP and SL are both touched on one daily bar, STOP is assumed first (conservative).")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
