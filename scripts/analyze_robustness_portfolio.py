#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import itertools
import math

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness + transaction cost + portfolio backtest."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/robustness_portfolio",
    )
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--min-cross-section", type=int, default=20)

    p.add_argument(
        "--stop-losses",
        nargs="*",
        type=float,
        default=[0.10, 0.125, 0.15, 0.175, 0.20],
    )
    p.add_argument(
        "--take-profits",
        nargs="*",
        type=float,
        default=[0.20, 0.25, 0.30, 0.40],
    )
    p.add_argument(
        "--max-holds",
        nargs="*",
        type=int,
        default=[10, 15, 20, 30],
    )

    p.add_argument(
        "--commission-bps",
        type=float,
        default=15.0,
        help="Round-trip commission/fees in basis points.",
    )
    p.add_argument(
        "--slippage-bps",
        type=float,
        default=20.0,
        help="Round-trip slippage in basis points.",
    )

    p.add_argument(
        "--portfolio-stop-loss",
        type=float,
        default=0.15,
    )
    p.add_argument(
        "--portfolio-take-profit",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--portfolio-max-hold",
        type=int,
        default=20,
    )
    p.add_argument(
        "--max-positions",
        type=int,
        default=5,
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


def resolve_intraday_exit(high, low, tp_price, sl_price):
    hit_tp = pd.notna(high) and high >= tp_price
    hit_sl = pd.notna(low) and low <= sl_price

    if hit_tp and hit_sl:
        return "stop_same_bar", sl_price
    if hit_sl:
        return "stop", sl_price
    if hit_tp:
        return "take_profit", tp_price
    return None, None


def make_nonoverlap_trades(
    df,
    stop_loss,
    take_profit,
    max_hold,
    round_trip_cost,
):
    trades = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        next_allowed_signal_pos = 0

        for sig_pos in range(len(g)):
            if sig_pos < next_allowed_signal_pos:
                continue
            if g.loc[sig_pos, "signal"] != 1:
                continue

            entry_pos = sig_pos + 1
            if entry_pos >= len(g):
                continue

            entry_price = float(g.loc[entry_pos, "open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

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

            gross_return = exit_price / entry_price - 1
            net_return = gross_return - round_trip_cost

            trades.append({
                "ticker": ticker,
                "signal_date": g.loc[sig_pos, "date"],
                "entry_date": g.loc[entry_pos, "date"],
                "exit_date": g.loc[exit_pos, "date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "holding_days": exit_pos - entry_pos + 1,
                "gross_return": gross_return,
                "net_return": net_return,
                "mfe": mfe,
                "mae": mae,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "max_hold": max_hold,
                "entry_year": int(pd.Timestamp(g.loc[entry_pos, "date"]).year),
            })

            next_allowed_signal_pos = exit_pos + 1

    return pd.DataFrame(trades)


def trade_summary(trades, label):
    if trades.empty:
        return {"sample": label, "trades": 0}

    r = trades["net_return"]
    wins = r > 0
    losses = r < 0

    gross_profit = r[wins].sum()
    gross_loss = -r[losses].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

    return {
        "sample": label,
        "trades": int(len(trades)),
        "win_rate": float(wins.mean()),
        "avg_net_return": float(r.mean()),
        "median_net_return": float(r.median()),
        "profit_factor": float(pf) if pd.notna(pf) else np.nan,
        "avg_holding_days": float(trades["holding_days"].mean()),
        "avg_mfe": float(trades["mfe"].mean()),
        "avg_mae": float(trades["mae"].mean()),
        "tp_exit_rate": float(trades["exit_reason"].eq("take_profit").mean()),
        "stop_exit_rate": float(
            trades["exit_reason"].isin(["stop", "stop_same_bar"]).mean()
        ),
        "time_exit_rate": float(trades["exit_reason"].eq("time_exit").mean()),
    }


def sensitivity_analysis(df, args, round_trip_cost):
    rows = []
    all_trade_frames = []

    combos = list(itertools.product(
        args.stop_losses,
        args.take_profits,
        args.max_holds,
    ))

    for sl, tp, hold in combos:
        trades = make_nonoverlap_trades(
            df,
            stop_loss=sl,
            take_profit=tp,
            max_hold=hold,
            round_trip_cost=round_trip_cost,
        )

        if trades.empty:
            continue

        trades["param_id"] = f"SL{sl:.3f}_TP{tp:.3f}_H{hold}"
        all_trade_frames.append(trades)

        for year in ["all", 2025, 2026]:
            if year == "all":
                t = trades
                label = "all"
            else:
                t = trades[trades["entry_year"] == year]
                label = str(year)

            s = trade_summary(t, label)
            s.update({
                "stop_loss": sl,
                "take_profit": tp,
                "max_hold": hold,
            })
            rows.append(s)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary["score"] = (
            summary["avg_net_return"].fillna(-999)
            * np.log1p(summary["trades"].fillna(0))
        )

    all_trades = (
        pd.concat(all_trade_frames, ignore_index=True)
        if all_trade_frames else pd.DataFrame()
    )

    return summary, all_trades


def build_trade_candidates_for_portfolio(
    df,
    stop_loss,
    take_profit,
    max_hold,
    round_trip_cost,
):
    """
    Build every first-entry candidate per ticker without portfolio capital constraints.
    These candidates are later accepted/rejected by the portfolio simulator.
    """
    candidates = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        for sig_pos in range(len(g) - 1):
            if g.loc[sig_pos, "signal"] != 1:
                continue

            entry_pos = sig_pos + 1
            entry_price = float(g.loc[entry_pos, "open"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue

            tp_price = entry_price * (1 + take_profit)
            sl_price = entry_price * (1 - stop_loss)
            last_pos = min(entry_pos + max_hold - 1, len(g) - 1)

            exit_reason = "time_exit"
            exit_pos = last_pos
            exit_price = float(g.loc[last_pos, "close"])

            for pos in range(entry_pos, last_pos + 1):
                high = float(g.loc[pos, "high"])
                low = float(g.loc[pos, "low"])

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

            gross_return = exit_price / entry_price - 1
            net_return = gross_return - round_trip_cost

            candidates.append({
                "ticker": ticker,
                "signal_date": g.loc[sig_pos, "date"],
                "entry_date": g.loc[entry_pos, "date"],
                "exit_date": g.loc[exit_pos, "date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": net_return,
                "exit_reason": exit_reason,
                "holding_days": exit_pos - entry_pos + 1,
                "signal_turnover": float(g.loc[sig_pos, "float_turnover_1d"]),
                "signal_float_mcap": float(g.loc[sig_pos, "free_float_market_cap"]),
            })

    return pd.DataFrame(candidates)


def portfolio_simulation(candidates, max_positions):
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    c = candidates.copy()
    c["entry_date"] = pd.to_datetime(c["entry_date"])
    c["exit_date"] = pd.to_datetime(c["exit_date"])

    # Rank same-day candidates by stronger turnover first, then smaller free-float mcap.
    c = c.sort_values(
        ["entry_date", "signal_turnover", "signal_float_mcap"],
        ascending=[True, False, True],
    )

    accepted = []
    active = []

    for entry_date, day in c.groupby("entry_date", sort=True):
        # Remove positions already exited before this entry date.
        active = [x for x in active if x["exit_date"] >= entry_date]

        active_tickers = {x["ticker"] for x in active}
        slots = max_positions - len(active)

        if slots <= 0:
            continue

        for _, row in day.iterrows():
            if slots <= 0:
                break
            if row["ticker"] in active_tickers:
                continue

            rec = row.to_dict()
            accepted.append(rec)
            active.append(rec)
            active_tickers.add(row["ticker"])
            slots -= 1

    trades = pd.DataFrame(accepted)
    if trades.empty:
        return trades, pd.DataFrame(), {}

    # Each accepted trade receives a fixed 1/max_positions fraction of starting equity.
    # Capital not used by open slots remains cash.
    trades["portfolio_weight"] = 1.0 / max_positions
    trades["portfolio_pnl"] = (
        trades["portfolio_weight"] * trades["net_return"]
    )

    # Realized-equity curve by exit date. This is conservative/simple and avoids
    # inventing intraday mark-to-market paths from daily bars.
    realized = (
        trades.groupby("exit_date", as_index=False)["portfolio_pnl"]
        .sum()
        .sort_values("exit_date")
    )

    realized["equity"] = 1.0 + realized["portfolio_pnl"].cumsum()
    realized["peak"] = realized["equity"].cummax()
    realized["drawdown"] = realized["equity"] / realized["peak"] - 1

    start = trades["entry_date"].min()
    end = trades["exit_date"].max()
    years = max((end - start).days / 365.25, 1 / 365.25)

    ending_equity = float(realized["equity"].iloc[-1])
    cagr = (
        ending_equity ** (1 / years) - 1
        if ending_equity > 0 else np.nan
    )

    # Dailyized Sharpe based on realized PnL days expanded onto business-day calendar.
    idx = pd.date_range(start, end, freq="B")
    daily = realized.set_index("exit_date")["portfolio_pnl"].reindex(idx, fill_value=0.0)

    sharpe = np.nan
    if daily.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))

    monthly = daily.resample("ME").sum()
    monthly_win_rate = float((monthly > 0).mean()) if len(monthly) else np.nan

    stats = {
        "accepted_trades": int(len(trades)),
        "max_positions": int(max_positions),
        "ending_equity_simple": ending_equity,
        "total_return_simple": ending_equity - 1,
        "cagr_simple": float(cagr) if pd.notna(cagr) else np.nan,
        "max_drawdown_realized": float(realized["drawdown"].min()),
        "sharpe_realized_pnl_days": sharpe,
        "monthly_win_rate": monthly_win_rate,
        "avg_trade_net_return": float(trades["net_return"].mean()),
        "median_trade_net_return": float(trades["net_return"].median()),
        "trade_win_rate": float((trades["net_return"] > 0).mean()),
    }

    return trades, realized, stats


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
        "free_float_market_cap", "float_turnover_1d",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    df = add_signal(
        df,
        bins=args.bins,
        min_cross_section=args.min_cross_section,
    )

    round_trip_cost = (
        args.commission_bps + args.slippage_bps
    ) / 10000.0

    sensitivity, sensitivity_trades = sensitivity_analysis(
        df,
        args,
        round_trip_cost=round_trip_cost,
    )

    sensitivity.to_csv(
        out_dir / "parameter_sensitivity.csv",
        index=False,
    )

    if not sensitivity_trades.empty:
        sensitivity_trades.to_csv(
            out_dir / "parameter_sensitivity_trades.csv",
            index=False,
        )

    candidates = build_trade_candidates_for_portfolio(
        df,
        stop_loss=args.portfolio_stop_loss,
        take_profit=args.portfolio_take_profit,
        max_hold=args.portfolio_max_hold,
        round_trip_cost=round_trip_cost,
    )

    candidates.to_csv(
        out_dir / "portfolio_candidates.csv",
        index=False,
    )

    portfolio_trades, equity_curve, portfolio_stats = portfolio_simulation(
        candidates,
        max_positions=args.max_positions,
    )

    portfolio_trades.to_csv(
        out_dir / "portfolio_trades.csv",
        index=False,
    )
    equity_curve.to_csv(
        out_dir / "portfolio_equity_realized.csv",
        index=False,
    )

    pd.DataFrame([portfolio_stats]).to_csv(
        out_dir / "portfolio_summary.csv",
        index=False,
    )

    print("\n=== Transaction Cost Assumption ===")
    print(f"commission/fees round-trip: {args.commission_bps:.1f} bps")
    print(f"slippage round-trip       : {args.slippage_bps:.1f} bps")
    print(f"total round-trip cost     : {(round_trip_cost * 100):.3f}%")

    print("\n=== Parameter Sensitivity: 2026 top 15 by avg net return ===")
    if sensitivity.empty:
        print("No sensitivity results.")
    else:
        test = sensitivity[
            (sensitivity["sample"] == "2026")
            & (sensitivity["trades"] >= 20)
        ].copy()

        test = test.sort_values(
            ["avg_net_return", "profit_factor"],
            ascending=[False, False],
        )

        cols = [
            "stop_loss",
            "take_profit",
            "max_hold",
            "trades",
            "win_rate",
            "avg_net_return",
            "median_net_return",
            "profit_factor",
            "avg_holding_days",
        ]
        print(test[cols].head(15).to_string(index=False))

        print("\n=== Parameter neighborhood around SL15 / TP30 ===")
        neighborhood = sensitivity[
            (sensitivity["sample"].isin(["2025", "2026"]))
            & (sensitivity["stop_loss"].isin([0.125, 0.15, 0.175]))
            & (sensitivity["take_profit"].isin([0.25, 0.30, 0.40]))
            & (sensitivity["max_hold"].isin([15, 20, 30]))
        ].copy()

        print(
            neighborhood[
                [
                    "sample",
                    "stop_loss",
                    "take_profit",
                    "max_hold",
                    "trades",
                    "win_rate",
                    "avg_net_return",
                    "profit_factor",
                ]
            ].sort_values(
                ["sample", "stop_loss", "take_profit", "max_hold"]
            ).to_string(index=False)
        )

    print("\n=== Portfolio Summary ===")
    if portfolio_stats:
        for k, v in portfolio_stats.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
    else:
        print("No portfolio trades.")

    print("\nPortfolio assumptions:")
    print("- Signal known at t close; entry at t+1 open.")
    print(f"- SL {args.portfolio_stop_loss:.1%}, TP {args.portfolio_take_profit:.1%}, hold {args.portfolio_max_hold}d.")
    print(f"- Maximum simultaneous positions: {args.max_positions}.")
    print("- Equal 1/max_positions capital per accepted trade; unused slots stay cash.")
    print("- Same-day candidates ranked by higher turnover, then smaller float mcap.")
    print("- Portfolio equity is REALIZED-PnL based, not full daily mark-to-market.")
    print("- CAGR/MDD/Sharpe are therefore diagnostics, not final production statistics.")

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
