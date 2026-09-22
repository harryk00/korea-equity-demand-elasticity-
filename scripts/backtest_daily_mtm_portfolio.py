#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Daily mark-to-market portfolio backtest for Small Float + High Turnover."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/mtm_portfolio",
    )
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--min-cross-section", type=int, default=20)

    p.add_argument("--stop-loss", type=float, default=0.15)
    p.add_argument("--take-profit", type=float, default=0.30)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--max-positions", type=int, default=5)

    p.add_argument("--initial-capital", type=float, default=100_000_000.0)

    # Prior robustness test used 35 bps round-trip total.
    # Split here into per-side commission + slippage.
    p.add_argument(
        "--commission-bps-per-side",
        type=float,
        default=7.5,
    )
    p.add_argument(
        "--slippage-bps-per-side",
        type=float,
        default=10.0,
    )
    p.add_argument(
        "--sell-tax-bps",
        type=float,
        default=0.0,
        help="Optional sell-side tax in bps. Default 0; set explicitly if desired.",
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


def add_signal_and_entry_candidates(
    df: pd.DataFrame,
    bins: int,
    min_cross_section: int,
) -> pd.DataFrame:
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

    # Signal is known only after t close. Candidate entry is ticker's next available bar.
    out = out.sort_values(["ticker", "date"]).copy()
    grp = out.groupby("ticker", sort=False)

    out["entry_candidate"] = grp["signal"].shift(1).fillna(0).astype("Int64")
    out["signal_date_for_entry"] = grp["date"].shift(1)
    out["signal_turnover_for_entry"] = grp["float_turnover_1d"].shift(1)
    out["signal_float_mcap_for_entry"] = grp["free_float_market_cap"].shift(1)

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass
class Position:
    ticker: str
    shares: int
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_raw_open: float
    entry_fill: float
    entry_commission: float
    entry_cost_total: float
    tp_price: float
    sl_price: float
    days_held: int
    last_close: float
    signal_turnover: float
    signal_float_mcap: float


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def exit_trigger_for_bar(
    position: Position,
    bar: pd.Series,
):
    """
    Conservative daily-bar execution.

    Existing position:
    - gap below stop -> exit at open
    - gap above TP -> conservatively exit at TP threshold
    - if both TP and SL touched intraday, assume stop first

    New position:
    - same logic after the position is entered at today's open; a stop/TP may occur same day.
    """
    o = safe_float(bar["open"])
    h = safe_float(bar["high"])
    l = safe_float(bar["low"])

    if np.isnan(o) or np.isnan(h) or np.isnan(l):
        return None, None

    if o <= position.sl_price:
        return "stop_gap", o

    if o >= position.tp_price and position.entry_date != bar["date"]:
        # Standing TP order; use threshold rather than favorable gap-open price.
        return "take_profit_gap", position.tp_price

    hit_tp = h >= position.tp_price
    hit_sl = l <= position.sl_price

    if hit_tp and hit_sl:
        return "stop_same_bar", position.sl_price
    if hit_sl:
        return "stop", position.sl_price
    if hit_tp:
        return "take_profit", position.tp_price

    return None, None


def calculate_stats(equity: pd.DataFrame, trades: pd.DataFrame, initial_capital: float):
    if equity.empty:
        return {}

    e = equity.sort_values("date").copy()
    e["daily_return"] = e["equity"].pct_change().fillna(0.0)

    start = e["date"].iloc[0]
    end = e["date"].iloc[-1]
    years = max((end - start).days / 365.25, 1 / 365.25)

    final_equity = float(e["equity"].iloc[-1])
    total_return = final_equity / initial_capital - 1

    cagr = (
        (final_equity / initial_capital) ** (1 / years) - 1
        if final_equity > 0 and initial_capital > 0
        else np.nan
    )

    peak = e["equity"].cummax()
    drawdown = e["equity"] / peak - 1
    mdd = float(drawdown.min())

    daily = e["daily_return"]
    sharpe = np.nan
    if daily.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * daily.mean() / daily.std(ddof=1))

    downside = daily[daily < 0]
    sortino = np.nan
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(np.sqrt(252) * daily.mean() / downside.std(ddof=1))

    calmar = cagr / abs(mdd) if pd.notna(cagr) and mdd < 0 else np.nan

    if trades.empty:
        trade_win_rate = np.nan
        avg_trade = np.nan
        median_trade = np.nan
        profit_factor = np.nan
        avg_hold = np.nan
    else:
        r = trades["net_return"]
        trade_win_rate = float((r > 0).mean())
        avg_trade = float(r.mean())
        median_trade = float(r.median())
        gp = r[r > 0].sum()
        gl = -r[r < 0].sum()
        profit_factor = float(gp / gl) if gl > 0 else np.nan
        avg_hold = float(trades["holding_days"].mean())

    return {
        "start_date": start,
        "end_date": end,
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr) if pd.notna(cagr) else np.nan,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(calmar) if pd.notna(calmar) else np.nan,
        "trades": int(len(trades)),
        "trade_win_rate": trade_win_rate,
        "avg_trade_net_return": avg_trade,
        "median_trade_net_return": median_trade,
        "profit_factor": profit_factor,
        "avg_holding_days": avg_hold,
        "avg_exposure": float(e["exposure"].mean()),
        "max_exposure": float(e["exposure"].max()),
        "avg_positions": float(e["positions"].mean()),
        "max_positions_observed": int(e["positions"].max()),
        "pct_days_in_market": float((e["positions"] > 0).mean()),
    }


def period_returns(equity: pd.DataFrame, freq: str) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()

    e = equity.set_index("date")["equity"].sort_index()

    if freq == "year":
        endpoints = e.resample("YE").last()
        labels = endpoints.index.year.astype(str)
    elif freq == "month":
        endpoints = e.resample("ME").last()
        labels = endpoints.index.strftime("%Y-%m")
    else:
        raise ValueError(freq)

    previous = endpoints.shift(1)
    if len(previous):
        previous.iloc[0] = e.iloc[0]

    rets = endpoints / previous - 1

    return pd.DataFrame({
        "period": labels,
        "ending_equity": endpoints.values,
        "return": rets.values,
    })


def simulate_portfolio(
    panel: pd.DataFrame,
    *,
    initial_capital: float,
    stop_loss: float,
    take_profit: float,
    max_hold: int,
    max_positions: int,
    commission_rate: float,
    slippage_rate: float,
    sell_tax_rate: float,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
):
    df = panel.copy()

    if start_date is not None:
        df = df[df["date"] >= start_date].copy()
    if end_date is not None:
        df = df[df["date"] <= end_date].copy()

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    bars_by_date = {
        d: g.set_index("ticker", drop=False)
        for d, g in df.groupby("date", sort=True)
    }
    dates = sorted(bars_by_date)

    cash = float(initial_capital)
    positions: dict[str, Position] = {}
    trades = []
    equity_rows = []

    last_date = dates[-1]

    for date in dates:
        day = bars_by_date[date]

        # Update last known closes for held names when a bar exists today.
        for ticker, pos in list(positions.items()):
            if ticker in day.index:
                c = safe_float(day.loc[ticker, "close"])
                if np.isfinite(c):
                    pos.last_close = c

        # ---------- OPEN: value portfolio before new entries ----------
        market_value_open = 0.0
        for ticker, pos in positions.items():
            if ticker in day.index:
                px = safe_float(day.loc[ticker, "open"])
                if not np.isfinite(px):
                    px = pos.last_close
            else:
                px = pos.last_close
            market_value_open += pos.shares * px

        equity_open = cash + market_value_open
        target_notional = equity_open / max_positions if max_positions > 0 else 0.0

        # Entry candidates whose previous ticker bar had the signal.
        candidates = day[day["entry_candidate"] == 1].copy()
        if not candidates.empty:
            candidates = candidates.sort_values(
                ["signal_turnover_for_entry", "signal_float_mcap_for_entry"],
                ascending=[False, True],
            )

        slots = max_positions - len(positions)

        for _, bar in candidates.iterrows():
            if slots <= 0:
                break

            ticker = str(bar["ticker"]).zfill(6)
            if ticker in positions:
                continue

            raw_open = safe_float(bar["open"])
            if not np.isfinite(raw_open) or raw_open <= 0:
                continue

            entry_fill = raw_open * (1 + slippage_rate)

            # Integer shares. Make sure entry commission also fits in cash.
            max_affordable = int(
                np.floor(cash / (entry_fill * (1 + commission_rate)))
            )
            target_shares = int(np.floor(target_notional / entry_fill))
            shares = min(max_affordable, target_shares)

            if shares <= 0:
                continue

            gross_entry_value = shares * entry_fill
            entry_commission = gross_entry_value * commission_rate
            total_entry_cost = gross_entry_value + entry_commission

            cash -= total_entry_cost

            pos = Position(
                ticker=ticker,
                shares=shares,
                signal_date=pd.Timestamp(bar["signal_date_for_entry"]),
                entry_date=pd.Timestamp(date),
                entry_raw_open=raw_open,
                entry_fill=entry_fill,
                entry_commission=entry_commission,
                entry_cost_total=total_entry_cost,
                tp_price=entry_fill * (1 + take_profit),
                sl_price=entry_fill * (1 - stop_loss),
                days_held=0,
                last_close=safe_float(bar["close"]),
                signal_turnover=safe_float(bar["signal_turnover_for_entry"]),
                signal_float_mcap=safe_float(bar["signal_float_mcap_for_entry"]),
            )

            positions[ticker] = pos
            slots -= 1

        # ---------- INTRADAY / CLOSE: manage every open position ----------
        tickers_to_close = []

        for ticker, pos in list(positions.items()):
            if ticker not in day.index:
                # No bar (e.g. suspension): cannot trigger OHLC exits or advance holding bar count.
                continue

            bar = day.loc[ticker]
            pos.days_held += 1

            reason, trigger_price = exit_trigger_for_bar(pos, bar)

            if reason is None and pos.days_held >= max_hold:
                reason = "time_exit"
                trigger_price = safe_float(bar["close"])

            # Force liquidation at the sample end so performance is fully realized.
            if reason is None and date == last_date:
                reason = "forced_end"
                trigger_price = safe_float(bar["close"])

            if reason is None:
                continue

            if not np.isfinite(trigger_price) or trigger_price <= 0:
                continue

            exit_fill = trigger_price * (1 - slippage_rate)
            gross_exit_value = pos.shares * exit_fill
            exit_commission = gross_exit_value * commission_rate
            sell_tax = gross_exit_value * sell_tax_rate
            net_exit_proceeds = gross_exit_value - exit_commission - sell_tax

            cash += net_exit_proceeds

            gross_return = exit_fill / pos.entry_fill - 1
            net_return = net_exit_proceeds / pos.entry_cost_total - 1

            trades.append({
                "ticker": ticker,
                "signal_date": pos.signal_date,
                "entry_date": pos.entry_date,
                "exit_date": pd.Timestamp(date),
                "shares": pos.shares,
                "entry_raw_open": pos.entry_raw_open,
                "entry_fill": pos.entry_fill,
                "exit_fill": exit_fill,
                "entry_commission": pos.entry_commission,
                "exit_commission": exit_commission,
                "sell_tax": sell_tax,
                "entry_cost_total": pos.entry_cost_total,
                "net_exit_proceeds": net_exit_proceeds,
                "gross_return_after_slippage": gross_return,
                "net_return": net_return,
                "exit_reason": reason,
                "holding_days": pos.days_held,
                "signal_turnover": pos.signal_turnover,
                "signal_float_mcap": pos.signal_float_mcap,
            })

            tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            positions.pop(ticker, None)

        # ---------- DAILY MARK TO MARKET AT CLOSE ----------
        market_value_close = 0.0
        for ticker, pos in positions.items():
            if ticker in day.index:
                close_px = safe_float(day.loc[ticker, "close"])
                if np.isfinite(close_px):
                    pos.last_close = close_px

            market_value_close += pos.shares * pos.last_close

        equity = cash + market_value_close
        exposure = market_value_close / equity if equity > 0 else np.nan

        equity_rows.append({
            "date": pd.Timestamp(date),
            "cash": cash,
            "market_value": market_value_close,
            "equity": equity,
            "positions": len(positions),
            "exposure": exposure,
        })

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_df["daily_return"] = equity_df["equity"].pct_change().fillna(0.0)
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1

    trades_df = pd.DataFrame(trades)
    stats = calculate_stats(equity_df, trades_df, initial_capital)

    return equity_df, trades_df, stats


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

    panel = add_signal_and_entry_candidates(
        df,
        bins=args.bins,
        min_cross_section=args.min_cross_section,
    )

    commission_rate = args.commission_bps_per_side / 10000.0
    slippage_rate = args.slippage_bps_per_side / 10000.0
    sell_tax_rate = args.sell_tax_bps / 10000.0

    # Full continuous portfolio.
    equity, trades, full_stats = simulate_portfolio(
        panel,
        initial_capital=args.initial_capital,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_hold=args.max_hold,
        max_positions=args.max_positions,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        sell_tax_rate=sell_tax_rate,
    )

    equity.to_csv(out_dir / "daily_equity.csv", index=False)
    trades.to_csv(out_dir / "trades.csv", index=False)

    yearly = period_returns(equity, "year")
    monthly = period_returns(equity, "month")
    yearly.to_csv(out_dir / "yearly_returns.csv", index=False)
    monthly.to_csv(out_dir / "monthly_returns.csv", index=False)

    # Clean year-isolated diagnostics. Signal construction still uses the full panel,
    # so early-Jan entry candidates may use the immediately preceding 2025 signal.
    isolated_rows = []

    for year in [2025, 2026]:
        y_start = pd.Timestamp(f"{year}-01-01")
        y_end = pd.Timestamp(f"{year}-12-31")

        y_eq, y_tr, y_stats = simulate_portfolio(
            panel,
            initial_capital=args.initial_capital,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            max_hold=args.max_hold,
            max_positions=args.max_positions,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            sell_tax_rate=sell_tax_rate,
            start_date=y_start,
            end_date=y_end,
        )

        y_eq.to_csv(out_dir / f"daily_equity_{year}.csv", index=False)
        y_tr.to_csv(out_dir / f"trades_{year}.csv", index=False)

        isolated_rows.append({
            "sample": str(year),
            **y_stats,
        })

    summary = pd.DataFrame([
        {"sample": "full", **full_stats},
        *isolated_rows,
    ])

    summary.to_csv(out_dir / "mtm_summary.csv", index=False)

    panel_cols = [
        "ticker", "date", "open", "high", "low", "close",
        "free_float_market_cap", "float_turnover_1d",
        "supply_bucket", "demand_bucket", "signal",
        "entry_candidate", "signal_date_for_entry",
    ]
    panel[panel_cols].to_parquet(
        out_dir / "signal_panel.parquet",
        index=False,
    )

    print("\n=== Daily MTM Portfolio Summary ===")
    show_cols = [
        "sample",
        "start_date",
        "end_date",
        "trades",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "sortino",
        "calmar",
        "trade_win_rate",
        "avg_trade_net_return",
        "median_trade_net_return",
        "profit_factor",
        "avg_exposure",
        "max_exposure",
        "avg_positions",
        "pct_days_in_market",
    ]
    existing = [c for c in show_cols if c in summary.columns]
    print(summary[existing].to_string(index=False))

    print("\n=== Calendar Year Returns (continuous portfolio) ===")
    print(yearly.to_string(index=False))

    print("\n=== Execution assumptions ===")
    print("- Signal known after t close; enter at ticker's next available open.")
    print("- Integer shares.")
    print(f"- Initial capital: {args.initial_capital:,.0f}")
    print(f"- Max positions: {args.max_positions}")
    print(f"- SL: -{args.stop_loss:.1%}, TP: +{args.take_profit:.1%}, max hold: {args.max_hold} bars")
    print(f"- Commission per side: {args.commission_bps_per_side:.1f} bps")
    print(f"- Slippage per side: {args.slippage_bps_per_side:.1f} bps")
    print(f"- Sell tax: {args.sell_tax_bps:.1f} bps")
    print("- Same-day TP+SL touch: stop assumed first.")
    print("- Gap through stop: exit at gap open, not at stop threshold.")
    print("- Daily equity = cash + all open positions marked at close.")
    print("- Current pilot still has survivor/universe-selection bias; this is not final production evidence.")

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
