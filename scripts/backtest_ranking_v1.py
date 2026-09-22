#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ranking v1 for the Small-Float x High-Turnover setup. "
            "Select one pre-defined filter rule using 2025 only, then freeze it for 2026."
        )
    )
    p.add_argument(
        "--input",
        default="data/pit_kosdaq/core/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/analysis/ranking_v1",
    )
    p.add_argument("--initial-capital", type=float, default=100_000_000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--commission-bps", type=float, default=7.5)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--sell-tax-bps", type=float, default=0.0)
    p.add_argument(
        "--min-train-trades",
        type=int,
        default=40,
        help="A rule needs at least this many accepted 2025 trades to be selectable.",
    )
    return p.parse_args()


def require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def qbucket_cross_section(s: pd.Series, q: int = 5) -> pd.Series:
    pct = s.rank(method="average", pct=True)
    return np.ceil(pct * q).clip(1, q).astype("Int64")


def add_base_signal(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["float_mcap_pct"] = (
        x.groupby("date")["free_float_market_cap"]
        .rank(method="average", pct=True, ascending=True)
    )
    x["turnover_pct"] = (
        x.groupby("date")["float_turnover_1d"]
        .rank(method="average", pct=True, ascending=True)
    )
    x["signal"] = (
        (x["float_mcap_pct"] <= 0.20)
        & (x["turnover_pct"] >= 0.80)
    )
    return x


@dataclass(frozen=True)
class Rule:
    name: str
    turnover_q_min: int = 1
    turnover_q_max: int = 5
    float_q_min: int = 1
    float_q_max: int = 5
    exclude_gap_q5: bool = False


RULES = [
    Rule("R0_baseline_all"),
    Rule("R1_turnover_q1_3", turnover_q_max=3),
    Rule("R2_turnover_q1_2", turnover_q_max=2),
    Rule("R3_float_q1_2", float_q_max=2),
    Rule("R4_turn_q1_3_float_q1_2", turnover_q_max=3, float_q_max=2),
    Rule("R5_turn_q1_2_float_q1_2", turnover_q_max=2, float_q_max=2),
    Rule(
        "R6_turn_q1_3_float_q1_2_no_gap_q5",
        turnover_q_max=3,
        float_q_max=2,
        exclude_gap_q5=True,
    ),
    Rule(
        "R7_turn_q1_2_float_q1_2_no_gap_q5",
        turnover_q_max=2,
        float_q_max=2,
        exclude_gap_q5=True,
    ),
]


def build_executable_events(
    df: pd.DataFrame,
    tp: float,
    sl: float,
    max_hold: int,
    buy_cost: float,
    sell_cost: float,
) -> pd.DataFrame:
    rows = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        sig_pos = np.flatnonzero(g["signal"].fillna(False).to_numpy())

        for pos in sig_pos:
            entry_pos = int(pos) + 1

            # Require a COMPLETE max-hold horizon after entry.
            # This removes the right-censored tail that contaminated the first diagnostic.
            if entry_pos + max_hold - 1 >= n:
                continue

            s = g.iloc[int(pos)]
            e = g.iloc[entry_pos]

            raw_open = float(e["open"])
            signal_close = float(s["close"])
            if not np.isfinite(raw_open) or raw_open <= 0:
                continue

            entry_px = raw_open * (1.0 + buy_cost)
            tp_px = entry_px * (1.0 + tp)
            sl_px = entry_px * (1.0 + sl)

            w = g.iloc[entry_pos : entry_pos + max_hold]

            exit_reason = "TIME"
            exit_raw = float(w.iloc[-1]["close"])
            exit_date = pd.Timestamp(w.iloc[-1]["date"])
            holding_bars = max_hold

            for i, r in enumerate(w.itertuples(index=False), start=1):
                o = float(r.open)
                h = float(r.high)
                l = float(r.low)

                if np.isfinite(o) and o <= sl_px:
                    exit_reason = "SL_GAP"
                    exit_raw = o
                    exit_date = pd.Timestamp(r.date)
                    holding_bars = i
                    break

                hit_sl = np.isfinite(l) and l <= sl_px
                hit_tp = np.isfinite(h) and h >= tp_px

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

            exit_px = exit_raw * (1.0 - sell_cost)
            net_ret = exit_px / entry_px - 1.0

            gap = (
                raw_open / signal_close - 1.0
                if np.isfinite(signal_close) and signal_close > 0
                else np.nan
            )

            rows.append(
                {
                    "ticker": ticker,
                    "signal_date": pd.Timestamp(s["date"]),
                    "entry_date": pd.Timestamp(e["date"]),
                    "exit_date": exit_date,
                    "signal_close": signal_close,
                    "entry_open_raw": raw_open,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "exit_reason": exit_reason,
                    "holding_bars": holding_bars,
                    "trade_net_return": net_ret,
                    "next_open_gap": gap,
                    "free_float_market_cap": float(s["free_float_market_cap"]),
                    "float_turnover_1d": float(s["float_turnover_1d"]),
                    "target_20d_30pct": s.get("target_20d_30pct", np.nan),
                    "entry_row_date": pd.Timestamp(e["date"]),
                }
            )

    events = pd.DataFrame(rows)
    if events.empty:
        raise SystemExit("No executable events were generated.")

    # Quintiles are defined INSIDE the base signal, by signal date.
    events["turnover_signal_quintile"] = (
        events.groupby("signal_date")["float_turnover_1d"]
        .transform(qbucket_cross_section)
    )
    events["float_signal_quintile"] = (
        events.groupby("signal_date")["free_float_market_cap"]
        .transform(qbucket_cross_section)
    )

    # Gap quintile is entry-date cross-sectional, among that day's base candidates.
    events["gap_quintile"] = (
        events.groupby("entry_date")["next_open_gap"]
        .transform(qbucket_cross_section)
    )

    # Deterministic rank that deliberately avoids highest-turnover-first.
    # Preference:
    # 1) lower turnover quintile inside setup,
    # 2) float Q2 then Q1 then farther buckets,
    # 3) smaller absolute overnight gap,
    # 4) smaller absolute free-float market cap as final tie-break.
    events["float_q_distance_from_2"] = (
        events["float_signal_quintile"].astype(float) - 2.0
    ).abs()

    return events


def apply_rule(events: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    x = events[
        events["turnover_signal_quintile"].between(
            rule.turnover_q_min, rule.turnover_q_max
        )
        & events["float_signal_quintile"].between(
            rule.float_q_min, rule.float_q_max
        )
    ].copy()

    if rule.exclude_gap_q5:
        x = x[x["gap_quintile"] < 5].copy()

    return x


def rank_daily_candidates(x: pd.DataFrame) -> pd.DataFrame:
    if x.empty:
        return x

    return x.sort_values(
        [
            "entry_date",
            "turnover_signal_quintile",
            "float_q_distance_from_2",
            "next_open_gap_abs",
            "free_float_market_cap",
            "ticker",
        ]
    )


def prepare_ranking_cols(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["next_open_gap_abs"] = x["next_open_gap"].abs()
    return x


def market_lookup(df: pd.DataFrame):
    # compact lookup for daily close marking
    cols = ["date", "ticker", "open", "close"]
    m = df[cols].copy()
    return {
        (pd.Timestamp(r.date), str(r.ticker)): (float(r.open), float(r.close))
        for r in m.itertuples(index=False)
    }


def backtest_rule(
    market_df: pd.DataFrame,
    events: pd.DataFrame,
    rule: Rule,
    start_date: str,
    end_date: str,
    initial_capital: float,
    max_positions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    ev = events[
        (events["signal_date"] >= start)
        & (events["signal_date"] <= end)
    ].copy()
    ev = apply_rule(ev, rule)
    ev = prepare_ranking_cols(ev)

    # Entry must also fall in requested period.
    ev = ev[
        (ev["entry_date"] >= start)
        & (ev["entry_date"] <= end)
    ].copy()

    entry_map = {
        d: g.copy()
        for d, g in ev.groupby("entry_date", sort=True)
    }

    dates = (
        market_df.loc[
            (market_df["date"] >= start)
            & (market_df["date"] <= end),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    lookup = market_lookup(
        market_df[
            (market_df["date"] >= start)
            & (market_df["date"] <= end)
        ]
    )

    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    trades = []
    equity_rows = []

    prev_equity = float(initial_capital)

    for date in dates:
        date = pd.Timestamp(date)

        # Opening equity approximation: cash + existing positions at today's open
        opening_equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            if px is not None:
                opening_equity += pos["shares"] * px[0]
            else:
                opening_equity += pos["shares"] * pos["last_close"]

        # Conservative slot logic:
        # positions scheduled to exit today still occupy a slot at the open.
        free_slots = max_positions - len(positions)

        candidates = entry_map.get(date)
        if free_slots > 0 and candidates is not None and not candidates.empty:
            candidates = rank_daily_candidates(candidates)

            for r in candidates.itertuples(index=False):
                if free_slots <= 0:
                    break
                ticker = str(r.ticker)

                # no overlapping same ticker
                if ticker in positions:
                    continue

                target_capital = opening_equity / max_positions
                shares = int(target_capital // float(r.entry_price))
                if shares <= 0:
                    continue

                cost = shares * float(r.entry_price)
                if cost > cash:
                    shares = int(cash // float(r.entry_price))
                    cost = shares * float(r.entry_price)
                if shares <= 0:
                    continue

                cash -= cost
                positions[ticker] = {
                    "ticker": ticker,
                    "shares": shares,
                    "entry_date": pd.Timestamp(r.entry_date),
                    "exit_date": pd.Timestamp(r.exit_date),
                    "entry_price": float(r.entry_price),
                    "exit_price": float(r.exit_price),
                    "exit_reason": str(r.exit_reason),
                    "signal_date": pd.Timestamp(r.signal_date),
                    "trade_net_return": float(r.trade_net_return),
                    "turnover_q": int(r.turnover_signal_quintile),
                    "float_q": int(r.float_signal_quintile),
                    "gap_q": int(r.gap_quintile),
                    "last_close": float(r.entry_open_raw),
                }
                free_slots -= 1

        # Process scheduled exits after entries (conservative for slot reuse).
        exiting = [
            ticker for ticker, pos in positions.items()
            if pos["exit_date"] == date
        ]

        for ticker in exiting:
            pos = positions.pop(ticker)
            proceeds = pos["shares"] * pos["exit_price"]
            cash += proceeds

            trades.append(
                {
                    **pos,
                    "exit_proceeds": proceeds,
                    "position_net_pnl": (
                        pos["shares"]
                        * (pos["exit_price"] - pos["entry_price"])
                    ),
                }
            )

        # Mark remaining positions to close.
        equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            if px is not None:
                close_px = px[1]
                pos["last_close"] = close_px
            else:
                close_px = pos["last_close"]
            equity += pos["shares"] * close_px

        daily_ret = equity / prev_equity - 1.0 if prev_equity > 0 else np.nan
        prev_equity = equity

        equity_rows.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": daily_ret,
                "cash": cash,
                "positions": len(positions),
                "exposure": (
                    (equity - cash) / equity
                    if equity > 0 else np.nan
                ),
            }
        )

    # Liquidate any positions left after period end at last known close, for summary only.
    if equity_rows:
        final_date = pd.Timestamp(equity_rows[-1]["date"])
        for ticker, pos in list(positions.items()):
            px = lookup.get((final_date, ticker))
            final_px = px[1] if px is not None else pos["last_close"]
            cash += pos["shares"] * final_px
            positions.pop(ticker, None)

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trades)

    if eq.empty:
        return eq, tr, {
            "rule": rule.name,
            "trades": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
        }

    curve = eq["equity"].astype(float)
    roll_max = curve.cummax()
    dd = curve / roll_max - 1.0
    mdd = float(dd.min())

    total_return = float(curve.iloc[-1] / initial_capital - 1.0)

    days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = (
        float((curve.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
        if curve.iloc[-1] > 0 else -1.0
    )

    dr = eq["daily_return"].dropna()
    sharpe = (
        float(np.sqrt(252) * dr.mean() / dr.std(ddof=1))
        if len(dr) > 1 and dr.std(ddof=1) > 0
        else np.nan
    )

    calmar = (
        float(cagr / abs(mdd))
        if np.isfinite(cagr) and mdd < 0
        else np.nan
    )

    if tr.empty:
        pf = np.nan
        win = np.nan
        avg_trade = np.nan
        median_trade = np.nan
    else:
        pnl = tr["position_net_pnl"].astype(float)
        gain = pnl[pnl > 0].sum()
        loss = -pnl[pnl < 0].sum()
        pf = float(gain / loss) if loss > 0 else np.nan
        win = float((pnl > 0).mean())
        avg_trade = float(tr["trade_net_return"].mean())
        median_trade = float(tr["trade_net_return"].median())

    summary = {
        "rule": rule.name,
        "trades": int(len(tr)),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "profit_factor": pf,
        "win_rate": win,
        "avg_trade_net_return": avg_trade,
        "median_trade_net_return": median_trade,
        "avg_exposure": float(eq["exposure"].mean()),
        "avg_positions": float(eq["positions"].mean()),
    }

    return eq, tr, summary


def event_quality(events: pd.DataFrame, rule: Rule, year: int) -> dict:
    x = events[events["signal_date"].dt.year == year].copy()
    x = apply_rule(x, rule)

    if x.empty:
        return {
            "year": year,
            "rule": rule.name,
            "events": 0,
        }

    pos = x.loc[x["trade_net_return"] > 0, "trade_net_return"].sum()
    neg = -x.loc[x["trade_net_return"] < 0, "trade_net_return"].sum()

    return {
        "year": year,
        "rule": rule.name,
        "events": len(x),
        "tp_rate": float((x["exit_reason"] == "TP").mean()),
        "sl_rate": float(
            x["exit_reason"].isin(
                ["SL", "SL_GAP", "SL_SAME_BAR"]
            ).mean()
        ),
        "avg_trade_return": float(x["trade_net_return"].mean()),
        "median_trade_return": float(x["trade_net_return"].median()),
        "event_profit_factor": float(pos / neg) if neg > 0 else np.nan,
        "original_target30_rate": float(x["target_20d_30pct"].mean()),
    }


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
    df = add_base_signal(df)

    buy_cost = (
        args.commission_bps + args.slippage_bps
    ) / 10000.0
    sell_cost = (
        args.commission_bps
        + args.slippage_bps
        + args.sell_tax_bps
    ) / 10000.0

    events = build_executable_events(
        df=df,
        tp=args.tp,
        sl=args.sl,
        max_hold=args.max_hold,
        buy_cost=buy_cost,
        sell_cost=sell_cost,
    )
    events.to_parquet(out_dir / "events_complete_horizon.parquet", index=False)

    # ---------------------------
    # TRAIN: 2025 only
    # ---------------------------
    train_rows = []
    train_equities = {}
    train_trades = {}

    for rule in RULES:
        eq, tr, summary = backtest_rule(
            market_df=df,
            events=events,
            rule=rule,
            start_date="2025-01-01",
            end_date="2025-12-31",
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
        )
        q = event_quality(events, rule, 2025)
        summary.update(
            {
                "train_event_count": q.get("events", 0),
                "train_event_avg_return": q.get("avg_trade_return", np.nan),
                "train_event_pf": q.get("event_profit_factor", np.nan),
                "train_tp_rate": q.get("tp_rate", np.nan),
            }
        )
        train_rows.append(summary)
        train_equities[rule.name] = eq
        train_trades[rule.name] = tr

    train_cmp = pd.DataFrame(train_rows)

    # Selection uses TRAIN ONLY.
    eligible = train_cmp[
        (train_cmp["trades"] >= args.min_train_trades)
        & train_cmp["calmar"].notna()
    ].copy()

    if eligible.empty:
        raise SystemExit(
            "No rule met the minimum train-trade threshold."
        )

    # Primary: Calmar. Tie-breaks: PF, CAGR, then more trades.
    eligible = eligible.sort_values(
        ["calmar", "profit_factor", "cagr", "trades"],
        ascending=[False, False, False, False],
    )
    selected_name = str(eligible.iloc[0]["rule"])
    selected_rule = next(r for r in RULES if r.name == selected_name)

    # ---------------------------
    # TEST: 2026, frozen rule
    # ---------------------------
    test_eq, test_tr, test_summary = backtest_rule(
        market_df=df,
        events=events,
        rule=selected_rule,
        start_date="2026-01-01",
        end_date="2026-08-29",
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
    )
    test_quality = event_quality(events, selected_rule, 2026)
    test_summary.update(
        {
            "test_event_count": test_quality.get("events", 0),
            "test_event_avg_return": test_quality.get(
                "avg_trade_return", np.nan
            ),
            "test_event_pf": test_quality.get(
                "event_profit_factor", np.nan
            ),
            "test_tp_rate": test_quality.get("tp_rate", np.nan),
        }
    )

    # Full continuous diagnostic with selected rule only.
    full_eq, full_tr, full_summary = backtest_rule(
        market_df=df,
        events=events,
        rule=selected_rule,
        start_date="2025-01-01",
        end_date="2026-08-29",
        initial_capital=args.initial_capital,
        max_positions=args.max_positions,
    )

    # Save outputs.
    train_cmp.to_csv(
        out_dir / "rule_comparison_train_2025.csv",
        index=False,
    )
    train_equities[selected_name].to_csv(
        out_dir / "selected_train_equity_2025.csv",
        index=False,
    )
    train_trades[selected_name].to_csv(
        out_dir / "selected_train_trades_2025.csv",
        index=False,
    )
    test_eq.to_csv(
        out_dir / "selected_test_equity_2026.csv",
        index=False,
    )
    test_tr.to_csv(
        out_dir / "selected_test_trades_2026.csv",
        index=False,
    )
    full_eq.to_csv(
        out_dir / "selected_full_equity.csv",
        index=False,
    )
    full_tr.to_csv(
        out_dir / "selected_full_trades.csv",
        index=False,
    )

    selection_payload = {
        "selected_rule": selected_rule.__dict__,
        "selection_data": "2025 only",
        "selection_metric": "Calmar; ties by PF, CAGR, trades",
        "min_train_trades": args.min_train_trades,
        "test_data": "2026-01-01 through 2026-08-29, not used for selection",
    }
    (out_dir / "selected_rule.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame([test_summary]).to_csv(
        out_dir / "selected_test_summary_2026.csv",
        index=False,
    )
    pd.DataFrame([full_summary]).to_csv(
        out_dir / "selected_full_summary.csv",
        index=False,
    )

    print("\n=== Ranking v1: 2025 TRAIN Rule Comparison ===")
    print(
        train_cmp.sort_values(
            ["calmar", "profit_factor"],
            ascending=False,
        ).to_string(index=False)
    )

    print("\n=== SELECTED RULE (2025 only) ===")
    print(json.dumps(selection_payload, ensure_ascii=False, indent=2))

    print("\n=== 2026 FROZEN-RULE TEST ===")
    print(pd.DataFrame([test_summary]).to_string(index=False))

    print("\n=== FULL CONTINUOUS DIAGNOSTIC ===")
    print(pd.DataFrame([full_summary]).to_string(index=False))

    print("\nImportant:")
    print("- 2026 is NEVER used to choose the rule.")
    print("- Signals without a complete 20-bar future horizon are removed.")
    print("- Ranking no longer buys highest turnover first.")
    print("- TP/SL/hold remain fixed at +30% / -15% / 20 bars.")
    print("- This is Ranking v1, not final production optimization.")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
