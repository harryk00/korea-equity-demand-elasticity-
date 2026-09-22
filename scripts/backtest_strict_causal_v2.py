#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Strict causal ranking: t-close info only -> t+1 open execution")
    p.add_argument("--input", default="data/pit_kosdaq/core/processed/stock_master_model.parquet")
    p.add_argument("--out-dir", default="data/pit_kosdaq/analysis/strict_causal_v2")
    p.add_argument("--initial-capital", type=float, default=100_000_000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--commission-bps", type=float, default=7.5)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--sell-tax-bps", type=float, default=0.0)
    p.add_argument("--min-train-trades", type=int, default=40)
    return p.parse_args()


def require(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def qbucket(pct, q=5):
    return np.ceil(pct * q).clip(1, q).astype("Int64")


def add_causal_features(df):
    x = df.sort_values(["ticker", "date"]).copy()
    g = x.groupby("ticker", group_keys=False)

    x["prev_close"] = g["close"].shift(1)
    x["close_lag5"] = g["close"].shift(5)
    x["turnover_lag1"] = g["float_turnover_1d"].shift(1)
    x["ret_1d"] = x["close"] / x["prev_close"] - 1.0
    x["ret_5d"] = x["close"] / x["close_lag5"] - 1.0
    x["turnover_accel_1d"] = x["float_turnover_1d"] / x["turnover_lag1"] - 1.0

    rolling_high20 = (
        g["high"].rolling(20, min_periods=5).max().reset_index(level=0, drop=True)
    )
    x["drawdown_from_20d_high"] = x["close"] / rolling_high20 - 1.0

    rng = x["high"] - x["low"]
    x["close_location"] = np.where(rng > 0, (x["close"] - x["low"]) / rng, 0.5)
    x["intraday_return"] = x["close"] / x["open"] - 1.0

    x["float_mcap_pct"] = x.groupby("date")["free_float_market_cap"].rank(method="average", pct=True)
    x["turnover_pct"] = x.groupby("date")["float_turnover_1d"].rank(method="average", pct=True)
    x["signal"] = (x["float_mcap_pct"] <= 0.20) & (x["turnover_pct"] >= 0.80)

    sig = x["signal"].fillna(False)
    rank_specs = [
        ("float_turnover_1d", "turnover_signal_pct"),
        ("free_float_market_cap", "float_signal_pct"),
        ("ret_1d", "ret1_signal_pct"),
        ("ret_5d", "ret5_signal_pct"),
        ("turnover_accel_1d", "turn_accel_signal_pct"),
    ]
    for col, out in rank_specs:
        x[out] = np.nan
        x.loc[sig, out] = x.loc[sig].groupby("date")[col].rank(method="average", pct=True)

    x["turnover_signal_q"] = qbucket(x["turnover_signal_pct"])
    x["float_signal_q"] = qbucket(x["float_signal_pct"])
    x["ret1_signal_q"] = qbucket(x["ret1_signal_pct"])
    x["ret5_signal_q"] = qbucket(x["ret5_signal_pct"])
    x["turn_accel_signal_q"] = qbucket(x["turn_accel_signal_pct"])
    x["float_q_distance_from_2"] = (x["float_signal_q"].astype(float) - 2.0).abs()
    return x


@dataclass(frozen=True)
class Rule:
    name: str
    turnover_q_max: int
    float_q_max: int
    avoid_ret1_q5: bool = False
    avoid_ret5_q5: bool = False


RULES = [
    Rule("C0_base_all", 5, 5),
    Rule("C1_turn_q1_3", 3, 5),
    Rule("C2_turn_q1_2", 2, 5),
    Rule("C3_turn_q1_3_float_q1_2", 3, 2),
    Rule("C4_turn_q1_2_float_q1_2", 2, 2),
    Rule("C5_C4_avoid_ret1_q5", 2, 2, avoid_ret1_q5=True),
    Rule("C6_C4_avoid_ret5_q5", 2, 2, avoid_ret5_q5=True),
    Rule("C7_C4_avoid_ret1_ret5_q5", 2, 2, avoid_ret1_q5=True, avoid_ret5_q5=True),
]


def apply_rule(events, rule):
    x = events[
        (events["turnover_signal_q"] <= rule.turnover_q_max)
        & (events["float_signal_q"] <= rule.float_q_max)
    ].copy()
    if rule.avoid_ret1_q5:
        x = x[x["ret1_signal_q"] < 5].copy()
    if rule.avoid_ret5_q5:
        x = x[x["ret5_signal_q"] < 5].copy()
    return x


def causal_sort(x):
    return x.sort_values(
        [
            "entry_date",
            "turnover_signal_q",
            "float_q_distance_from_2",
            "ret1_signal_pct",
            "ret5_signal_pct",
            "free_float_market_cap",
            "ticker",
        ],
        ascending=[True, True, True, True, True, True, True],
        na_position="last",
    )


def build_events(df, tp, sl, max_hold, buy_cost, sell_cost):
    rows = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        for pos in np.flatnonzero(g["signal"].fillna(False).to_numpy()):
            pos = int(pos)
            entry_pos = pos + 1
            if entry_pos >= n:
                continue
            # Evaluation hygiene only: remove right-censored events.
            if entry_pos + max_hold - 1 >= n:
                continue

            s = g.iloc[pos]
            e = g.iloc[entry_pos]
            raw_open = float(e["open"])
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
                o, h, l = float(r.open), float(r.high), float(r.low)
                if np.isfinite(o) and o <= sl_px:
                    exit_reason, exit_raw, exit_date, holding_bars = "SL_GAP", o, pd.Timestamp(r.date), i
                    break
                hit_sl = np.isfinite(l) and l <= sl_px
                hit_tp = np.isfinite(h) and h >= tp_px
                if hit_sl and hit_tp:
                    exit_reason, exit_raw, exit_date, holding_bars = "SL_SAME_BAR", sl_px, pd.Timestamp(r.date), i
                    break
                if hit_sl:
                    exit_reason, exit_raw, exit_date, holding_bars = "SL", sl_px, pd.Timestamp(r.date), i
                    break
                if hit_tp:
                    exit_reason, exit_raw, exit_date, holding_bars = "TP", tp_px, pd.Timestamp(r.date), i
                    break

            exit_px = exit_raw * (1.0 - sell_cost)
            row = {
                "ticker": ticker,
                "signal_date": pd.Timestamp(s["date"]),
                "entry_date": pd.Timestamp(e["date"]),
                "exit_date": exit_date,
                "entry_open_raw": raw_open,
                "entry_price": entry_px,
                "exit_price": exit_px,
                "exit_reason": exit_reason,
                "holding_bars": holding_bars,
                "trade_net_return": exit_px / entry_px - 1.0,
            }
            for c in [
                "free_float_market_cap", "float_turnover_1d", "float_mcap_pct", "turnover_pct",
                "turnover_signal_pct", "float_signal_pct", "turnover_signal_q", "float_signal_q",
                "ret_1d", "ret_5d", "ret1_signal_pct", "ret5_signal_pct", "ret1_signal_q", "ret5_signal_q",
                "turnover_accel_1d", "turn_accel_signal_pct", "turn_accel_signal_q",
                "drawdown_from_20d_high", "close_location", "intraday_return", "float_q_distance_from_2",
                "target_20d_30pct",
            ]:
                if c in s.index:
                    row[c] = s[c]
            rows.append(row)

    ev = pd.DataFrame(rows)
    if ev.empty:
        raise SystemExit("No complete-horizon executable events generated.")
    return ev


def market_lookup(df, start, end):
    m = df.loc[(df["date"] >= start) & (df["date"] <= end), ["date", "ticker", "open", "close"]]
    return {(pd.Timestamp(r.date), str(r.ticker)): (float(r.open), float(r.close)) for r in m.itertuples(index=False)}


def summarize(eq, tr, initial_capital):
    if eq.empty:
        return {k: np.nan for k in ["total_return", "cagr", "max_drawdown", "sharpe", "calmar", "profit_factor", "win_rate", "avg_trade_net_return", "median_trade_net_return", "avg_exposure", "avg_positions"]} | {"trades": 0}

    curve = eq["equity"].astype(float)
    dd = curve / curve.cummax() - 1.0
    mdd = float(dd.min())
    total_return = float(curve.iloc[-1] / initial_capital - 1.0)
    days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = float((curve.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0) if curve.iloc[-1] > 0 else -1.0
    dr = eq["daily_return"].dropna()
    sharpe = float(np.sqrt(252) * dr.mean() / dr.std(ddof=1)) if len(dr) > 1 and dr.std(ddof=1) > 0 else np.nan
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan

    if tr.empty:
        pf = win = avg_trade = med_trade = np.nan
    else:
        pnl = tr["position_net_pnl"].astype(float)
        gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        pf = float(gains / losses) if losses > 0 else np.nan
        win = float((pnl > 0).mean())
        avg_trade = float(tr["trade_net_return"].mean())
        med_trade = float(tr["trade_net_return"].median())

    return {
        "trades": int(len(tr)), "total_return": total_return, "cagr": cagr,
        "max_drawdown": mdd, "sharpe": sharpe, "calmar": calmar,
        "profit_factor": pf, "win_rate": win,
        "avg_trade_net_return": avg_trade, "median_trade_net_return": med_trade,
        "avg_exposure": float(eq["exposure"].mean()), "avg_positions": float(eq["positions"].mean()),
    }


def backtest(df, events, rule, start_date, end_date, initial_capital, max_positions):
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    ev = events[
        (events["signal_date"] >= start) & (events["signal_date"] <= end)
        & (events["entry_date"] >= start) & (events["entry_date"] <= end)
    ].copy()
    ev = causal_sort(apply_rule(ev, rule))
    entry_map = {d: g.copy() for d, g in ev.groupby("entry_date", sort=True)}
    dates = df.loc[(df["date"] >= start) & (df["date"] <= end), "date"].drop_duplicates().sort_values().tolist()
    lookup = market_lookup(df, start, end)

    cash, prev_equity = float(initial_capital), float(initial_capital)
    positions, trades, eq_rows = {}, [], []

    for date in dates:
        date = pd.Timestamp(date)
        opening_equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            opening_equity += pos["shares"] * (px[0] if px is not None else pos["last_close"])

        free_slots = max_positions - len(positions)
        cand = entry_map.get(date)
        if free_slots > 0 and cand is not None and not cand.empty:
            for r in causal_sort(cand).itertuples(index=False):
                if free_slots <= 0:
                    break
                ticker = str(r.ticker)
                if ticker in positions:
                    continue
                target_cap = opening_equity / max_positions
                shares = int(target_cap // float(r.entry_price))
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
                    "ticker": ticker, "shares": shares,
                    "signal_date": pd.Timestamp(r.signal_date), "entry_date": pd.Timestamp(r.entry_date),
                    "exit_date": pd.Timestamp(r.exit_date), "entry_price": float(r.entry_price),
                    "exit_price": float(r.exit_price), "exit_reason": str(r.exit_reason),
                    "trade_net_return": float(r.trade_net_return),
                    "turnover_signal_q": int(r.turnover_signal_q), "float_signal_q": int(r.float_signal_q),
                    "ret_1d": float(r.ret_1d) if np.isfinite(r.ret_1d) else np.nan,
                    "ret_5d": float(r.ret_5d) if np.isfinite(r.ret_5d) else np.nan,
                    "last_close": float(r.entry_open_raw),
                }
                free_slots -= 1

        # Conservative: positions exiting today occupied a slot at today's open.
        for ticker in [t for t, p in positions.items() if p["exit_date"] == date]:
            pos = positions.pop(ticker)
            cash += pos["shares"] * pos["exit_price"]
            pos["position_net_pnl"] = pos["shares"] * (pos["exit_price"] - pos["entry_price"])
            trades.append(pos)

        equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            close_px = px[1] if px is not None else pos["last_close"]
            pos["last_close"] = close_px
            equity += pos["shares"] * close_px

        eq_rows.append({
            "date": date, "equity": equity,
            "daily_return": equity / prev_equity - 1.0 if prev_equity > 0 else np.nan,
            "cash": cash, "positions": len(positions),
            "exposure": (equity - cash) / equity if equity > 0 else np.nan,
        })
        prev_equity = equity

    eq, tr = pd.DataFrame(eq_rows), pd.DataFrame(trades)
    out = summarize(eq, tr, initial_capital)
    out["rule"] = rule.name
    return eq, tr, out


def event_quality(events, rule, year):
    x = apply_rule(events[events["signal_date"].dt.year == year].copy(), rule)
    if x.empty:
        return {"events": 0, "avg_event_return": np.nan, "event_pf": np.nan, "tp_rate": np.nan}
    pos = x.loc[x["trade_net_return"] > 0, "trade_net_return"].sum()
    neg = -x.loc[x["trade_net_return"] < 0, "trade_net_return"].sum()
    return {
        "events": len(x), "avg_event_return": float(x["trade_net_return"].mean()),
        "event_pf": float(pos / neg) if neg > 0 else np.nan,
        "tp_rate": float((x["exit_reason"] == "TP").mean()),
    }


def causality_audit():
    return pd.DataFrame([
        ["free_float_market_cap", "t close", True],
        ["float_turnover_1d", "t close", True],
        ["turnover_signal_q", "t close cross-section", True],
        ["float_signal_q", "t close cross-section", True],
        ["ret_1d", "t close vs t-1 close", True],
        ["ret_5d", "t close vs t-5 close", True],
        ["next_open_gap", "NOT USED", False],
        ["t+1 open", "execution price only", False],
        ["t+1 high/low/close", "outcome/exit simulation only", False],
    ], columns=["field", "information_time", "used_for_selection"])


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    require(df, ["ticker", "date", "open", "high", "low", "close", "free_float_market_cap", "float_turnover_1d"])
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = add_causal_features(df)

    buy_cost = (args.commission_bps + args.slippage_bps) / 10000.0
    sell_cost = (args.commission_bps + args.slippage_bps + args.sell_tax_bps) / 10000.0
    events = build_events(df, args.tp, args.sl, args.max_hold, buy_cost, sell_cost)
    events.to_parquet(out_dir / "causal_events_complete_horizon.parquet", index=False)

    train_rows, train_store = [], {}
    for rule in RULES:
        eq, tr, s = backtest(df, events, rule, "2025-01-01", "2025-12-31", args.initial_capital, args.max_positions)
        q = event_quality(events, rule, 2025)
        s.update({
            "train_event_count": q["events"], "train_event_avg_return": q["avg_event_return"],
            "train_event_pf": q["event_pf"], "train_tp_rate": q["tp_rate"],
        })
        train_rows.append(s)
        train_store[rule.name] = (eq, tr)

    train_cmp = pd.DataFrame(train_rows)
    eligible = train_cmp[(train_cmp["trades"] >= args.min_train_trades) & train_cmp["calmar"].notna()].copy()
    if eligible.empty:
        raise SystemExit("No rule met minimum train trades.")
    eligible = eligible.sort_values(["calmar", "profit_factor", "cagr", "trades"], ascending=[False, False, False, False])
    selected_name = str(eligible.iloc[0]["rule"])
    selected_rule = next(r for r in RULES if r.name == selected_name)

    test_eq, test_tr, test_s = backtest(df, events, selected_rule, "2026-01-01", "2026-08-29", args.initial_capital, args.max_positions)
    tq = event_quality(events, selected_rule, 2026)
    test_s.update({
        "test_event_count": tq["events"], "test_event_avg_return": tq["avg_event_return"],
        "test_event_pf": tq["event_pf"], "test_tp_rate": tq["tp_rate"],
    })

    full_eq, full_tr, full_s = backtest(df, events, selected_rule, "2025-01-01", "2026-08-29", args.initial_capital, args.max_positions)

    train_cmp.to_csv(out_dir / "train_rule_comparison_2025.csv", index=False)
    train_store[selected_name][0].to_csv(out_dir / "selected_train_equity_2025.csv", index=False)
    train_store[selected_name][1].to_csv(out_dir / "selected_train_trades_2025.csv", index=False)
    test_eq.to_csv(out_dir / "selected_test_equity_2026.csv", index=False)
    test_tr.to_csv(out_dir / "selected_test_trades_2026.csv", index=False)
    full_eq.to_csv(out_dir / "selected_full_equity.csv", index=False)
    full_tr.to_csv(out_dir / "selected_full_trades.csv", index=False)
    pd.DataFrame([test_s]).to_csv(out_dir / "selected_test_summary_2026.csv", index=False)
    pd.DataFrame([full_s]).to_csv(out_dir / "selected_full_summary.csv", index=False)

    audit = causality_audit()
    audit.to_csv(out_dir / "causality_audit.csv", index=False)

    selection = {
        "selected_rule": asdict(selected_rule),
        "selection_period": "2025 only",
        "test_period": "2026-01-01 through 2026-08-29",
        "selection_metric": "Calmar; tie-break PF, CAGR, trades",
        "strict_causal_rule": "all ranking/filter features known by signal-date close; t+1 open is execution only",
        "tp": args.tp, "sl": args.sl, "max_hold": args.max_hold, "max_positions": args.max_positions,
    }
    (out_dir / "selected_rule.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== STRICT CAUSAL RANKING v2: 2025 TRAIN ===")
    print(train_cmp.sort_values(["calmar", "profit_factor"], ascending=[False, False]).to_string(index=False))
    print("\n=== SELECTED RULE (2025 ONLY) ===")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    print("\n=== 2026 FROZEN TEST ===")
    print(pd.DataFrame([test_s]).to_string(index=False))
    print("\n=== FULL CONTINUOUS DIAGNOSTIC ===")
    print(pd.DataFrame([full_s]).to_string(index=False))
    print("\n=== CAUSALITY AUDIT ===")
    print(audit.to_string(index=False))
    print("\nImportant:")
    print("- NO next-open gap filter.")
    print("- NO t+1 data is used to rank/filter candidates.")
    print("- t+1 open is execution price only.")
    print("- Base setup fixed: float mcap bottom 20% + turnover top 20%.")
    print("- TP/SL/hold fixed: +30% / -15% / 20 bars.")
    print("- 2025 selects; 2026 only tests.")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
