#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Demand Acceleration v3. Tests whether causal demand-acceleration "
            "features add value on top of the strict-causal C5 baseline."
        )
    )
    p.add_argument(
        "--input",
        default="data/pit_kosdaq/core/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/analysis/demand_acceleration_v3",
    )
    p.add_argument("--initial-capital", type=float, default=100_000_000.0)
    p.add_argument("--max-positions", type=int, default=5)
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--commission-bps", type=float, default=7.5)
    p.add_argument(
        "--selection-slippage-bps",
        type=float,
        default=10.0,
        help="Slippage used for 2025 rule selection and primary 2026 test.",
    )
    p.add_argument(
        "--stress-slippage-bps",
        type=str,
        default="10,30,50,100",
        help="Comma-separated per-side slippage stress scenarios.",
    )
    p.add_argument("--min-train-trades", type=int, default=30)
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def require(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")


def first_existing(cols: list[str], aliases: list[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for a in aliases:
        if a.lower() in lower:
            return lower[a.lower()]
    return None


def finite_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    return out.replace([np.inf, -np.inf], np.nan)


def daily_pct_rank(
    df: pd.DataFrame,
    mask: pd.Series,
    col: str,
    ascending: bool = True,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if mask.any():
        ranked = (
            df.loc[mask]
            .groupby("date")[col]
            .rank(method="average", pct=True, ascending=ascending)
        )
        out.loc[ranked.index] = ranked
    return out


def q_from_pct(pct: pd.Series, q: int = 5) -> pd.Series:
    vals = np.ceil(pct * q).clip(1, q)
    return vals.astype("Int64")


def year_tax_rate(exit_date: pd.Timestamp) -> float:
    """
    KOSDAQ sell-side securities transaction tax assumption:
    - 2025: 15 bps
    - 2026 onward in this sample: 20 bps
    """
    y = pd.Timestamp(exit_date).year
    if y <= 2025:
        return 0.0015
    return 0.0020


# ---------------------------------------------------------------------
# Causal feature engineering
# ---------------------------------------------------------------------

def add_causal_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = df.sort_values(["ticker", "date"]).copy()
    g = x.groupby("ticker", group_keys=False)

    # Price / baseline features known at t close.
    x["prev_close"] = g["close"].shift(1)
    x["close_lag5"] = g["close"].shift(5)
    x["ret_1d"] = finite_ratio(x["close"], x["prev_close"]) - 1.0
    x["ret_5d"] = finite_ratio(x["close"], x["close_lag5"]) - 1.0

    # Demand acceleration core: turnover.
    x["turnover_lag1"] = g["float_turnover_1d"].shift(1)
    x["turnover_prev5_median"] = (
        g["float_turnover_1d"]
        .apply(lambda s: s.shift(1).rolling(5, min_periods=3).median())
        .reset_index(level=0, drop=True)
    )
    x["turnover_accel_1d"] = (
        finite_ratio(x["float_turnover_1d"], x["turnover_lag1"]) - 1.0
    )
    x["turnover_accel_5d"] = (
        finite_ratio(x["float_turnover_1d"], x["turnover_prev5_median"]) - 1.0
    )

    # Optional raw columns.
    cols = list(x.columns)

    volume_col = first_existing(
        cols,
        [
            "volume",
            "trading_volume",
            "trade_volume",
            "accumulated_volume",
            "acml_vol",
            "acc_trading_volume",
        ],
    )

    value_col = first_existing(
        cols,
        [
            "trading_value",
            "trade_value",
            "trading_amount",
            "trade_amount",
            "value",
            "amount",
            "acc_trading_value",
            "acml_tr_pbmn",
        ],
    )

    program_col = first_existing(
        cols,
        [
            "program_net_buy_value",
            "program_net_buy",
            "program_netbuy_value",
            "program_netbuy",
            "program_net_purchase",
            "program_net_purchase_value",
        ],
    )

    execution_col = first_existing(
        cols,
        [
            "execution_strength",
            "execution_intensity",
            "chegyeol_strength",
            "trade_strength",
        ],
    )

    detected = {
        "volume_col": volume_col,
        "trading_value_col": value_col,
        "program_net_buy_col": program_col,
        "execution_strength_col": execution_col,
    }

    # Volume acceleration, if available.
    if volume_col:
        x["_volume"] = pd.to_numeric(x[volume_col], errors="coerce")
        gv = x.groupby("ticker", group_keys=False)["_volume"]
        x["volume_lag1"] = gv.shift(1)
        x["volume_prev5_median"] = (
            gv.apply(lambda s: s.shift(1).rolling(5, min_periods=3).median())
            .reset_index(level=0, drop=True)
        )
        x["volume_accel_1d"] = finite_ratio(x["_volume"], x["volume_lag1"]) - 1.0
        x["volume_accel_5d"] = (
            finite_ratio(x["_volume"], x["volume_prev5_median"]) - 1.0
        )

    # Actual trading value if present; otherwise derive an explicit close*volume proxy.
    if value_col:
        x["_trading_value"] = pd.to_numeric(x[value_col], errors="coerce")
        detected["trading_value_source"] = value_col
    elif volume_col:
        x["_trading_value"] = x["close"] * x["_volume"]
        detected["trading_value_source"] = "PROXY_close_x_volume"
    else:
        detected["trading_value_source"] = None

    if "_trading_value" in x.columns:
        gv = x.groupby("ticker", group_keys=False)["_trading_value"]
        x["value_lag1"] = gv.shift(1)
        x["value_prev5_median"] = (
            gv.apply(lambda s: s.shift(1).rolling(5, min_periods=3).median())
            .reset_index(level=0, drop=True)
        )
        x["value_accel_1d"] = (
            finite_ratio(x["_trading_value"], x["value_lag1"]) - 1.0
        )
        x["value_accel_5d"] = (
            finite_ratio(x["_trading_value"], x["value_prev5_median"]) - 1.0
        )

    # Program net buying, if already merged into master.
    if program_col:
        x["_program_net_buy"] = pd.to_numeric(x[program_col], errors="coerce")

        if "_trading_value" in x.columns:
            # Signed demand normalized by turnover value.
            x["program_net_buy_ratio"] = (
                x["_program_net_buy"] / x["_trading_value"].abs().replace(0, np.nan)
            )
        else:
            # If denominator unavailable, still test signed level cross-sectionally.
            x["program_net_buy_ratio"] = x["_program_net_buy"]

        gp = x.groupby("ticker", group_keys=False)["program_net_buy_ratio"]
        x["program_net_buy_ratio_lag1"] = gp.shift(1)
        x["program_net_buy_accel_1d"] = (
            x["program_net_buy_ratio"] - x["program_net_buy_ratio_lag1"]
        )

    # Execution strength, if available.
    if execution_col:
        x["_execution_strength"] = pd.to_numeric(
            x[execution_col], errors="coerce"
        )
        ge = x.groupby("ticker", group_keys=False)["_execution_strength"]
        x["execution_strength_lag1"] = ge.shift(1)
        x["execution_strength_accel_1d"] = (
            x["_execution_strength"] - x["execution_strength_lag1"]
        )

    # Base Small Float x High Turnover setup.
    x["float_mcap_pct"] = (
        x.groupby("date")["free_float_market_cap"]
        .rank(method="average", pct=True, ascending=True)
    )
    x["turnover_pct"] = (
        x.groupby("date")["float_turnover_1d"]
        .rank(method="average", pct=True, ascending=True)
    )
    x["base_signal"] = (
        (x["float_mcap_pct"] <= 0.20)
        & (x["turnover_pct"] >= 0.80)
    )

    # Signal-internal ranks needed to reproduce strict-causal C5.
    sig = x["base_signal"].fillna(False)

    x["turnover_signal_pct"] = daily_pct_rank(
        x, sig, "float_turnover_1d", ascending=True
    )
    x["float_signal_pct"] = daily_pct_rank(
        x, sig, "free_float_market_cap", ascending=True
    )
    x["ret1_signal_pct"] = daily_pct_rank(
        x, sig, "ret_1d", ascending=True
    )
    x["ret5_signal_pct"] = daily_pct_rank(
        x, sig, "ret_5d", ascending=True
    )

    x["turnover_signal_q"] = q_from_pct(x["turnover_signal_pct"])
    x["float_signal_q"] = q_from_pct(x["float_signal_pct"])
    x["ret1_signal_q"] = q_from_pct(x["ret1_signal_pct"])
    x["ret5_signal_q"] = q_from_pct(x["ret5_signal_pct"])

    # Fixed strict-causal C5 baseline from previous stage.
    x["c5_baseline"] = (
        x["base_signal"]
        & (x["turnover_signal_q"] <= 2)
        & (x["float_signal_q"] <= 2)
        & (x["ret1_signal_q"] < 5)
    )

    # Demand acceleration ranks are computed WITHIN the C5 candidate set,
    # using only signal-date-close data.
    c5 = x["c5_baseline"].fillna(False)

    candidate_features = [
        "turnover_accel_1d",
        "turnover_accel_5d",
    ]
    for c in [
        "volume_accel_1d",
        "volume_accel_5d",
        "value_accel_1d",
        "value_accel_5d",
        "program_net_buy_ratio",
        "program_net_buy_accel_1d",
        "execution_strength_accel_1d",
    ]:
        if c in x.columns:
            candidate_features.append(c)

    for col in candidate_features:
        pct_col = f"{col}_c5_pct"
        q_col = f"{col}_c5_q"
        x[pct_col] = daily_pct_rank(x, c5, col, ascending=True)
        x[q_col] = q_from_pct(x[pct_col])

    detected["candidate_demand_features"] = candidate_features
    return x, detected


# ---------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    name: str
    feature: Optional[str] = None
    min_q: Optional[int] = None
    second_feature: Optional[str] = None
    second_min_q: Optional[int] = None


def build_rules(available_features: list[str]) -> list[Rule]:
    rules = [Rule("D0_C5_baseline")]

    # Core turnover-acceleration rules are always present.
    rules.extend(
        [
            Rule("D1_turnacc1_top60", "turnover_accel_1d", 3),
            Rule("D2_turnacc1_top40", "turnover_accel_1d", 4),
            Rule("D3_turnacc5_top60", "turnover_accel_5d", 3),
            Rule(
                "D4_turnacc1_and_5_top60",
                "turnover_accel_1d",
                3,
                "turnover_accel_5d",
                3,
            ),
        ]
    )

    # Optional pre-defined single-factor filters.
    optional_specs = [
        ("volume_accel_1d", "D5_volumeacc1_top60"),
        ("value_accel_1d", "D6_valueacc1_top60"),
        ("program_net_buy_ratio", "D7_programbuy_top60"),
        ("program_net_buy_accel_1d", "D8_programacc_top60"),
        ("execution_strength_accel_1d", "D9_execstrengthacc_top60"),
    ]
    for feat, name in optional_specs:
        if feat in available_features:
            rules.append(Rule(name, feat, 3))

    return rules


def apply_rule(events: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    x = events.copy()
    if rule.feature:
        qcol = f"{rule.feature}_c5_q"
        x = x[x[qcol] >= rule.min_q].copy()
    if rule.second_feature:
        qcol2 = f"{rule.second_feature}_c5_q"
        x = x[x[qcol2] >= rule.second_min_q].copy()
    return x


# ---------------------------------------------------------------------
# Executable outcomes
# ---------------------------------------------------------------------

def build_events(
    df: pd.DataFrame,
    tp: float,
    sl: float,
    max_hold: int,
    commission_bps: float,
    slippage_bps: float,
    demand_features: list[str],
) -> pd.DataFrame:
    rows = []
    buy_cost = (commission_bps + slippage_bps) / 10000.0
    sell_exec_cost = (commission_bps + slippage_bps) / 10000.0

    # Keep all causal C5/demand fields that will be used later.
    feature_cols = [
        "free_float_market_cap",
        "float_turnover_1d",
        "ret_1d",
        "ret_5d",
        "turnover_signal_pct",
        "float_signal_pct",
        "ret1_signal_pct",
        "ret5_signal_pct",
        "turnover_signal_q",
        "float_signal_q",
        "ret1_signal_q",
        "ret5_signal_q",
    ]

    for feat in demand_features:
        feature_cols.extend(
            [
                feat,
                f"{feat}_c5_pct",
                f"{feat}_c5_q",
            ]
        )

    feature_cols = [c for c in feature_cols if c in df.columns]

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        positions = np.flatnonzero(g["c5_baseline"].fillna(False).to_numpy())

        for pos in positions:
            pos = int(pos)
            entry_pos = pos + 1

            # Evaluation hygiene only: require complete future horizon.
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
                o = float(r.open)
                h = float(r.high)
                l = float(r.low)

                # Gap through stop.
                if np.isfinite(o) and o <= sl_px:
                    exit_reason = "SL_GAP"
                    exit_raw = o
                    exit_date = pd.Timestamp(r.date)
                    holding_bars = i
                    break

                hit_sl = np.isfinite(l) and l <= sl_px
                hit_tp = np.isfinite(h) and h >= tp_px

                # Conservative same-day assumption: stop first.
                if hit_sl and hit_tp:
                    exit_reason = "SL_SAME_BAR"
                    exit_raw = sl_px
                    exit_date = pd.Timestamp(r.date)
                    holding_bars = i
                    break
                elif hit_sl:
                    exit_reason = "SL"
                    exit_raw = sl_px
                    exit_date = pd.Timestamp(r.date)
                    holding_bars = i
                    break
                elif hit_tp:
                    exit_reason = "TP"
                    exit_raw = tp_px
                    exit_date = pd.Timestamp(r.date)
                    holding_bars = i
                    break

            tax = year_tax_rate(exit_date)
            exit_px = exit_raw * (1.0 - sell_exec_cost - tax)
            net_ret = exit_px / entry_px - 1.0

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
                "sell_tax_rate": tax,
                "trade_net_return": net_ret,
            }

            if "target_20d_30pct" in s.index:
                row["target_20d_30pct"] = s["target_20d_30pct"]

            for c in feature_cols:
                row[c] = s[c]

            rows.append(row)

    ev = pd.DataFrame(rows)
    if ev.empty:
        raise SystemExit("No complete-horizon C5 executable events generated.")
    return ev


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def event_stats(x: pd.DataFrame) -> dict:
    if x.empty:
        return {
            "events": 0,
            "tp_rate": np.nan,
            "sl_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "event_pf": np.nan,
        }

    pos = x.loc[x["trade_net_return"] > 0, "trade_net_return"].sum()
    neg = -x.loc[x["trade_net_return"] < 0, "trade_net_return"].sum()

    return {
        "events": int(len(x)),
        "tp_rate": float((x["exit_reason"] == "TP").mean()),
        "sl_rate": float(
            x["exit_reason"].isin(["SL", "SL_GAP", "SL_SAME_BAR"]).mean()
        ),
        "avg_return": float(x["trade_net_return"].mean()),
        "median_return": float(x["trade_net_return"].median()),
        "event_pf": float(pos / neg) if neg > 0 else np.nan,
    }


def feature_quintile_diagnostics(
    events: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for feat in features:
        qcol = f"{feat}_c5_q"
        if qcol not in events.columns:
            continue

        for year in [2025, 2026]:
            y = events[events["signal_date"].dt.year == year]
            for q in range(1, 6):
                z = y[y[qcol] == q]
                s = event_stats(z)
                rows.append(
                    {
                        "feature": feat,
                        "year": year,
                        "quintile": q,
                        **s,
                    }
                )
    return pd.DataFrame(rows)


def incremental_lift_table(
    events: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    for feat in features:
        qcol = f"{feat}_c5_q"
        if qcol not in events.columns:
            continue

        for year in [2025, 2026]:
            base = events[events["signal_date"].dt.year == year]
            top60 = base[base[qcol] >= 3]
            top40 = base[base[qcol] >= 4]

            b = event_stats(base)
            a = event_stats(top60)
            c = event_stats(top40)

            rows.append(
                {
                    "feature": feat,
                    "year": year,
                    "baseline_events": b["events"],
                    "baseline_avg_return": b["avg_return"],
                    "baseline_pf": b["event_pf"],
                    "top60_events": a["events"],
                    "top60_avg_return": a["avg_return"],
                    "top60_pf": a["event_pf"],
                    "top60_avg_return_lift": a["avg_return"] - b["avg_return"],
                    "top60_pf_lift": a["event_pf"] - b["event_pf"],
                    "top40_events": c["events"],
                    "top40_avg_return": c["avg_return"],
                    "top40_pf": c["event_pf"],
                    "top40_avg_return_lift": c["avg_return"] - b["avg_return"],
                    "top40_pf_lift": c["event_pf"] - b["event_pf"],
                }
            )
    return pd.DataFrame(rows)


def feature_corr(events: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    train = events[events["signal_date"].dt.year == 2025]
    usable = [f for f in features if f in train.columns]
    if not usable:
        return pd.DataFrame()
    return train[usable].corr(method="spearman")


# ---------------------------------------------------------------------
# Portfolio backtest
# ---------------------------------------------------------------------

def make_market_lookup(df, start, end):
    m = df.loc[
        (df["date"] >= start) & (df["date"] <= end),
        ["date", "ticker", "open", "close"],
    ]
    return {
        (pd.Timestamp(r.date), str(r.ticker)): (float(r.open), float(r.close))
        for r in m.itertuples(index=False)
    }


def rank_candidates(x: pd.DataFrame, rule: Rule) -> pd.DataFrame:
    """
    Causal ordering. Demand feature is used first only when the selected rule has one.
    Higher acceleration percentile is preferred after passing the rule filter.
    """
    y = x.copy()

    sort_cols = ["entry_date"]
    ascending = [True]

    if rule.feature:
        sort_cols.append(f"{rule.feature}_c5_pct")
        ascending.append(False)
    if rule.second_feature:
        sort_cols.append(f"{rule.second_feature}_c5_pct")
        ascending.append(False)

    # Preserve the successful strict-causal C5 structure after demand ranking.
    sort_cols += [
        "turnover_signal_q",
        "float_signal_q",
        "ret1_signal_pct",
        "ret5_signal_pct",
        "free_float_market_cap",
        "ticker",
    ]
    ascending += [True, True, True, True, True, True]

    return y.sort_values(sort_cols, ascending=ascending, na_position="last")


def summarize_equity(eq, tr, initial_capital):
    if eq.empty:
        return {
            "trades": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "calmar": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "avg_trade_net_return": np.nan,
            "median_trade_net_return": np.nan,
            "avg_exposure": np.nan,
            "avg_positions": np.nan,
        }

    curve = eq["equity"].astype(float)
    dd = curve / curve.cummax() - 1.0
    mdd = float(dd.min())
    total_return = float(curve.iloc[-1] / initial_capital - 1.0)

    days = max((eq["date"].iloc[-1] - eq["date"].iloc[0]).days, 1)
    years = days / 365.25
    cagr = (
        float((curve.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0)
        if curve.iloc[-1] > 0
        else -1.0
    )

    dr = eq["daily_return"].dropna()
    sharpe = (
        float(np.sqrt(252) * dr.mean() / dr.std(ddof=1))
        if len(dr) > 1 and dr.std(ddof=1) > 0
        else np.nan
    )
    calmar = float(cagr / abs(mdd)) if mdd < 0 else np.nan

    if tr.empty:
        pf = win = avg_trade = med_trade = np.nan
    else:
        pnl = tr["position_net_pnl"].astype(float)
        gain = pnl[pnl > 0].sum()
        loss = -pnl[pnl < 0].sum()
        pf = float(gain / loss) if loss > 0 else np.nan
        win = float((pnl > 0).mean())
        avg_trade = float(tr["trade_net_return"].mean())
        med_trade = float(tr["trade_net_return"].median())

    return {
        "trades": int(len(tr)),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": mdd,
        "sharpe": sharpe,
        "calmar": calmar,
        "profit_factor": pf,
        "win_rate": win,
        "avg_trade_net_return": avg_trade,
        "median_trade_net_return": med_trade,
        "avg_exposure": float(eq["exposure"].mean()),
        "avg_positions": float(eq["positions"].mean()),
    }


def backtest_rule(
    market_df,
    events,
    rule,
    start_date,
    end_date,
    initial_capital,
    max_positions,
):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    ev = events[
        (events["signal_date"] >= start)
        & (events["signal_date"] <= end)
        & (events["entry_date"] >= start)
        & (events["entry_date"] <= end)
    ].copy()

    ev = apply_rule(ev, rule)
    ev = rank_candidates(ev, rule)

    entry_map = {d: g.copy() for d, g in ev.groupby("entry_date", sort=True)}

    dates = (
        market_df.loc[
            (market_df["date"] >= start) & (market_df["date"] <= end),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    lookup = make_market_lookup(market_df, start, end)

    cash = float(initial_capital)
    positions = {}
    trades = []
    equity_rows = []
    prev_equity = float(initial_capital)

    for date in dates:
        date = pd.Timestamp(date)

        opening_equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            if px is not None:
                opening_equity += pos["shares"] * px[0]
            else:
                opening_equity += pos["shares"] * pos["last_close"]

        # Conservative: positions scheduled to exit today still occupy a slot at open.
        free_slots = max_positions - len(positions)

        candidates = entry_map.get(date)
        if free_slots > 0 and candidates is not None and not candidates.empty:
            candidates = rank_candidates(candidates, rule)

            for r in candidates.itertuples(index=False):
                if free_slots <= 0:
                    break

                ticker = str(r.ticker)
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
                    "signal_date": pd.Timestamp(r.signal_date),
                    "entry_date": pd.Timestamp(r.entry_date),
                    "exit_date": pd.Timestamp(r.exit_date),
                    "entry_price": float(r.entry_price),
                    "exit_price": float(r.exit_price),
                    "exit_reason": str(r.exit_reason),
                    "trade_net_return": float(r.trade_net_return),
                    "last_close": float(r.entry_open_raw),
                }
                free_slots -= 1

        exiting = [
            ticker for ticker, pos in positions.items()
            if pos["exit_date"] == date
        ]
        for ticker in exiting:
            pos = positions.pop(ticker)
            proceeds = pos["shares"] * pos["exit_price"]
            cash += proceeds
            pos["position_net_pnl"] = (
                pos["shares"] * (pos["exit_price"] - pos["entry_price"])
            )
            trades.append(pos)

        equity = cash
        for ticker, pos in positions.items():
            px = lookup.get((date, ticker))
            if px is not None:
                close_px = px[1]
                pos["last_close"] = close_px
            else:
                close_px = pos["last_close"]
            equity += pos["shares"] * close_px

        daily_return = equity / prev_equity - 1.0 if prev_equity > 0 else np.nan
        prev_equity = equity

        equity_rows.append(
            {
                "date": date,
                "equity": equity,
                "daily_return": daily_return,
                "cash": cash,
                "positions": len(positions),
                "exposure": (equity - cash) / equity if equity > 0 else np.nan,
            }
        )

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trades)
    summary = summarize_equity(eq, tr, initial_capital)
    summary["rule"] = rule.name
    return eq, tr, summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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

    df, detected = add_causal_features(df)
    features = detected["candidate_demand_features"]
    rules = build_rules(features)

    (out_dir / "detected_columns.json").write_text(
        json.dumps(detected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Primary event set at selection slippage.
    events = build_events(
        df=df,
        tp=args.tp,
        sl=args.sl,
        max_hold=args.max_hold,
        commission_bps=args.commission_bps,
        slippage_bps=args.selection_slippage_bps,
        demand_features=features,
    )
    events.to_parquet(out_dir / "demand_events_base_cost.parquet", index=False)

    # 1) Pure factor diagnostics.
    quintiles = feature_quintile_diagnostics(events, features)
    lifts = incremental_lift_table(events, features)
    corr = feature_corr(events, features)

    quintiles.to_csv(out_dir / "feature_quintiles_2025_2026.csv", index=False)
    lifts.to_csv(out_dir / "feature_incremental_lift.csv", index=False)
    if not corr.empty:
        corr.to_csv(out_dir / "feature_spearman_corr_2025.csv")

    # 2) 2025-only rule selection.
    train_rows = []
    train_outputs = {}

    for rule in rules:
        eq, tr, summary = backtest_rule(
            df,
            events,
            rule,
            "2025-01-01",
            "2025-12-31",
            args.initial_capital,
            args.max_positions,
        )

        event_train = apply_rule(
            events[events["signal_date"].dt.year == 2025],
            rule,
        )
        es = event_stats(event_train)
        summary.update(
            {
                "train_event_count": es["events"],
                "train_event_avg_return": es["avg_return"],
                "train_event_pf": es["event_pf"],
                "train_tp_rate": es["tp_rate"],
            }
        )
        train_rows.append(summary)
        train_outputs[rule.name] = (eq, tr)

    train_cmp = pd.DataFrame(train_rows)
    train_cmp.to_csv(out_dir / "train_rule_comparison_2025.csv", index=False)

    eligible = train_cmp[
        (train_cmp["trades"] >= args.min_train_trades)
        & train_cmp["calmar"].notna()
    ].copy()

    if eligible.empty:
        raise SystemExit("No demand rule met min train trades.")

    eligible = eligible.sort_values(
        ["calmar", "profit_factor", "cagr", "trades"],
        ascending=[False, False, False, False],
    )
    selected_name = str(eligible.iloc[0]["rule"])
    selected_rule = next(r for r in rules if r.name == selected_name)

    # 3) Frozen 2026 test at primary costs.
    test_eq, test_tr, test_summary = backtest_rule(
        df,
        events,
        selected_rule,
        "2026-01-01",
        "2026-08-29",
        args.initial_capital,
        args.max_positions,
    )
    test_ev = apply_rule(
        events[events["signal_date"].dt.year == 2026],
        selected_rule,
    )
    tes = event_stats(test_ev)
    test_summary.update(
        {
            "test_event_count": tes["events"],
            "test_event_avg_return": tes["avg_return"],
            "test_event_pf": tes["event_pf"],
            "test_tp_rate": tes["tp_rate"],
        }
    )

    full_eq, full_tr, full_summary = backtest_rule(
        df,
        events,
        selected_rule,
        "2025-01-01",
        "2026-08-29",
        args.initial_capital,
        args.max_positions,
    )

    # Save primary selected outputs.
    train_eq, train_tr = train_outputs[selected_name]
    train_eq.to_csv(out_dir / "selected_train_equity_2025.csv", index=False)
    train_tr.to_csv(out_dir / "selected_train_trades_2025.csv", index=False)
    test_eq.to_csv(out_dir / "selected_test_equity_2026.csv", index=False)
    test_tr.to_csv(out_dir / "selected_test_trades_2026.csv", index=False)
    full_eq.to_csv(out_dir / "selected_full_equity.csv", index=False)
    full_tr.to_csv(out_dir / "selected_full_trades.csv", index=False)
    pd.DataFrame([test_summary]).to_csv(
        out_dir / "selected_test_summary_2026.csv", index=False
    )
    pd.DataFrame([full_summary]).to_csv(
        out_dir / "selected_full_summary.csv", index=False
    )

    selection = {
        "selected_rule": asdict(selected_rule),
        "selection_period": "2025 only",
        "test_period": "2026-01-01 through 2026-08-29",
        "selection_metric": "Calmar; tie-break PF, CAGR, trades",
        "baseline": (
            "strict-causal C5: base setup + turnover signal Q1-2 + "
            "float signal Q1-2 + exclude ret1 signal Q5"
        ),
        "tax_assumption": {
            "2025_KOSDAQ_sell_tax": 0.0015,
            "2026_KOSDAQ_sell_tax": 0.0020,
        },
        "commission_bps_per_side": args.commission_bps,
        "selection_slippage_bps_per_side": args.selection_slippage_bps,
        "detected_columns": detected,
    }
    (out_dir / "selected_rule.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 4) Slippage stress: selected rule is frozen first, then costs change.
    stress_rows = []
    stress_values = [
        float(x.strip())
        for x in args.stress_slippage_bps.split(",")
        if x.strip()
    ]

    for slip in stress_values:
        ev_stress = build_events(
            df=df,
            tp=args.tp,
            sl=args.sl,
            max_hold=args.max_hold,
            commission_bps=args.commission_bps,
            slippage_bps=slip,
            demand_features=features,
        )

        for label, start, end in [
            ("2025", "2025-01-01", "2025-12-31"),
            ("2026", "2026-01-01", "2026-08-29"),
            ("full", "2025-01-01", "2026-08-29"),
        ]:
            _, _, sm = backtest_rule(
                df,
                ev_stress,
                selected_rule,
                start,
                end,
                args.initial_capital,
                args.max_positions,
            )
            stress_rows.append(
                {
                    "sample": label,
                    "slippage_bps_per_side": slip,
                    **sm,
                }
            )

    stress = pd.DataFrame(stress_rows)
    stress.to_csv(out_dir / "selected_rule_slippage_stress.csv", index=False)

    # Console
    print("\n=== DEMAND FEATURE AVAILABILITY ===")
    print(json.dumps(detected, ensure_ascii=False, indent=2))

    print("\n=== DEMAND ACCELERATION: INCREMENTAL LIFT ===")
    if lifts.empty:
        print("No demand features available.")
    else:
        print(lifts.to_string(index=False))

    print("\n=== 2025 TRAIN: DEMAND RULE COMPARISON ===")
    print(
        train_cmp.sort_values(
            ["calmar", "profit_factor"],
            ascending=[False, False],
        ).to_string(index=False)
    )

    print("\n=== SELECTED DEMAND RULE (2025 ONLY) ===")
    print(json.dumps(selection, ensure_ascii=False, indent=2))

    print("\n=== 2026 FROZEN DEMAND TEST ===")
    print(pd.DataFrame([test_summary]).to_string(index=False))

    print("\n=== FULL CONTINUOUS DIAGNOSTIC ===")
    print(pd.DataFrame([full_summary]).to_string(index=False))

    print("\n=== SELECTED RULE: SLIPPAGE STRESS ===")
    show_cols = [
        "sample",
        "slippage_bps_per_side",
        "trades",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "calmar",
        "profit_factor",
        "avg_trade_net_return",
    ]
    print(stress[show_cols].to_string(index=False))

    print("\nImportant:")
    print("- C5 strict-causal baseline is NOT retuned.")
    print("- All demand features are known by signal-date close.")
    print("- 2025 selects; 2026 is not used by the code for selection.")
    print("- KOSDAQ sell tax: 15 bps in 2025, 20 bps in 2026.")
    print("- Program / execution-strength factors are tested only if columns exist.")
    print("- Missing optional factors remain untested; they are not fabricated.")
    print("- Slippage stress is applied AFTER the rule is frozen.")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
