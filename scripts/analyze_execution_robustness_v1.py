#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

MAX_POSITIONS = [3, 10]
SLIPPAGE_BPS = [10.0, 30.0, 50.0, 100.0]
PARTICIPATION = [0.001, 0.0025, 0.005, 0.01]  # 0.10%, 0.25%, 0.50%, 1.00%
CAPITALS = [1_000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000]

BASE_COMMISSION_BPS = 7.5
BASE_SLIPPAGE_BPS = 10.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--candidate-trades", default="data/pit_kosdaq/analysis/portfolio_construction_v1/portfolio_candidate_trades_full.parquet")
    p.add_argument("--out-dir", default="data/pit_kosdaq/analysis/execution_robustness_v1")
    p.add_argument("--commission-bps", type=float, default=7.5)
    return p.parse_args()


def tax_bps(year):
    return 15.0 if int(year) == 2025 else 20.0


def add_execution_fields(model, trades):
    x = model.sort_values(["ticker", "date"]).copy()

    # Signal-date-known liquidity. Includes t-day value, which is known after t close.
    x["adv20_value_t"] = x.groupby("ticker")["trading_value"].transform(
        lambda z: z.rolling(20, min_periods=10).mean()
    )
    x["median20_value_t"] = x.groupby("ticker")["trading_value"].transform(
        lambda z: z.rolling(20, min_periods=10).median()
    )

    sig = x[["ticker","date","close","adv20_value_t","median20_value_t","trading_value"]].rename(
        columns={
            "date":"signal_date",
            "close":"signal_close",
            "trading_value":"signal_trading_value",
        }
    )
    ent = x[["ticker","date","open","high","low","close"]].rename(
        columns={
            "date":"entry_date",
            "open":"entry_open",
            "high":"entry_high",
            "low":"entry_low",
            "close":"entry_close",
        }
    )

    t = trades.merge(sig, on=["ticker","signal_date"], how="left")
    t = t.merge(ent, on=["ticker","entry_date"], how="left")

    # Reconstruct pre-cost raw entry/exit from the v1 portfolio candidate table.
    base_buy_factor = (1 + BASE_SLIPPAGE_BPS/10000) * (1 + BASE_COMMISSION_BPS/10000)
    t["entry_raw_reconstructed"] = t["entry_cash_ps"] / base_buy_factor

    base_sell_factor = 1 - (
        BASE_COMMISSION_BPS
        + BASE_SLIPPAGE_BPS
        + t["exit_date"].dt.year.map(tax_bps)
    ) / 10000
    t["exit_raw_reconstructed"] = t["exit_cash_ps"] / base_sell_factor

    # Market-calendar true next trading date.
    market_dates = pd.DatetimeIndex(sorted(pd.to_datetime(x["date"].dropna().unique())))
    next_map = {market_dates[i]: market_dates[i+1] for i in range(len(market_dates)-1)}
    t["expected_next_market_date"] = t["signal_date"].map(next_map)
    t["delayed_entry"] = t["entry_date"] != t["expected_next_market_date"]

    # Conservative one-price upper-limit lock proxy.
    gap = t["entry_open"] / t["signal_close"] - 1
    one_price = (
        np.isclose(t["entry_open"], t["entry_high"], rtol=0, atol=1e-12)
        & np.isclose(t["entry_open"], t["entry_low"], rtol=0, atol=1e-12)
        & np.isclose(t["entry_open"], t["entry_close"], rtol=0, atol=1e-12)
    )
    t["entry_gap"] = gap
    t["locked_limit_up_proxy"] = one_price & (gap >= 0.285)
    t["strict_tradable"] = (~t["delayed_entry"]) & (~t["locked_limit_up_proxy"])

    return t


def apply_fixed_path_costs(t, commission_bps, slippage_bps):
    z = t.copy()
    buy = (1 + slippage_bps/10000) * (1 + commission_bps/10000)
    sell = 1 - (
        commission_bps
        + slippage_bps
        + z["exit_date"].dt.year.map(tax_bps)
    ) / 10000
    z["stress_entry_cash_ps"] = z["entry_raw_reconstructed"] * buy
    z["stress_exit_cash_ps"] = z["exit_raw_reconstructed"] * sell
    z["stress_net_return"] = z["stress_exit_cash_ps"] / z["stress_entry_cash_ps"] - 1
    return z


def portfolio_sim(
    trades,
    model,
    max_positions,
    initial_capital,
    participation_cap=None,
):
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    trades = trades.sort_values(["entry_date","priority_score","ticker"]).copy()
    by_entry = {
        d:g.sort_values(["priority_score","ticker"])
        for d,g in trades.groupby("entry_date")
    }

    closes = {
        (str(t), pd.Timestamp(d)): float(c)
        for t,d,c in model[["ticker","date","close"]].itertuples(index=False, name=None)
    }
    dates = sorted(pd.to_datetime(model["date"].unique()))
    min_d, max_d = trades["entry_date"].min(), trades["exit_date"].max()
    dates = [d for d in dates if d >= min_d and d <= max_d]

    cash = float(initial_capital)
    positions = {}
    curve, executed = [], []
    prev_equity = float(initial_capital)

    for d in dates:
        candidates = by_entry.get(pd.Timestamp(d))
        slots = max_positions - len(positions)

        if candidates is not None and slots > 0 and cash > 1e-9:
            target = prev_equity / max_positions

            for r in candidates.itertuples(index=False):
                if slots <= 0 or cash <= 1e-9:
                    break
                if r.ticker in positions:
                    continue

                alloc = min(target, cash)

                if participation_cap is not None:
                    adv = float(r.adv20_value_t) if pd.notna(r.adv20_value_t) else np.nan
                    if not np.isfinite(adv) or adv <= 0:
                        continue
                    alloc = min(alloc, adv * participation_cap)

                if alloc <= 1.0:
                    continue

                shares = alloc / float(r.stress_entry_cash_ps)
                actual_notional = shares * float(r.stress_entry_cash_ps)
                cash -= actual_notional

                positions[r.ticker] = {
                    "shares":shares,
                    "exit_date":pd.Timestamp(r.exit_date),
                    "exit_cash_ps":float(r.stress_exit_cash_ps),
                    "entry_date":pd.Timestamp(r.entry_date),
                    "entry_notional":actual_notional,
                    "exit_reason":r.exit_reason,
                    "signal_date":pd.Timestamp(r.signal_date),
                    "adv20_value_t":float(r.adv20_value_t) if pd.notna(r.adv20_value_t) else np.nan,
                }
                slots -= 1

        # Same-day exits are processed after entries: no same-day cash recycling.
        exits = [t for t,p in positions.items() if p["exit_date"] == pd.Timestamp(d)]
        for ticker in exits:
            p = positions.pop(ticker)
            proceeds = p["shares"] * p["exit_cash_ps"]
            cash += proceeds
            executed.append({
                "ticker":ticker,
                "signal_date":p["signal_date"],
                "entry_date":p["entry_date"],
                "exit_date":pd.Timestamp(d),
                "entry_notional":p["entry_notional"],
                "proceeds":proceeds,
                "trade_return":proceeds/p["entry_notional"]-1,
                "exit_reason":p["exit_reason"],
                "adv20_value_t":p["adv20_value_t"],
                "realized_participation":(
                    p["entry_notional"]/p["adv20_value_t"]
                    if np.isfinite(p["adv20_value_t"]) and p["adv20_value_t"] > 0
                    else np.nan
                ),
            })

        mtm = 0.0
        for ticker,p in positions.items():
            px = closes.get((str(ticker),pd.Timestamp(d)))
            if px is not None and np.isfinite(px):
                mtm += p["shares"] * px
            else:
                mtm += p["entry_notional"]

        equity = cash + mtm
        curve.append({
            "date":pd.Timestamp(d),
            "equity":equity,
            "cash":cash,
            "positions":len(positions),
        })
        prev_equity = equity

    return pd.DataFrame(curve), pd.DataFrame(executed)


def performance(curve, trades, initial_capital, period):
    if curve.empty:
        return {}

    c = curve.sort_values("date").reset_index(drop=True)

    if period == "2025":
        z = c[c["date"].dt.year == 2025].copy()
    elif period == "2026":
        z = c[c["date"].dt.year == 2026].copy()
    else:
        z = c.copy()

    if z.empty:
        return {}

    first_i = z.index[0]
    if first_i > 0:
        start_equity = float(c.loc[first_i-1,"equity"])
    else:
        start_equity = float(initial_capital)
    end_equity = float(z.iloc[-1]["equity"])

    days = max((z.iloc[-1]["date"] - z.iloc[0]["date"]).days, 1)
    total = end_equity/start_equity - 1
    annualized = (1+total)**(365/days)-1 if total > -1 else -1

    eq = pd.concat(
        [pd.Series([start_equity]), z["equity"].reset_index(drop=True)],
        ignore_index=True
    )
    dd = eq/eq.cummax()-1
    r = eq.pct_change().dropna()
    sharpe = np.sqrt(252)*r.mean()/r.std() if r.std() > 0 else np.nan

    tt = trades.copy()
    if not tt.empty:
        if period == "2025":
            tt = tt[tt["entry_date"].dt.year == 2025]
        elif period == "2026":
            tt = tt[tt["entry_date"].dt.year == 2026]

    if tt.empty:
        n=0; pf=avg=win=part=np.nan
    else:
        pos = tt.loc[tt["trade_return"]>0,"trade_return"].sum()
        neg = -tt.loc[tt["trade_return"]<0,"trade_return"].sum()
        pf = pos/neg if neg > 0 else np.nan
        avg = tt["trade_return"].mean()
        win = (tt["trade_return"]>0).mean()
        part = tt["realized_participation"].median()
        n = len(tt)

    return {
        "period":period,
        "total_return":total,
        "annualized_return":annualized,
        "mdd":dd.min(),
        "sharpe":sharpe,
        "trades":n,
        "trade_pf":pf,
        "avg_trade_return":avg,
        "win_rate":win,
        "median_realized_participation":part,
        "avg_positions":z["positions"].mean(),
        "avg_cash_pct":(z["cash"]/z["equity"]).mean(),
    }


def capacity_summary(t):
    z=t[t["strict_tradable"] & t["adv20_value_t"].notna() & (t["adv20_value_t"]>0)].copy()
    rows=[]
    for n in MAX_POSITIONS:
        for p in PARTICIPATION:
            cap = z["adv20_value_t"] * p * n
            rows.append({
                "max_positions":n,
                "participation_cap":p,
                "capacity_p10_krw":cap.quantile(.10),
                "capacity_p25_krw":cap.quantile(.25),
                "capacity_median_krw":cap.median(),
                "capacity_p75_krw":cap.quantile(.75),
                "capacity_p90_krw":cap.quantile(.90),
            })
    return pd.DataFrame(rows)


def main():
    a=parse_args()
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)

    model_cols=[
        "ticker","date","open","high","low","close",
        "trading_value",
    ]
    m=pd.read_parquet(a.model,columns=model_cols)
    m["ticker"]=m["ticker"].astype(str).str.zfill(6)
    m["date"]=pd.to_datetime(m["date"])

    t=pd.read_parquet(a.candidate_trades)
    t["ticker"]=t["ticker"].astype(str).str.zfill(6)
    for c in ["signal_date","entry_date","exit_date"]:
        t[c]=pd.to_datetime(t[c])

    # Main strategy only: Market Weakness + original exit + open.
    t=t[
        (t["entry_rule"]=="OPEN")
        & (t["exit_rule"]=="BASE_EXIT")
        & (t["market_weak2"])
    ].copy()

    t=add_execution_fields(m,t)
    t.to_parquet(out/"execution_enriched_candidates.parquet",index=False)

    cov=pd.DataFrame([{
        "candidate_trades":len(t),
        "adv20_coverage":t["adv20_value_t"].notna().mean(),
        "delayed_entry_count":int(t["delayed_entry"].sum()),
        "delayed_entry_rate":t["delayed_entry"].mean(),
        "locked_limit_up_proxy_count":int(t["locked_limit_up_proxy"].sum()),
        "locked_limit_up_proxy_rate":t["locked_limit_up_proxy"].mean(),
        "strict_tradable_count":int(t["strict_tradable"].sum()),
        "strict_tradable_rate":t["strict_tradable"].mean(),
    }])
    cov.to_csv(out/"execution_coverage.csv",index=False)

    cap=capacity_summary(t)
    cap.to_csv(out/"liquidity_capacity_summary.csv",index=False)

    # A. Fixed-path slippage stress, no liquidity cap, both raw and strict tradability.
    stress_rows=[]
    for tradability_name,base_t in [
        ("RAW_BACKTEST",t),
        ("STRICT_TRADABLE",t[t["strict_tradable"]]),
    ]:
        for slip in SLIPPAGE_BPS:
            z=apply_fixed_path_costs(base_t,a.commission_bps,slip)
            for n in MAX_POSITIONS:
                curve,ex=portfolio_sim(
                    z,m,n,initial_capital=100_000_000,
                    participation_cap=None
                )
                for period in ["2025","2026","full"]:
                    p=performance(curve,ex,100_000_000,period)
                    if p:
                        stress_rows.append({
                            "tradability":tradability_name,
                            "slippage_bps_per_side":slip,
                            "max_positions":n,
                            **p
                        })
    stress=pd.DataFrame(stress_rows)
    stress.to_csv(out/"fixed_path_slippage_stress.csv",index=False)

    # B. Liquidity-cap stress at base 10bp slippage, strict tradability.
    liq_rows=[]
    strict=apply_fixed_path_costs(
        t[t["strict_tradable"]],
        a.commission_bps,
        BASE_SLIPPAGE_BPS
    )
    for capital in CAPITALS:
        for part in PARTICIPATION:
            for n in MAX_POSITIONS:
                curve,ex=portfolio_sim(
                    strict,m,n,
                    initial_capital=capital,
                    participation_cap=part
                )
                for period in ["2025","2026","full"]:
                    p=performance(curve,ex,capital,period)
                    if p:
                        liq_rows.append({
                            "initial_capital_krw":capital,
                            "participation_cap":part,
                            "max_positions":n,
                            **p
                        })
    liq=pd.DataFrame(liq_rows)
    liq.to_csv(out/"capital_liquidity_stress.csv",index=False)

    print("\n=== EXECUTION ROBUSTNESS COVERAGE ===")
    print(cov.to_string(index=False))

    print("\n=== LIQUIDITY CAPACITY SUMMARY ===")
    print(cap.to_string(index=False))

    print("\n=== FIXED-PATH SLIPPAGE STRESS ===")
    print(
        stress[
            ["tradability","slippage_bps_per_side","max_positions","period",
             "total_return","annualized_return","mdd","sharpe","trades",
             "trade_pf","avg_trade_return","avg_positions","avg_cash_pct"]
        ].to_string(index=False)
    )

    print("\n=== CAPITAL / ADV PARTICIPATION STRESS: FULL PERIOD ===")
    full=liq[liq["period"]=="full"].copy()
    print(
        full[
            ["initial_capital_krw","participation_cap","max_positions",
             "total_return","annualized_return","mdd","sharpe","trades",
             "trade_pf","avg_trade_return","median_realized_participation",
             "avg_positions","avg_cash_pct"]
        ].to_string(index=False)
    )

    print("\n=== 100M KRW LIQUIDITY STRESS BY YEAR ===")
    h=liq[liq["initial_capital_krw"]==100_000_000].copy()
    print(
        h[
            ["participation_cap","max_positions","period","total_return",
             "annualized_return","mdd","sharpe","trades","trade_pf",
             "avg_trade_return","median_realized_participation","avg_cash_pct"]
        ].to_string(index=False)
    )

    print("\nNOTES")
    print("- Fixed-path stress keeps the original TP/SL/hold path and changes costs only.")
    print("- ADV20 uses trailing 20 trading-value observations through signal date t, so it is known after t close.")
    print("- strict_tradable excludes signals whose ticker has no row on the market's true next trading date.")
    print("- strict_tradable also excludes one-price entry days with >= +28.5% gap as a conservative locked-upper-limit proxy.")
    print("- Liquidity cap limits each entry notional to participation_cap * signal-date ADV20.")
    print("\nwrote ->",out)

if __name__=="__main__":
    main()
