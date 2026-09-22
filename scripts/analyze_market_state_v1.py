#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--index", default="data/pit_kosdaq/market_state/kosdaq_index_kq11.parquet")
    p.add_argument("--out-dir", default="data/pit_kosdaq/analysis/market_state_v1")
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    return p.parse_args()

def pick_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def qbucket(s):
    return np.ceil(s * 5).clip(1, 5).astype("Int64")

def add_c5(x):
    x = x.sort_values(["ticker","date"]).copy()
    x["prev_close"] = x.groupby("ticker")["close"].shift(1)
    x["ret_1d"] = x["close"] / x["prev_close"] - 1
    x["float_mcap_pct"] = x.groupby("date")["free_float_market_cap"].rank(pct=True)
    x["turnover_pct"] = x.groupby("date")["float_turnover_1d"].rank(pct=True)
    x["base_signal"] = (x["float_mcap_pct"] <= 0.20) & (x["turnover_pct"] >= 0.80)
    sig = x["base_signal"].fillna(False)

    for col,out in [
        ("float_turnover_1d","turn_signal_pct"),
        ("free_float_market_cap","float_signal_pct"),
        ("ret_1d","ret1_signal_pct"),
    ]:
        x[out] = np.nan
        r = x.loc[sig].groupby("date")[col].rank(method="average", pct=True)
        x.loc[r.index, out] = r

    x["turn_q"] = qbucket(x["turn_signal_pct"])
    x["float_q"] = qbucket(x["float_signal_pct"])
    x["ret1_q"] = qbucket(x["ret1_signal_pct"])
    x["c5"] = (
        x["base_signal"] &
        (x["turn_q"] <= 2) &
        (x["float_q"] <= 2) &
        (x["ret1_q"] < 5)
    )
    return x

def build_market_state(model, idx):
    x = model.sort_values(["ticker","date"]).copy()

    # PIT cross-sectional state from the investable KOSDAQ panel.
    x["ma20"] = x.groupby("ticker")["close"].transform(
        lambda z: z.rolling(20, min_periods=15).mean()
    )
    x["ret5"] = x.groupby("ticker")["close"].pct_change(5, fill_method=None)
    x["ret1_mkt"] = x.groupby("ticker")["close"].pct_change(fill_method=None)

    value_col = pick_col(
        x,
        ["trading_value", "trading_value_1d", "trade_value", "value", "amount", "acc_trdval"]
    )
    value_source = value_col if value_col else "close_x_volume_proxy"
    if value_col is None:
        x["_market_value"] = x["close"] * x["volume"]
    else:
        x["_market_value"] = pd.to_numeric(x[value_col], errors="coerce")

    daily = x.groupby("date").agg(
        universe_n=("ticker","nunique"),
        breadth_above_ma20=("ma20", lambda z: np.nan),  # replaced below
        agg_trading_value=("_market_value","sum"),
        xsec_dispersion=("ret1_mkt","std"),
    ).reset_index()

    cs = x.groupby("date").agg(
        breadth_above_ma20=("close", lambda z: np.nan),
    )
    # Explicit calculations with aligned masks.
    breadth_rows = []
    for d,g in x.groupby("date", sort=True):
        valid_ma = g["ma20"].notna() & g["close"].notna()
        valid_r5 = g["ret5"].notna()
        valid_r1 = g["ret1_mkt"].notna()
        breadth_rows.append({
            "date": d,
            "breadth_above_ma20": float((g.loc[valid_ma,"close"] > g.loc[valid_ma,"ma20"]).mean()) if valid_ma.any() else np.nan,
            "breadth_ret5_pos": float((g.loc[valid_r5,"ret5"] > 0).mean()) if valid_r5.any() else np.nan,
            "advance_ratio": float((g.loc[valid_r1,"ret1_mkt"] > 0).mean()) if valid_r1.any() else np.nan,
        })
    breadth = pd.DataFrame(breadth_rows)
    daily = daily.drop(columns=["breadth_above_ma20"]).merge(breadth, on="date", how="left")

    daily = daily.sort_values("date")
    daily["value_med20_lag1"] = daily["agg_trading_value"].shift(1).rolling(20, min_periods=15).median()
    daily["value_ratio20"] = daily["agg_trading_value"] / daily["value_med20_lag1"].replace(0,np.nan)

    k = idx.copy().sort_values("date")
    k["kq_ma20"] = k["kq_close"].rolling(20, min_periods=15).mean()
    k["kq_ma60"] = k["kq_close"].rolling(60, min_periods=40).mean()
    k["kq_ret20"] = k["kq_close"].pct_change(20, fill_method=None)
    k["kq_ret1"] = k["kq_close"].pct_change(fill_method=None)
    k["kq_rv20"] = k["kq_ret1"].rolling(20, min_periods=15).std() * np.sqrt(252)
    k["kq_high60"] = k["kq_close"].rolling(60, min_periods=40).max()
    k["kq_drawdown60"] = k["kq_close"] / k["kq_high60"] - 1

    state = daily.merge(
        k[["date","kq_close","kq_ma20","kq_ma60","kq_ret20","kq_rv20","kq_drawdown60"]],
        on="date", how="left"
    )
    return state, value_source

def train_thresholds(state):
    tr = state[state["date"].dt.year == 2025].copy()
    return {
        "breadth_above_ma20_median_2025": float(tr["breadth_above_ma20"].median()),
        "breadth_ret5_pos_median_2025": float(tr["breadth_ret5_pos"].median()),
        "value_ratio20_median_2025": float(tr["value_ratio20"].median()),
        "kq_rv20_p80_2025": float(tr["kq_rv20"].quantile(.80)),
        "kq_drawdown60_p20_2025": float(tr["kq_drawdown60"].quantile(.20)),
    }

def apply_regimes(state, thr):
    s = state.copy()
    s["trend_up"] = (s["kq_close"] > s["kq_ma60"]) & (s["kq_ret20"] > 0)
    s["breadth_strong"] = s["breadth_above_ma20"] >= thr["breadth_above_ma20_median_2025"]
    s["breadth5_strong"] = s["breadth_ret5_pos"] >= thr["breadth_ret5_pos_median_2025"]
    s["value_active"] = s["value_ratio20"] >= thr["value_ratio20_median_2025"]
    s["not_high_vol"] = s["kq_rv20"] <= thr["kq_rv20_p80_2025"]
    s["not_deep_drawdown"] = s["kq_drawdown60"] >= thr["kq_drawdown60_p20_2025"]

    cols = ["trend_up","breadth_strong","value_active","not_high_vol","not_deep_drawdown"]
    s["risk_on_score"] = s[cols].fillna(False).astype(int).sum(axis=1)

    s["trend_breadth"] = s["trend_up"] & s["breadth_strong"]
    s["trend_breadth_value"] = s["trend_breadth"] & s["value_active"]
    s["trend_breadth_value_safevol"] = s["trend_breadth_value"] & s["not_high_vol"]
    s["risk_on_3of5"] = s["risk_on_score"] >= 3
    s["risk_on_4of5"] = s["risk_on_score"] >= 4
    return s

def make_events(x, hold, tp, sl):
    rows=[]
    for ticker,g in x.groupby("ticker", sort=False):
        g=g.sort_values("date").reset_index(drop=True); n=len(g)
        for pos in np.flatnonzero(g["c5"].fillna(False).to_numpy()):
            pos=int(pos); ep=pos+1
            if ep+hold-1>=n: continue
            sig=g.iloc[pos]; ent=g.iloc[ep]
            entry=float(ent["open"])
            if not np.isfinite(entry) or entry<=0: continue
            w=g.iloc[ep:ep+hold]
            tp_px=entry*(1+tp); sl_px=entry*(1+sl)
            reason="TIME"; exit_px=float(w.iloc[-1]["close"])
            for r in w.itertuples(index=False):
                o,h,l=float(r.open),float(r.high),float(r.low)
                if o<=sl_px: reason,exit_px="SL",o; break
                if l<=sl_px: reason,exit_px="SL",sl_px; break
                if h>=tp_px: reason,exit_px="TP",tp_px; break
            max_high=float(w["high"].max()); sc=float(sig["close"])
            max_ret=max_high/sc-1 if sc>0 else np.nan
            rows.append({
                "ticker":ticker, "signal_date":pd.Timestamp(sig["date"]),
                "year":pd.Timestamp(sig["date"]).year,
                "trade_return_gross":exit_px/entry-1,
                "tp_before_sl":int(reason=="TP"),
                "target30":int(max_ret>=.30),
                "target50":int(max_ret>=.50),
                "target100":int(max_ret>=1.0),
                "max_signal_return_20d":max_ret,
            })
    return pd.DataFrame(rows)

def pf(r):
    pos=r[r>0].sum(); neg=-r[r<0].sum()
    return pos/neg if neg>0 else np.nan

def stats(g):
    if len(g)==0:
        return dict(n=0,p30=np.nan,p50=np.nan,p100=np.nan,tp=np.nan,avg_return=np.nan,median_return=np.nan,pf=np.nan,avg_max20=np.nan)
    return dict(
        n=len(g),
        p30=g["target30"].mean(),
        p50=g["target50"].mean(),
        p100=g["target100"].mean(),
        tp=g["tp_before_sl"].mean(),
        avg_return=g["trade_return_gross"].mean(),
        median_return=g["trade_return_gross"].median(),
        pf=pf(g["trade_return_gross"]),
        avg_max20=g["max_signal_return_20d"].mean(),
    )

def frozen_quintile_table(ev, col):
    tr=ev[(ev.year==2025)&ev[col].notna()]
    if tr.empty: return pd.DataFrame()
    cuts=np.unique(tr[col].quantile([0,.2,.4,.6,.8,1]).to_numpy(float))
    if len(cuts)<3: return pd.DataFrame()
    cuts[0],cuts[-1]=-np.inf,np.inf
    rows=[]
    for yr in [2025,2026]:
        y=ev[(ev.year==yr)&ev[col].notna()].copy()
        y["bucket"]=pd.cut(y[col], bins=cuts, labels=False, include_lowest=True)+1
        for b,g in y.groupby("bucket"):
            rows.append({"feature":col,"year":yr,"bucket_2025_cutoffs":int(b),**stats(g)})
    return pd.DataFrame(rows)

def main():
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    import pyarrow.parquet as pq

    # Read only columns needed for Market State/C5 reconstruction.
    schema_cols = set(pq.ParquetFile(a.model).schema_arrow.names)

    cols = [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "free_float_market_cap",
        "float_turnover_1d",
    ]

    # Prefer an actual trading-value column if present.
    optional_value_cols = [
        "trading_value",
        "trading_value_1d",
        "trade_value",
        "value",
        "amount",
        "acc_trdval",
    ]

    for c in optional_value_cols:
        if c in schema_cols:
            cols.append(c)
            break

    missing = [c for c in cols[:9] if c not in schema_cols]
    if missing:
        raise RuntimeError(f"Missing required model columns: {missing}")

    print("Reading model columns:", cols)
    m = pd.read_parquet(a.model, columns=cols)
    m["ticker"]=m["ticker"].astype(str).str.zfill(6); m["date"]=pd.to_datetime(m["date"])
    m=add_c5(m)

    idx=pd.read_parquet(a.index); idx["date"]=pd.to_datetime(idx["date"])
    state,value_source=build_market_state(m,idx)
    thr=train_thresholds(state)
    state=apply_regimes(state,thr)

    ev=make_events(m,a.max_hold,a.tp,a.sl)
    ev=ev.merge(
        state.rename(columns={"date": "signal_date"}),
        on="signal_date",
        how="left"
    )

    regime_cols=[
        "trend_up","breadth_strong","breadth5_strong","value_active",
        "not_high_vol","not_deep_drawdown","trend_breadth",
        "trend_breadth_value","trend_breadth_value_safevol",
        "risk_on_3of5","risk_on_4of5"
    ]

    rows=[]
    for yr in [2025,2026]:
        y=ev[ev.year==yr]
        rows.append({"year":yr,"regime":"baseline_all_C5",**stats(y)})
        for c in regime_cols:
            rows.append({"year":yr,"regime":c,**stats(y[y[c].fillna(False)])})
    summary=pd.DataFrame(rows)

    qtables=[]
    for col in ["kq_ret20","breadth_above_ma20","breadth_ret5_pos","value_ratio20","kq_rv20","kq_drawdown60","xsec_dispersion"]:
        q=frozen_quintile_table(ev,col)
        if not q.empty: qtables.append(q)
    qdiag=pd.concat(qtables,ignore_index=True) if qtables else pd.DataFrame()

    score_rows=[]
    for yr in [2025,2026]:
        y=ev[ev.year==yr]
        for score,g in y.groupby("risk_on_score"):
            score_rows.append({"year":yr,"risk_on_score":int(score),**stats(g)})
    score=pd.DataFrame(score_rows)

    coverage=ev.groupby("year").agg(
        events=("ticker","size"),
        state_coverage=("kq_close",lambda z:z.notna().mean()),
        breadth_coverage=("breadth_above_ma20",lambda z:z.notna().mean()),
        value_coverage=("value_ratio20",lambda z:z.notna().mean()),
        vol_coverage=("kq_rv20",lambda z:z.notna().mean()),
    ).reset_index()

    state.to_parquet(out/"market_state_daily.parquet",index=False)
    ev.to_parquet(out/"market_state_c5_events.parquet",index=False)
    summary.to_csv(out/"market_state_regime_summary.csv",index=False)
    qdiag.to_csv(out/"market_state_quintile_diagnostic.csv",index=False)
    score.to_csv(out/"market_state_risk_on_score.csv",index=False)
    coverage.to_csv(out/"market_state_coverage.csv",index=False)
    (out/"market_state_thresholds_2025.json").write_text(json.dumps(thr,indent=2),encoding="utf-8")
    (out/"market_state_meta.json").write_text(json.dumps({"trading_value_source":value_source},indent=2),encoding="utf-8")

    print("\n=== MARKET STATE COVERAGE ===")
    print(coverage.to_string(index=False))
    print("\n=== MARKET STATE 2025 FROZEN THRESHOLDS ===")
    print(json.dumps(thr,indent=2))
    print("\n=== MARKET STATE REGIME SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== MARKET STATE RISK-ON SCORE ===")
    print(score.to_string(index=False))
    print("\n=== MARKET STATE QUINTILE DIAGNOSTIC ===")
    print(qdiag.to_string(index=False))
    print("\ntrading_value_source =",value_source)
    print("wrote ->",out)

if __name__=="__main__":
    main()
