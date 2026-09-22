#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--short-balance", default="data/pit_kosdaq/true_short_balance/true_short_balance_daily.parquet")
    p.add_argument("--out-dir", default="data/pit_kosdaq/analysis/true_dtc_zero_aware_final")
    p.add_argument("--publication-lag-bars", type=int, default=2)
    p.add_argument("--max-hold", type=int, default=20)
    p.add_argument("--tp", type=float, default=0.30)
    p.add_argument("--sl", type=float, default=-0.15)
    return p.parse_args()

def qbucket(s): return np.ceil(s*5).clip(1,5).astype("Int64")

def add_c5(x):
    x=x.sort_values(["ticker","date"]).copy()
    x["prev_close"]=x.groupby("ticker")["close"].shift(1)
    x["ret_1d"]=x["close"]/x["prev_close"]-1
    x["float_mcap_pct"]=x.groupby("date")["free_float_market_cap"].rank(pct=True)
    x["turnover_pct"]=x.groupby("date")["float_turnover_1d"].rank(pct=True)
    x["base_signal"]=(x["float_mcap_pct"]<=.20)&(x["turnover_pct"]>=.80)
    sig=x["base_signal"].fillna(False)
    for col,out in [("float_turnover_1d","turn_signal_pct"),("free_float_market_cap","float_signal_pct"),("ret_1d","ret1_signal_pct")]:
        x[out]=np.nan
        r=x.loc[sig].groupby("date")[col].rank(method="average",pct=True)
        x.loc[r.index,out]=r
    x["turn_q"]=qbucket(x["turn_signal_pct"]); x["float_q"]=qbucket(x["float_signal_pct"]); x["ret1_q"]=qbucket(x["ret1_signal_pct"])
    x["c5"]=x["base_signal"]&(x["turn_q"]<=2)&(x["float_q"]<=2)&(x["ret1_q"]<5)
    return x

def enrich(model,short,lag):
    x=model.sort_values(["ticker","date"]).copy()
    s=short.sort_values(["ticker","date"]).copy()
    if "free_float_shares" not in x.columns:
        x["free_float_shares"]=x["free_float_market_cap"]/x["close"].replace(0,np.nan)
    x["adv20_lag1"]=x.groupby("ticker")["volume"].transform(lambda z:z.shift(1).rolling(20,min_periods=10).mean())
    s["reported_short_balance_shares"]=s.groupby("ticker")["short_balance_shares"].shift(lag)
    s["reported_short_balance_lag5"]=s.groupby("ticker")["reported_short_balance_shares"].shift(5)
    s["reported_short_balance_delta5_shares"]=s["reported_short_balance_shares"]-s["reported_short_balance_lag5"]
    keep=["ticker","date","reported_short_balance_shares","reported_short_balance_delta5_shares"]
    x=x.merge(s[keep],on=["ticker","date"],how="left")
    x["has_positive_short_balance"]=x["reported_short_balance_shares"].fillna(0)>0
    x["short_interest_to_float"]=x["reported_short_balance_shares"]/x["free_float_shares"].replace(0,np.nan)
    x["true_dtc"]=x["reported_short_balance_shares"]/x["adv20_lag1"].replace(0,np.nan)
    x["short_build_5d_to_float"]=x["reported_short_balance_delta5_shares"]/x["free_float_shares"].replace(0,np.nan)
    return x

def make_events(x,hold,tp,sl):
    rows=[]
    for ticker,g in x.groupby("ticker",sort=False):
        g=g.sort_values("date").reset_index(drop=True); n=len(g)
        for pos in np.flatnonzero(g["c5"].fillna(False).to_numpy()):
            pos=int(pos); ep=pos+1
            if ep+hold-1>=n: continue
            sig=g.iloc[pos]; ent=g.iloc[ep]; entry=float(ent["open"])
            if not np.isfinite(entry) or entry<=0: continue
            w=g.iloc[ep:ep+hold]; tp_px=entry*(1+tp); sl_px=entry*(1+sl)
            reason="TIME"; exit_px=float(w.iloc[-1]["close"])
            for r in w.itertuples(index=False):
                o,h,l=float(r.open),float(r.high),float(r.low)
                if o<=sl_px: reason,exit_px="SL",o; break
                if l<=sl_px: reason,exit_px="SL",sl_px; break
                if h>=tp_px: reason,exit_px="TP",tp_px; break
            mh=float(w["high"].max()); sc=float(sig["close"]); mr=mh/sc-1 if sc>0 else np.nan
            rows.append({
                "ticker":ticker,"signal_date":pd.Timestamp(sig["date"]),"year":pd.Timestamp(sig["date"]).year,
                "trade_return_gross":exit_px/entry-1,"tp_before_sl":int(reason=="TP"),
                "target30":int(mr>=.30),"target50":int(mr>=.50),"target100":int(mr>=1.0),
                "max_signal_return_20d":mr,
                "reported_short_balance_shares":sig.get("reported_short_balance_shares",np.nan),
                "has_positive_short_balance":bool(sig.get("has_positive_short_balance",False)),
                "short_interest_to_float":sig.get("short_interest_to_float",np.nan),
                "true_dtc":sig.get("true_dtc",np.nan),
                "short_build_5d_to_float":sig.get("short_build_5d_to_float",np.nan),
            })
    return pd.DataFrame(rows)

def pf(r):
    pos=r[r>0].sum(); neg=-r[r<0].sum()
    return pos/neg if neg>0 else np.nan

def stat(g):
    if len(g)==0: return {"n":0,"p30":np.nan,"p50":np.nan,"p100":np.nan,"tp":np.nan,"avg_return":np.nan,"median_return":np.nan,"pf":np.nan,"avg_max20":np.nan}
    return {"n":len(g),"p30":g.target30.mean(),"p50":g.target50.mean(),"p100":g.target100.mean(),"tp":g.tp_before_sl.mean(),"avg_return":g.trade_return_gross.mean(),"median_return":g.trade_return_gross.median(),"pf":pf(g.trade_return_gross),"avg_max20":g.max_signal_return_20d.mean()}

def threshold_2025(ev,col,q=.80):
    g=ev[(ev.year==2025)&ev.has_positive_short_balance&ev[col].notna()]
    return float(g[col].quantile(q)) if not g.empty else np.nan

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    m=pd.read_parquet(a.model); m["ticker"]=m["ticker"].astype(str).str.zfill(6); m["date"]=pd.to_datetime(m["date"]); m=add_c5(m)
    s=pd.read_parquet(a.short_balance); s["ticker"]=s["ticker"].astype(str).str.zfill(6); s["date"]=pd.to_datetime(s["date"])
    ev=make_events(enrich(m,s,a.publication_lag_bars),a.max_hold,a.tp,a.sl)
    thr={"short_interest_to_float_p80_2025":threshold_2025(ev,"short_interest_to_float"),"true_dtc_p80_2025":threshold_2025(ev,"true_dtc"),"short_build_5d_to_float_p80_2025":threshold_2025(ev,"short_build_5d_to_float")}
    si,dtc,bld=thr.values()
    ev["high_short_interest"]=ev.has_positive_short_balance&(ev.short_interest_to_float>=si)
    ev["high_true_dtc"]=ev.has_positive_short_balance&(ev.true_dtc>=dtc)
    ev["high_high"]=ev.high_short_interest&ev.high_true_dtc
    ev["triple_high"]=ev.high_high&(ev.short_build_5d_to_float>=bld)
    groups={
        "baseline_all_C5":lambda y:y,
        "zero_reported_balance":lambda y:y[~y.has_positive_short_balance],
        "positive_reported_balance":lambda y:y[y.has_positive_short_balance],
        "high_short_interest_2025p80":lambda y:y[y.high_short_interest],
        "high_true_dtc_2025p80":lambda y:y[y.high_true_dtc],
        "high_short_and_high_dtc":lambda y:y[y.high_high],
        "triple_high_with_5d_build":lambda y:y[y.triple_high],
    }
    rows=[]
    for yr in [2025,2026]:
        y=ev[ev.year==yr]
        for name,fn in groups.items(): rows.append({"year":yr,"group":name,**stat(fn(y))})
    summary=pd.DataFrame(rows)
    qrows=[]
    for col in ["short_interest_to_float","true_dtc","short_build_5d_to_float"]:
        tr=ev[(ev.year==2025)&ev.has_positive_short_balance&ev[col].notna()]
        if tr.empty: continue
        cuts=np.unique(tr[col].quantile([0,.2,.4,.6,.8,1]).to_numpy(float))
        if len(cuts)<3: continue
        cuts[0],cuts[-1]=-np.inf,np.inf
        for yr in [2025,2026]:
            yy=ev[(ev.year==yr)&ev.has_positive_short_balance&ev[col].notna()].copy()
            yy["bucket"]=pd.cut(yy[col],bins=cuts,labels=False,include_lowest=True)+1
            for b,gg in yy.groupby("bucket"): qrows.append({"feature":col,"year":yr,"bucket_from_2025_cutoffs":int(b),**stat(gg)})
    qdiag=pd.DataFrame(qrows)
    coverage=ev.groupby("year").agg(events=("ticker","size"),positive_balance_events=("has_positive_short_balance","sum"),positive_balance_rate=("has_positive_short_balance","mean"),true_dtc_coverage=("true_dtc",lambda z:z.notna().mean())).reset_index()
    ev.to_parquet(out/"true_dtc_zero_aware_events.parquet",index=False)
    summary.to_csv(out/"true_dtc_zero_aware_summary.csv",index=False)
    qdiag.to_csv(out/"true_dtc_positive_only_buckets.csv",index=False)
    coverage.to_csv(out/"true_dtc_zero_aware_coverage.csv",index=False)
    (out/"thresholds_2025.json").write_text(json.dumps(thr,indent=2),encoding="utf-8")
    print("\\n=== TRUE DTC ZERO-AWARE COVERAGE ==="); print(coverage.to_string(index=False))
    print("\\n=== FROZEN 2025 P80 THRESHOLDS ==="); print(json.dumps(thr,indent=2))
    print("\\n=== TRUE DTC ZERO-AWARE FINAL SUMMARY ==="); print(summary.to_string(index=False))
    print("\\n=== POSITIVE-BALANCE ONLY BUCKET DIAGNOSTIC ==="); print(qdiag.to_string(index=False))
    print(f"\\nwrote -> {out}")
if __name__=="__main__": main()
