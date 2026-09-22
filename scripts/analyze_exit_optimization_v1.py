#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TP_LIST=[0.20,0.30,0.40,0.50]
SL_LIST=[-0.10,-0.15,-0.20]
HOLD_LIST=[10,20,30]
BASELINE=(0.30,-0.15,20)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model",default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--out-dir",default="data/pit_kosdaq/analysis/exit_optimization_v1")
    p.add_argument("--commission-bps",type=float,default=7.5)
    p.add_argument("--slippage-bps",type=float,default=10.0)
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
    x["turn_q"]=qbucket(x["turn_signal_pct"])
    x["float_q"]=qbucket(x["float_signal_pct"])
    x["ret1_q"]=qbucket(x["ret1_signal_pct"])
    x["c5"]=x["base_signal"]&(x["turn_q"]<=2)&(x["float_q"]<=2)&(x["ret1_q"]<5)
    return x

def tax_bps(year): return 15.0 if int(year)==2025 else 20.0
def combo_id(tp,sl,hold): return f"TP{int(tp*100)}_SL{int(abs(sl)*100)}_H{hold}"

def simulate(g,pos,tp,sl,hold,comm_bps,slip_bps):
    ep=pos+1
    if ep+30-1>=len(g): return None
    entry_raw=float(g.iloc[ep]["open"])
    if not np.isfinite(entry_raw) or entry_raw<=0: return None
    slip=slip_bps/10000
    buy_fee=comm_bps/10000
    entry_fill=entry_raw*(1+slip)
    entry_cash=entry_fill*(1+buy_fee)
    tp_px=entry_fill*(1+tp)
    sl_px=entry_fill*(1+sl)
    exit_raw=reason=exit_date=exit_bar=None
    for j in range(ep,ep+hold):
        d=g.iloc[j]
        o,h,l=float(d["open"]),float(d["high"]),float(d["low"])
        if o<=sl_px:
            exit_raw,reason,exit_date,exit_bar=o,"SL_GAP",pd.Timestamp(d["date"]),j-ep+1; break
        if l<=sl_px:
            exit_raw,reason,exit_date,exit_bar=sl_px,"SL",pd.Timestamp(d["date"]),j-ep+1; break
        if h>=tp_px:
            exit_raw,reason,exit_date,exit_bar=tp_px,"TP",pd.Timestamp(d["date"]),j-ep+1; break
    if exit_raw is None:
        d=g.iloc[ep+hold-1]
        exit_raw,reason,exit_date,exit_bar=float(d["close"]),"TIME",pd.Timestamp(d["date"]),hold
    sell_cost=(comm_bps+slip_bps+tax_bps(exit_date.year))/10000
    exit_cash=exit_raw*(1-sell_cost)
    return {
        "entry_date":pd.Timestamp(g.iloc[ep]["date"]),
        "exit_date":exit_date,
        "net_return":exit_cash/entry_cash-1,
        "gross_return":exit_raw/entry_raw-1,
        "exit_reason":reason,
        "exit_bar":exit_bar,
    }

def pf(s):
    p=s[s>0].sum(); n=-s[s<0].sum()
    return p/n if n>0 else np.nan

def stats(g):
    return {
        "n":len(g),
        "avg_net":g.net_return.mean(),
        "median_net":g.net_return.median(),
        "pf":pf(g.net_return),
        "positive_rate":(g.net_return>0).mean(),
        "tp_rate":g.exit_reason.eq("TP").mean(),
        "sl_rate":g.exit_reason.str.startswith("SL").mean(),
        "time_rate":g.exit_reason.eq("TIME").mean(),
        "p10":g.net_return.quantile(.10),
        "p25":g.net_return.quantile(.25),
        "p75":g.net_return.quantile(.75),
        "p90":g.net_return.quantile(.90),
        "avg_exit_bar":g.exit_bar.mean(),
    }

def main():
    a=parse_args()
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cols=["ticker","date","open","high","low","close","free_float_market_cap","float_turnover_1d"]
    m=pd.read_parquet(a.model,columns=cols)
    m["ticker"]=m.ticker.astype(str).str.zfill(6); m["date"]=pd.to_datetime(m.date)
    m=add_c5(m)

    rec=[]; elig=[]
    for ticker,g in m.groupby("ticker",sort=False):
        g=g.sort_values("date").reset_index(drop=True)
        for pos in np.flatnonzero(g.c5.fillna(False).to_numpy()):
            pos=int(pos)
            if simulate(g,pos,*BASELINE[:2],30,a.commission_bps,a.slippage_bps) is None: continue
            sd=pd.Timestamp(g.iloc[pos]["date"])
            elig.append({"ticker":ticker,"signal_date":sd,"year":sd.year})
            for tp in TP_LIST:
                for sl in SL_LIST:
                    for hold in HOLD_LIST:
                        s=simulate(g,pos,tp,sl,hold,a.commission_bps,a.slippage_bps)
                        if s is not None:
                            rec.append({"ticker":ticker,"signal_date":sd,"year":sd.year,"tp":tp,"sl":sl,"hold":hold,"combo":combo_id(tp,sl,hold),**s})
    trades=pd.DataFrame(rec); eligible=pd.DataFrame(elig)
    if trades.empty: raise SystemExit("No exit optimization trades generated.")
    trades.to_parquet(out/"exit_grid_trades.parquet",index=False)
    eligible.to_csv(out/"exit_grid_common_eligible_events.csv",index=False)

    rows=[]
    for yr in [2025,2026]:
        y=trades[trades.year==yr]
        for (tp,sl,hold,combo),g in y.groupby(["tp","sl","hold","combo"]):
            rows.append({"year":yr,"tp":tp,"sl":sl,"hold":int(hold),"combo":combo,**stats(g)})
    summ=pd.DataFrame(rows)
    base_combo=combo_id(*BASELINE)
    b=summ[summ.combo==base_combo][["year","avg_net","median_net","pf","positive_rate","p10"]].rename(columns={
        "avg_net":"base_avg_net","median_net":"base_median_net","pf":"base_pf","positive_rate":"base_positive_rate","p10":"base_p10"})
    summ=summ.merge(b,on="year",how="left")
    summ["delta_avg_vs_base"]=summ.avg_net-summ.base_avg_net
    summ["delta_pf_vs_base"]=summ.pf-summ.base_pf
    summ["delta_positive_vs_base"]=summ.positive_rate-summ.base_positive_rate
    summ["delta_p10_vs_base"]=summ.p10-summ.base_p10

    s25=summ[summ.year==2025].copy()
    b25=s25[s25.combo==base_combo].iloc[0]
    cand=s25[(s25.avg_net>=b25.avg_net)&(s25.pf>=b25.pf)].copy()
    if cand.empty: cand=s25.copy()
    cand["rank_avg"]=cand.avg_net.rank(ascending=False,method="min")
    cand["rank_pf"]=cand.pf.rank(ascending=False,method="min")
    cand["rank_p10"]=cand.p10.rank(ascending=False,method="min")
    cand["selection_score"]=cand.rank_avg+0.5*cand.rank_pf+0.25*cand.rank_p10
    selected=str(cand.sort_values(["selection_score","avg_net","pf"],ascending=[True,False,False]).iloc[0].combo)

    champion=summ[summ.combo.isin([base_combo,selected])].copy()
    champion["role"]=np.where(champion.combo==base_combo,"BASELINE","2025_SELECTED")

    base_tr=trades[trades.combo==base_combo][["ticker","signal_date","year","net_return"]].rename(columns={"net_return":"base_net"})
    pr=[]
    for combo,g in trades.groupby("combo"):
        z=g.merge(base_tr,on=["ticker","signal_date","year"],how="inner")
        for yr in [2025,2026]:
            yy=z[z.year==yr]
            if yy.empty: continue
            d=yy.net_return-yy.base_net
            pr.append({"year":yr,"combo":combo,"paired_n":len(yy),"avg_combo_net":yy.net_return.mean(),"avg_base_same_events":yy.base_net.mean(),
                       "avg_delta_vs_base":d.mean(),"median_delta_vs_base":d.median(),"win_vs_base_rate":(d>0).mean(),"lose_vs_base_rate":(d<0).mean()})
    paired=pd.DataFrame(pr)

    b25=summ[(summ.year==2025)&(summ.combo==base_combo)].iloc[0]
    b26=summ[(summ.year==2026)&(summ.combo==base_combo)].iloc[0]
    rr=[]
    for combo in summ.combo.unique():
        r25=summ[(summ.year==2025)&(summ.combo==combo)].iloc[0]
        r26=summ[(summ.year==2026)&(summ.combo==combo)].iloc[0]
        rr.append({"combo":combo,"tp":r25.tp,"sl":r25.sl,"hold":int(r25.hold),
                   "avg_2025":r25.avg_net,"pf_2025":r25.pf,"avg_2026":r26.avg_net,"pf_2026":r26.pf,
                   "beats_avg_and_pf_both_years":bool(r25.avg_net>b25.avg_net and r25.pf>b25.pf and r26.avg_net>b26.avg_net and r26.pf>b26.pf)})
    robust=pd.DataFrame(rr).sort_values(["beats_avg_and_pf_both_years","avg_2026","avg_2025"],ascending=[False,False,False])

    summ.to_csv(out/"exit_grid_summary.csv",index=False)
    cand.sort_values("selection_score").to_csv(out/"exit_grid_2025_selection_ranking.csv",index=False)
    champion.to_csv(out/"exit_grid_frozen_champion_vs_baseline.csv",index=False)
    paired.to_csv(out/"exit_grid_paired_vs_baseline.csv",index=False)
    robust.to_csv(out/"exit_grid_cross_year_robustness.csv",index=False)
    meta={"grid_tp":TP_LIST,"grid_sl":SL_LIST,"grid_hold":HOLD_LIST,"grid_combinations":36,"baseline_combo":base_combo,
          "selected_on_2025_combo":selected,"commission_bps_per_side":a.commission_bps,"slippage_bps_per_side":a.slippage_bps,
          "sell_tax_bps_2025":15.0,"sell_tax_bps_2026":20.0,"common_sample_requirement":"30 forward trading bars"}
    (out/"exit_grid_meta.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")

    print("\n=== EXIT OPTIMIZATION COVERAGE ===")
    print("common_eligible_events",len(eligible))
    print("2025_events",int((eligible.year==2025).sum()))
    print("2026_events",int((eligible.year==2026).sum()))
    print("grid_combinations",36)
    print("trade_rows",len(trades))
    print("\n=== BASELINE ===")
    print(summ[summ.combo==base_combo][["year","combo","n","avg_net","median_net","pf","positive_rate","tp_rate","sl_rate","time_rate","p10","avg_exit_bar"]].to_string(index=False))
    print("\n=== 2025 TOP 12 EXIT COMBINATIONS ===")
    print(cand.sort_values("selection_score").head(12)[["combo","n","avg_net","median_net","pf","positive_rate","tp_rate","sl_rate","time_rate","p10","avg_exit_bar","selection_score"]].to_string(index=False))
    print("\n=== 2025 SELECTED FROZEN CHAMPION VS BASELINE ===")
    print(champion[["year","role","combo","n","avg_net","median_net","pf","positive_rate","tp_rate","sl_rate","time_rate","p10","avg_exit_bar","delta_avg_vs_base","delta_pf_vs_base"]].sort_values(["year","role"]).to_string(index=False))
    print("\n=== SELECTED CHAMPION PAIRED VS BASELINE ===")
    print(paired[paired.combo==selected].to_string(index=False))
    print("\n=== CROSS-YEAR ROBUST CANDIDATES ===")
    z=robust[robust.beats_avg_and_pf_both_years].head(15)
    print("None" if z.empty else z.to_string(index=False))
    print("\nselected_on_2025 =",selected)
    print("baseline =",base_combo)
    print("wrote ->",out)

if __name__=="__main__":
    main()
