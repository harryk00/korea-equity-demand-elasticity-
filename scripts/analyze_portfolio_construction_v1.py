#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_EXIT = (0.30, -0.15, 20)
ALT_EXIT  = (0.40, -0.10, 30)
MAX_POSITIONS = [3, 5, 10]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--model", default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--market-state", default="data/pit_kosdaq/analysis/market_state_v1/market_state_daily.parquet")
    p.add_argument("--short-balance", default="data/pit_kosdaq/true_short_balance/true_short_balance_daily.parquet")
    p.add_argument("--minute-bars", default="data/pit_kosdaq/entry_timing/entry_timing_minute_bars.parquet")
    p.add_argument("--out-dir", default="data/pit_kosdaq/analysis/portfolio_construction_v1")
    p.add_argument("--commission-bps", type=float, default=7.5)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--publication-lag-bars", type=int, default=2)
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
    # smaller is preferred: smaller float and lower/less-overheated turnover within C5
    x["priority_score"]=x["float_signal_pct"].fillna(1)+x["turn_signal_pct"].fillna(1)
    return x

def tax_bps(year): return 15.0 if int(year)==2025 else 20.0

def simulate_trade(g,pos,entry_raw,entry_time,minute_day,tp,sl,hold,comm_bps,slip_bps):
    ep=pos+1
    if ep+hold-1>=len(g) or not np.isfinite(entry_raw) or entry_raw<=0: return None
    slip=slip_bps/10000
    buy_fee=comm_bps/10000
    entry_fill=entry_raw*(1+slip)
    entry_cash_ps=entry_fill*(1+buy_fee)
    tp_px=entry_fill*(1+tp); sl_px=entry_fill*(1+sl)
    exit_raw=reason=exit_date=exit_bar=None

    # entry day
    if entry_time=="OPEN":
        d=g.iloc[ep]
        o,h,l=float(d.open),float(d.high),float(d.low)
        if o<=sl_px: exit_raw,reason,exit_date,exit_bar=o,"SL_GAP",pd.Timestamp(d.date),1
        elif l<=sl_px: exit_raw,reason,exit_date,exit_bar=sl_px,"SL",pd.Timestamp(d.date),1
        elif h>=tp_px: exit_raw,reason,exit_date,exit_bar=tp_px,"TP",pd.Timestamp(d.date),1
    else:
        mm=minute_day[minute_day["time"]>=entry_time].sort_values("datetime") if minute_day is not None else pd.DataFrame()
        for r in mm.itertuples(index=False):
            o,h,l=float(r.open),float(r.high),float(r.low)
            if o<=sl_px: exit_raw,reason,exit_date,exit_bar=o,"SL_GAP",pd.Timestamp(r.date),1; break
            if l<=sl_px: exit_raw,reason,exit_date,exit_bar=sl_px,"SL",pd.Timestamp(r.date),1; break
            if h>=tp_px: exit_raw,reason,exit_date,exit_bar=tp_px,"TP",pd.Timestamp(r.date),1; break

    start_j=ep+1
    if exit_raw is None:
        for j in range(start_j,ep+hold):
            d=g.iloc[j]; o,h,l=float(d.open),float(d.high),float(d.low)
            if o<=sl_px: exit_raw,reason,exit_date,exit_bar=o,"SL_GAP",pd.Timestamp(d.date),j-ep+1; break
            if l<=sl_px: exit_raw,reason,exit_date,exit_bar=sl_px,"SL",pd.Timestamp(d.date),j-ep+1; break
            if h>=tp_px: exit_raw,reason,exit_date,exit_bar=tp_px,"TP",pd.Timestamp(d.date),j-ep+1; break
    if exit_raw is None:
        d=g.iloc[ep+hold-1]
        exit_raw,reason,exit_date,exit_bar=float(d.close),"TIME",pd.Timestamp(d.date),hold

    sell_cost=(comm_bps+slip_bps+tax_bps(exit_date.year))/10000
    exit_cash_ps=exit_raw*(1-sell_cost)
    return {
        "entry_date":pd.Timestamp(g.iloc[ep].date),
        "exit_date":exit_date,
        "entry_cash_ps":entry_cash_ps,
        "exit_cash_ps":exit_cash_ps,
        "net_return":exit_cash_ps/entry_cash_ps-1,
        "exit_reason":reason,
        "exit_bar":exit_bar,
    }

def add_market_and_short(x,state,short,lag):
    # Market thresholds estimated only on 2025 C5 events.
    ev=x.loc[x.c5,["date"]].drop_duplicates().merge(state,on="date",how="left")
    tr=ev[ev.date.dt.year==2025]
    thr={
        "kq_ret20_q40_2025":float(tr.kq_ret20.quantile(.40)),
        "breadth_above_ma20_q40_2025":float(tr.breadth_above_ma20.quantile(.40)),
        "kq_drawdown60_q40_2025":float(tr.kq_drawdown60.quantile(.40)),
    }
    state=state.copy()
    state["weak_trend"]=state.kq_ret20<=thr["kq_ret20_q40_2025"]
    state["weak_breadth"]=state.breadth_above_ma20<=thr["breadth_above_ma20_q40_2025"]
    state["deep_drawdown"]=state.kq_drawdown60<=thr["kq_drawdown60_q40_2025"]
    state["market_weak_score"]=state[["weak_trend","weak_breadth","deep_drawdown"]].fillna(False).astype(int).sum(axis=1)
    state["market_weak2"]=state.market_weak_score>=2
    x=x.merge(state[["date","weak_trend","weak_breadth","deep_drawdown","market_weak_score","market_weak2"]],on="date",how="left")

    s=short.sort_values(["ticker","date"]).copy()
    s["reported_short_balance_shares"]=s.groupby("ticker")["short_balance_shares"].shift(lag)
    x=x.merge(s[["ticker","date","reported_short_balance_shares"]],on=["ticker","date"],how="left")
    x["short_zero"]=x.reported_short_balance_shares.eq(0)
    return x,thr

def minute_snapshot(mm,daily_open):
    m=mm.sort_values("datetime")
    pre=m[m.time<"090500"]; aft=m[m.time>="090500"]
    if aft.empty: return None
    px=float(aft.iloc[0].open); tm=str(aft.iloc[0].time)
    r5=float(pre.iloc[-1].close)/daily_open-1 if len(pre) and daily_open>0 else np.nan
    return px,tm,r5

def build_trade_tables(model,state,short,mins,a):
    x,thr=add_market_and_short(model,state,short,a.publication_lag_bars)
    groups={t:g.sort_values("date").reset_index(drop=True) for t,g in x.groupby("ticker",sort=False)}
    minute_groups={(t,pd.Timestamp(d)):g for (t,d),g in mins.groupby(["ticker","date"],sort=False)} if mins is not None else {}

    full=[]
    recent=[]
    for ticker,g in groups.items():
        for pos in np.flatnonzero(g.c5.fillna(False).to_numpy()):
            pos=int(pos)
            if pos+1>=len(g): continue
            sig=g.iloc[pos]; ent=g.iloc[pos+1]
            sd=pd.Timestamp(sig.date); ed=pd.Timestamp(ent.date)
            common={
                "ticker":ticker,"signal_date":sd,"entry_date":ed,
                "priority_score":float(sig.priority_score),
                "market_weak2":bool(sig.market_weak2) if pd.notna(sig.market_weak2) else False,
                "market_weak_score":int(sig.market_weak_score) if pd.notna(sig.market_weak_score) else 0,
                "short_zero":bool(sig.short_zero) if pd.notna(sig.short_zero) else False,
            }
            for exit_name,(tp,sl,hold) in [("BASE_EXIT",BASE_EXIT),("ALT_EXIT",ALT_EXIT)]:
                sim=simulate_trade(g,pos,float(ent.open),"OPEN",None,tp,sl,hold,a.commission_bps,a.slippage_bps)
                if sim is not None: full.append({**common,"entry_rule":"OPEN","exit_rule":exit_name,**sim})

            mm=minute_groups.get((ticker,ed))
            if mm is not None and not mm.empty:
                snap=minute_snapshot(mm,float(ent.open))
                if snap is not None:
                    px,tm,r5=snap
                    gap=float(ent.open)/float(sig.close)-1 if float(sig.close)>0 else np.nan
                    no_chase=(gap<=.10 and np.isfinite(r5) and r5<=.03)
                    if no_chase:
                        for exit_name,(tp,sl,hold) in [("BASE_EXIT",BASE_EXIT),("ALT_EXIT",ALT_EXIT)]:
                            sim=simulate_trade(g,pos,px,tm,mm,tp,sl,hold,a.commission_bps,a.slippage_bps)
                            if sim is not None:
                                recent.append({**common,"entry_rule":"0905_NOCHASE","exit_rule":exit_name,"gap":gap,"first5_ret":r5,**sim})
    return pd.DataFrame(full),pd.DataFrame(recent),thr

def variant_filter(df,name):
    z=df
    if "MARKET" in name: z=z[z.market_weak2]
    if "SHORT0" in name: z=z[z.short_zero]
    if "ALTEXIT" in name: z=z[z.exit_rule=="ALT_EXIT"]
    else: z=z[z.exit_rule=="BASE_EXIT"]
    return z.copy()

def portfolio_sim(trades,model,max_pos):
    if trades.empty: return pd.DataFrame(),pd.DataFrame()
    trades=trades.sort_values(["entry_date","priority_score","ticker"]).copy()
    by_entry={d:g.sort_values(["priority_score","ticker"]) for d,g in trades.groupby("entry_date")}
    closes={(str(t),pd.Timestamp(d)):float(c) for t,d,c in model[["ticker","date","close"]].itertuples(index=False,name=None)}
    dates=sorted(pd.to_datetime(model.date.unique()))
    min_d,max_d=trades.entry_date.min(),trades.exit_date.max()
    dates=[d for d in dates if d>=min_d and d<=max_d]

    cash=1.0; positions={}; curve=[]; executed=[]; prev_equity=1.0
    for d in dates:
        # Entries first: conservative, same-day exit proceeds are not reused.
        candidates=by_entry.get(pd.Timestamp(d))
        slots=max_pos-len(positions)
        if candidates is not None and slots>0 and cash>1e-12:
            target=prev_equity/max_pos
            for r in candidates.itertuples(index=False):
                if slots<=0 or cash<=1e-12: break
                if r.ticker in positions: continue
                alloc=min(target,cash)
                if alloc < max(target*0.25,1e-10): continue
                shares=alloc/float(r.entry_cash_ps)
                cash-=alloc
                positions[r.ticker]={
                    "shares":shares,"exit_date":pd.Timestamp(r.exit_date),
                    "exit_cash_ps":float(r.exit_cash_ps),"entry_date":pd.Timestamp(r.entry_date),
                    "entry_notional":alloc,"net_return":float(r.net_return),
                    "exit_reason":r.exit_reason,"signal_date":pd.Timestamp(r.signal_date),
                }
                slots-=1

        # Exits after entries; proceeds become available from next decision day.
        exits=[t for t,p in positions.items() if p["exit_date"]==pd.Timestamp(d)]
        for t in exits:
            p=positions.pop(t)
            proceeds=p["shares"]*p["exit_cash_ps"]
            cash+=proceeds
            executed.append({
                "ticker":t,"signal_date":p["signal_date"],"entry_date":p["entry_date"],
                "exit_date":pd.Timestamp(d),"entry_notional":p["entry_notional"],
                "proceeds":proceeds,"trade_return":proceeds/p["entry_notional"]-1,
                "exit_reason":p["exit_reason"],
            })

        mtm=0.0
        for t,p in positions.items():
            px=closes.get((str(t),pd.Timestamp(d)))
            if px is not None and np.isfinite(px): mtm+=p["shares"]*px
            else: mtm+=p["entry_notional"]
        equity=cash+mtm
        curve.append({"date":pd.Timestamp(d),"equity":equity,"cash":cash,"positions":len(positions)})
        prev_equity=equity
    return pd.DataFrame(curve),pd.DataFrame(executed)

def perf(curve,trades,period):
    if curve.empty: return {}
    c=curve.sort_values("date").copy()
    if period=="2025":
        z=c[c.date.dt.year==2025].copy()
    elif period=="2026":
        z=c[c.date.dt.year==2026].copy()
    else:
        z=c.copy()
    if z.empty: return {}
    first_idx=z.index[0]
    start_equity=float(c.loc[first_idx-1,"equity"]) if first_idx>c.index.min() else 1.0
    end_equity=float(z.iloc[-1].equity)
    days=max((z.iloc[-1].date-z.iloc[0].date).days,1)
    total=end_equity/start_equity-1
    ann=(1+total)**(365/days)-1 if total>-1 else -1
    eq=pd.concat([pd.Series([start_equity]),z.equity.reset_index(drop=True)],ignore_index=True)
    dd=eq/eq.cummax()-1
    rets=eq.pct_change().dropna()
    sharpe=np.sqrt(252)*rets.mean()/rets.std() if rets.std()>0 else np.nan
    tt=trades.copy()
    if not tt.empty:
        if period=="2025": tt=tt[tt.entry_date.dt.year==2025]
        elif period=="2026": tt=tt[tt.entry_date.dt.year==2026]
    if tt.empty:
        pf=np.nan; avg_trade=np.nan; win=np.nan; n=0; turnover=0
    else:
        pos=tt.loc[tt.trade_return>0,"trade_return"].sum()
        neg=-tt.loc[tt.trade_return<0,"trade_return"].sum()
        pf=pos/neg if neg>0 else np.nan
        avg_trade=tt.trade_return.mean(); win=(tt.trade_return>0).mean(); n=len(tt)
        turnover=tt.entry_notional.sum()/start_equity
    return {
        "period":period,"total_return":total,"annualized_return":ann,"mdd":dd.min(),
        "sharpe":sharpe,"trades":n,"trade_pf":pf,"avg_trade_return":avg_trade,
        "win_rate":win,"capital_turnover":turnover,
        "avg_positions":z.positions.mean(),"avg_cash_pct":(z.cash/z.equity).mean(),
    }

def run_variants(trade_table,model,variant_names,label):
    rows=[]; curves={}
    for v in variant_names:
        z=variant_filter(trade_table,v)
        for npos in MAX_POSITIONS:
            curve,ex=portfolio_sim(z,model,npos)
            key=f"{label}|{v}|N{npos}"
            curves[key]=(curve,ex)
            for period in ["2025","2026","full"]:
                p=perf(curve,ex,period)
                if p: rows.append({"set":label,"variant":v,"max_positions":npos,**p})
    return pd.DataFrame(rows),curves

def main():
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cols=["ticker","date","open","high","low","close","free_float_market_cap","float_turnover_1d"]
    model=pd.read_parquet(a.model,columns=cols)
    model["ticker"]=model.ticker.astype(str).str.zfill(6); model["date"]=pd.to_datetime(model.date)
    model=add_c5(model)
    state=pd.read_parquet(a.market_state); state["date"]=pd.to_datetime(state.date)
    short=pd.read_parquet(a.short_balance); short["ticker"]=short.ticker.astype(str).str.zfill(6); short["date"]=pd.to_datetime(short.date)
    mins=pd.read_parquet(a.minute_bars); mins["ticker"]=mins.ticker.astype(str).str.zfill(6); mins["date"]=pd.to_datetime(mins.date); mins["datetime"]=pd.to_datetime(mins.datetime)

    full,recent_nc,thr=build_trade_tables(model,state,short,mins,a)
    full.to_parquet(out/"portfolio_candidate_trades_full.parquet",index=False)
    recent_nc.to_parquet(out/"portfolio_candidate_trades_0905_nochase.parquet",index=False)

    full_variants=[
        "BASE",
        "MARKET",
        "SHORT0",
        "MARKET_SHORT0",
        "ALTEXIT",
        "MARKET_ALTEXIT",
        "SHORT0_ALTEXIT",
        "MARKET_SHORT0_ALTEXIT",
    ]
    full_summary,_=run_variants(full,model,full_variants,"FULL_OPEN")

    # Recent common-window set: restrict OPEN trades to dates where minute collector could be used.
    if recent_nc.empty:
        recent_summary=pd.DataFrame()
    else:
        d0=recent_nc.entry_date.min(); d1=recent_nc.entry_date.max()
        open_recent=full[(full.entry_date>=d0)&(full.entry_date<=d1)].copy()
        open_recent["entry_mode"]="OPEN"
        nc=recent_nc.copy(); nc["entry_mode"]="0905_NOCHASE"

        recent_rows=[]
        for mode,tab in [("OPEN",open_recent),("0905_NOCHASE",nc)]:
            variants=["BASE","MARKET_SHORT0","ALTEXIT","MARKET_SHORT0_ALTEXIT"]
            sm,_=run_variants(tab,model,variants,f"RECENT_{mode}")
            recent_rows.append(sm)
        recent_summary=pd.concat(recent_rows,ignore_index=True)

    full_summary.to_csv(out/"portfolio_full_summary.csv",index=False)
    recent_summary.to_csv(out/"portfolio_recent_entry_summary.csv",index=False)
    (out/"portfolio_market_thresholds_2025.json").write_text(json.dumps(thr,indent=2),encoding="utf-8")

    print("\n=== PORTFOLIO MARKET THRESHOLDS (2025 C5 ONLY) ===")
    print(json.dumps(thr,indent=2))

    print("\n=== FULL-HISTORY PORTFOLIO SUMMARY ===")
    show=full_summary.sort_values(["period","max_positions","annualized_return"],ascending=[True,True,False])
    print(show.to_string(index=False))

    print("\n=== 2025 TOP BY MAX POSITIONS (FULL-HISTORY VARIANTS) ===")
    s25=full_summary[full_summary.period=="2025"].copy()
    for n in MAX_POSITIONS:
        z=s25[s25.max_positions==n].sort_values(["annualized_return","sharpe"],ascending=False).head(5)
        print(f"\n-- N={n} --")
        print(z.to_string(index=False))

    print("\n=== 2026 FROZEN CHECK FOR 2025 TOP-RETURN VARIANT BY N ===")
    checks=[]
    for n in MAX_POSITIONS:
        z=s25[s25.max_positions==n].sort_values(["annualized_return","sharpe"],ascending=False)
        if z.empty: continue
        v=str(z.iloc[0].variant)
        q=full_summary[(full_summary.period=="2026")&(full_summary.max_positions==n)&(full_summary.variant==v)]
        if not q.empty: checks.append(q.iloc[0].to_dict())
    print(pd.DataFrame(checks).to_string(index=False) if checks else "None")

    print("\n=== RECENT ENTRY-TIMING PORTFOLIO SUMMARY ===")
    print(recent_summary.sort_values(["period","max_positions","annualized_return"],ascending=[True,True,False]).to_string(index=False) if not recent_summary.empty else "None")

    print("\nNOTES")
    print("- MARKET = 2+ of: weak 20d KOSDAQ return, weak breadth, deep 60d drawdown; thresholds frozen from 2025 C5 events.")
    print("- SHORT0 = KRX reported short balance after 2-bar publication lag equals zero.")
    print("- ALTEXIT = TP40 / SL10 / Hold30 challenger.")
    print("- 0905_NOCHASE = gap <= +10% and first 5m return <= +3%; otherwise skip.")
    print("- Position sleeve = prior-close equity / max_positions; unused sleeves stay cash.")
    print("- Same ticker cannot overlap; entries are processed before same-day exits, so same-day exit cash is not reused.")
    print("\nwrote ->",out)

if __name__=="__main__":
    main()
