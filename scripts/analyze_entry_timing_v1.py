#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet'); p.add_argument('--candidates',default='data/pit_kosdaq/entry_timing/entry_timing_candidates.parquet'); p.add_argument('--minute-bars',default='data/pit_kosdaq/entry_timing/entry_timing_minute_bars.parquet'); p.add_argument('--out-dir',default='data/pit_kosdaq/analysis/entry_timing_v1'); p.add_argument('--tp',type=float,default=.30); p.add_argument('--sl',type=float,default=-.15); p.add_argument('--hold',type=int,default=20); p.add_argument('--commission-bps',type=float,default=7.5); p.add_argument('--slippage-bps',type=float,default=10.0); return p.parse_args()
def tax_bps(y): return 15. if int(y)==2025 else 20.
def snapshot(m,day_open):
    m=m.sort_values('datetime'); b5=m[m.time<'090500']; a5=m[m.time>='090500']; b10=m[m.time<'091000']; a10=m[m.time>='091000']
    return {'px_0905':float(a5.iloc[0].open) if len(a5) else np.nan,'time_0905':str(a5.iloc[0].time) if len(a5) else None,'px_0910':float(a10.iloc[0].open) if len(a10) else np.nan,'time_0910':str(a10.iloc[0].time) if len(a10) else None,'ret_first5':float(b5.iloc[-1].close)/day_open-1 if len(b5) and day_open>0 else np.nan,'ret_first10':float(b10.iloc[-1].close)/day_open-1 if len(b10) and day_open>0 else np.nan}
def simulate(g,entry_date,raw_entry,entry_time,mm,tp,sl,hold,comm,slip):
    g=g.sort_values('date').reset_index(drop=True); idx=np.flatnonzero((g.date==entry_date).to_numpy())
    if len(idx)==0 or not np.isfinite(raw_entry) or raw_entry<=0:return None
    ep=int(idx[0])
    if ep+hold-1>=len(g): return None
    entry_basis=raw_entry*(1+slip/10000); tp_px=entry_basis*(1+tp); sl_px=entry_basis*(1+sl); exit_raw=reason=exit_date=None
    if entry_time=='OPEN':
        d=g.iloc[ep]; o,h,l=map(float,[d.open,d.high,d.low])
        if o<=sl_px: exit_raw,reason,exit_date=o,'SL_GAP',entry_date
        elif l<=sl_px: exit_raw,reason,exit_date=sl_px,'SL',entry_date
        elif h>=tp_px: exit_raw,reason,exit_date=tp_px,'TP',entry_date
    else:
        z=mm[mm.time>=entry_time].sort_values('datetime')
        for r in z.itertuples(index=False):
            o,h,l=map(float,[r.open,r.high,r.low])
            if o<=sl_px: exit_raw,reason,exit_date=o,'SL_GAP',entry_date; break
            if l<=sl_px: exit_raw,reason,exit_date=sl_px,'SL',entry_date; break
            if h>=tp_px: exit_raw,reason,exit_date=tp_px,'TP',entry_date; break
    if exit_raw is None:
        for j in range(ep+1,ep+hold):
            d=g.iloc[j]; o,h,l=map(float,[d.open,d.high,d.low])
            if o<=sl_px: exit_raw,reason,exit_date=o,'SL_GAP',d.date; break
            if l<=sl_px: exit_raw,reason,exit_date=sl_px,'SL',d.date; break
            if h>=tp_px: exit_raw,reason,exit_date=tp_px,'TP',d.date; break
    if exit_raw is None:
        d=g.iloc[ep+hold-1]; exit_raw,reason,exit_date=float(d.close),'TIME',d.date
    buy=(comm+slip)/10000; sell=(comm+slip+tax_bps(pd.Timestamp(exit_date).year))/10000
    return {'net_return':exit_raw*(1-sell)/(raw_entry*(1+buy))-1,'gross_return':exit_raw/raw_entry-1,'exit_reason':reason,'exit_date':pd.Timestamp(exit_date)}
def pf(r):
    p=r[r>0].sum(); n=-r[r<0].sum(); return p/n if n>0 else np.nan
def summ(g):
    if len(g)==0:return {'n':0,'avg_net':np.nan,'median_net':np.nan,'pf':np.nan,'tp_rate':np.nan,'sl_rate':np.nan,'positive_rate':np.nan}
    return {'n':len(g),'avg_net':g.net_return.mean(),'median_net':g.net_return.median(),'pf':pf(g.net_return),'tp_rate':g.exit_reason.eq('TP').mean(),'sl_rate':g.exit_reason.str.startswith('SL').mean(),'positive_rate':(g.net_return>0).mean()}
def main():
    a=parse_args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    model=pd.read_parquet(a.model,columns=['ticker','date','open','high','low','close']); model['ticker']=model.ticker.astype(str).str.zfill(6); model['date']=pd.to_datetime(model.date)
    cand=pd.read_parquet(a.candidates); cand['ticker']=cand.ticker.astype(str).str.zfill(6); cand['signal_date']=pd.to_datetime(cand.signal_date); cand['entry_date']=pd.to_datetime(cand.entry_date)
    mins=pd.read_parquet(a.minute_bars); mins['ticker']=mins.ticker.astype(str).str.zfill(6); mins['date']=pd.to_datetime(mins.date); mins['datetime']=pd.to_datetime(mins.datetime)
    mg={(t,pd.Timestamp(d)):g for (t,d),g in mins.groupby(['ticker','date'],sort=False)}; dg={t:g for t,g in model.groupby('ticker',sort=False)}
    rows=[]; cov=[]
    for r in cand.itertuples(index=False):
        mm=mg.get((r.ticker,pd.Timestamp(r.entry_date)))
        if mm is None or mm.empty: continue
        s=snapshot(mm,float(r.entry_open_daily)); gap=float(r.entry_open_daily)/float(r.signal_close)-1 if float(r.signal_close)>0 else np.nan
        entries={'E0_OPEN':(float(r.entry_open_daily),'OPEN',True),'E1_OPEN_GAP_LE_10':(float(r.entry_open_daily),'OPEN',gap<=.10),'E2_OPEN_GAP_LE_5':(float(r.entry_open_daily),'OPEN',gap<=.05),'E3_0905_ALL':(s['px_0905'],s['time_0905'],np.isfinite(s['px_0905'])),'E4_0905_NO_CHASE':(s['px_0905'],s['time_0905'],np.isfinite(s['px_0905']) and gap<=.10 and np.isfinite(s['ret_first5']) and s['ret_first5']<=.03),'E5_0910_ALL':(s['px_0910'],s['time_0910'],np.isfinite(s['px_0910'])),'E6_0910_NO_CHASE':(s['px_0910'],s['time_0910'],np.isfinite(s['px_0910']) and gap<=.10 and np.isfinite(s['ret_first10']) and s['ret_first10']<=.05)}
        cov.append({'ticker':r.ticker,'signal_date':r.signal_date,'entry_date':r.entry_date,'gap':gap,**s})
        for rule,(px,tm,ok) in entries.items():
            if not ok or tm is None or not np.isfinite(px): continue
            z=simulate(dg[r.ticker],pd.Timestamp(r.entry_date),float(px),str(tm),mm,a.tp,a.sl,a.hold,a.commission_bps,a.slippage_bps)
            if z: rows.append({'rule':rule,'ticker':r.ticker,'signal_date':r.signal_date,'entry_date':r.entry_date,'gap':gap,'ret_first5':s['ret_first5'],'ret_first10':s['ret_first10'],'entry_price_raw':px,**z})
    tr=pd.DataFrame(rows); cv=pd.DataFrame(cov)
    if tr.empty: raise SystemExit('No entry timing trades produced.')
    tr.to_parquet(out/'entry_timing_trades.parquet',index=False); cv.to_csv(out/'entry_timing_event_features.csv',index=False)
    sums=[]
    periods=[('2025_calibration','2025-09-15','2025-12-31'),('2026_frozen','2026-01-01','2026-12-31'),('full_available','1900-01-01','2100-01-01')]
    for name,s,e in periods:
        z=tr[tr.signal_date.between(s,e)]
        for rule,g in z.groupby('rule'): sums.append({'period':name,'rule':rule,**summ(g)})
    sm=pd.DataFrame(sums); sm.to_csv(out/'entry_timing_summary.csv',index=False)
    paired=[]; base=tr[tr.rule=='E0_OPEN'][['ticker','signal_date','net_return']].rename(columns={'net_return':'base_net'})
    for rule,g0 in tr.groupby('rule'):
        g=g0.merge(base,on=['ticker','signal_date'],how='inner')
        for name,s,e in periods:
            h=g[g.signal_date.between(s,e)]
            if len(h)==0: continue
            d=h.net_return-h.base_net; paired.append({'period':name,'rule':rule,'paired_n':len(h),'avg_rule_net':h.net_return.mean(),'avg_base_same_events':h.base_net.mean(),'avg_delta_vs_open':d.mean(),'median_delta_vs_open':d.median(),'win_vs_open_rate':(d>0).mean()})
    pr=pd.DataFrame(paired); pr.to_csv(out/'entry_timing_paired_vs_open.csv',index=False)
    print('\n=== ENTRY TIMING EVENT COVERAGE ==='); print('events_with_minute_data',len(cv)); print('2025_calibration_events',int(cv.signal_date.between('2025-09-15','2025-12-31').sum())); print('2026_events',int(cv.signal_date.between('2026-01-01','2026-12-31').sum())); print('0905_price_coverage',float(cv.px_0905.notna().mean())); print('0910_price_coverage',float(cv.px_0910.notna().mean()))
    print('\n=== ENTRY TIMING SUMMARY ==='); print(sm.to_string(index=False)); print('\n=== ENTRY TIMING PAIRED VS OPEN ==='); print(pr.to_string(index=False)); print('\nwrote ->',out)
if __name__=='__main__': main()
