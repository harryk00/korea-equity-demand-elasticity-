#!/usr/bin/env python3
from __future__ import annotations
import argparse
from math import sqrt
from pathlib import Path
import numpy as np
import pandas as pd


def args():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet'); p.add_argument('--short-balance',default='data/pit_kosdaq/true_short_balance/true_short_balance_daily.parquet'); p.add_argument('--out-dir',default='data/pit_kosdaq/analysis/true_dtc_final'); p.add_argument('--publication-lag-bars',type=int,default=2); p.add_argument('--max-hold',type=int,default=20); p.add_argument('--tp',type=float,default=.30); p.add_argument('--sl',type=float,default=-.15); return p.parse_args()

def q(p): return np.ceil(p*5).clip(1,5).astype('Int64')

def add_c5(x):
    x=x.sort_values(['ticker','date']).copy(); x['ret_1d']=x.close/x.groupby('ticker').close.shift(1)-1; x['fm_pct']=x.groupby('date').free_float_market_cap.rank(pct=True,ascending=True); x['to_pct']=x.groupby('date').float_turnover_1d.rank(pct=True,ascending=True); x['base']=(x.fm_pct<=.2)&(x.to_pct>=.8); m=x.base.fillna(False)
    for c,o in [('float_turnover_1d','tq'),('free_float_market_cap','fq'),('ret_1d','rq')]:
        x[o]=pd.Series(pd.NA,index=x.index,dtype='Int64'); r=x.loc[m].groupby('date')[c].rank(pct=True,ascending=True); x.loc[r.index,o]=q(r)
    x['c5']=x.base&(x.tq<=2)&(x.fq<=2)&(x.rq<5); return x

def enrich(m,s,lag):
    m=m.sort_values(['ticker','date']).copy(); s=s.sort_values(['ticker','date']).copy()
    if 'free_float_shares' not in m: m['free_float_shares']=m.free_float_market_cap/m.close.replace(0,np.nan)
    m['adv20_lag1']=m.groupby('ticker').volume.transform(lambda z:z.shift(1).rolling(20,min_periods=10).mean())
    s['reported_short_balance_shares']=s.groupby('ticker').short_balance_shares.shift(lag)
    s['bal_lag5']=s.groupby('ticker').reported_short_balance_shares.shift(5); s['reported_short_balance_change_5d']=s.reported_short_balance_shares/s.bal_lag5.replace(0,np.nan)-1
    m=m.merge(s[['ticker','date','reported_short_balance_shares','reported_short_balance_change_5d']],on=['ticker','date'],how='left')
    m['short_interest_to_float']=m.reported_short_balance_shares/m.free_float_shares.replace(0,np.nan); m['true_dtc']=m.reported_short_balance_shares/m.adv20_lag1.replace(0,np.nan); return m

def events(x,hold,tp,sl):
    rows=[]
    for t,g in x.groupby('ticker',sort=False):
        g=g.sort_values('date').reset_index(drop=True); n=len(g)
        for pos in np.flatnonzero(g.c5.fillna(False).to_numpy()):
            ep=int(pos)+1
            if ep+hold-1>=n: continue
            s=g.iloc[int(pos)]; e=g.iloc[ep]; entry=float(e.open)
            if not np.isfinite(entry) or entry<=0:continue
            w=g.iloc[ep:ep+hold]; tpp=entry*(1+tp); slp=entry*(1+sl); reason='TIME'; exitp=float(w.iloc[-1].close)
            for r in w.itertuples(index=False):
                o,h,l=float(r.open),float(r.high),float(r.low)
                if o<=slp:reason='SL';exitp=o;break
                if l<=slp:reason='SL';exitp=slp;break
                if h>=tpp:reason='TP';exitp=tpp;break
            mx=float(w.high.max())/float(s.close)-1
            rows.append({'ticker':t,'signal_date':pd.Timestamp(s.date),'year':pd.Timestamp(s.date).year,'trade_return_gross':exitp/entry-1,'tp_before_sl':int(reason=='TP'),'target30':int(mx>=.3),'target50':int(mx>=.5),'target100':int(mx>=1.0),'short_interest_to_float':s.get('short_interest_to_float',np.nan),'true_dtc':s.get('true_dtc',np.nan),'reported_short_balance_change_5d':s.get('reported_short_balance_change_5d',np.nan)})
    return pd.DataFrame(rows)

def wilson(k,n,z=1.96):
    if n==0:return np.nan,np.nan
    p=k/n; den=1+z*z/n; ctr=(p+z*z/(2*n))/den; rad=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return ctr-rad,ctr+rad

def stats(g):
    if g.empty:return {'n':0,'p30':np.nan,'p50':np.nan,'p100':np.nan,'tp':np.nan,'avg_return':np.nan,'median_return':np.nan,'pf':np.nan}
    r=g.trade_return_gross; pos=r[r>0].sum(); neg=-r[r<0].sum(); out={'n':len(g),'tp':g.tp_before_sl.mean(),'avg_return':r.mean(),'median_return':r.median(),'pf':pos/neg if neg>0 else np.nan}
    for c,k in [('target30','p30'),('target50','p50'),('target100','p100')]: out[k]=g[c].mean()
    return out

def addq(ev,c):
    qc=c+'_q'; ev[qc]=pd.Series(pd.NA,index=ev.index,dtype='Int64'); m=ev[c].notna(); r=ev.loc[m].groupby('signal_date')[c].rank(pct=True,ascending=True); ev.loc[r.index,qc]=q(r); return ev

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    m=pd.read_parquet(a.model); m['ticker']=m.ticker.astype(str).str.zfill(6); m['date']=pd.to_datetime(m.date); m=add_c5(m)
    s=pd.read_parquet(a.short_balance); s['ticker']=s.ticker.astype(str).str.zfill(6); s['date']=pd.to_datetime(s.date)
    ev=events(enrich(m,s,a.publication_lag_bars),a.max_hold,a.tp,a.sl)
    feats=['short_interest_to_float','true_dtc','reported_short_balance_change_5d']
    qt=[]
    for f in feats:
        ev=addq(ev,f); qc=f+'_q'
        for y in [2025,2026]:
            yy=ev[ev.year==y]
            for z in range(1,6): qt.append({'feature':f,'year':y,'quintile':z,**stats(yy[yy[qc]==z])})
    qt=pd.DataFrame(qt)
    summ=[]; grid=[]
    for y in [2025,2026]:
        yy=ev[ev.year==y]
        for sq in range(1,6):
            for dq in range(1,6): grid.append({'year':y,'short_to_float_q':sq,'true_dtc_q':dq,**stats(yy[(yy.short_interest_to_float_q==sq)&(yy.true_dtc_q==dq)])})
        b=stats(yy); hh=stats(yy[(yy.short_interest_to_float_q>=4)&(yy.true_dtc_q>=4)]); tr=stats(yy[(yy.short_interest_to_float_q>=4)&(yy.true_dtc_q>=4)&(yy.reported_short_balance_change_5d_q>=4)])
        row={'year':y}
        for pref,d in [('baseline',b),('high_short_high_dtc',hh),('triple_high',tr)]:
            for k,v in d.items(): row[pref+'_'+k]=v
        row['p30_lift']=hh['p30']-b['p30'];row['p50_lift']=hh['p50']-b['p50'];row['p100_lift']=hh['p100']-b['p100'];row['tp_lift']=hh['tp']-b['tp'];row['avg_return_lift']=hh['avg_return']-b['avg_return']; summ.append(row)
    summ=pd.DataFrame(summ); grid=pd.DataFrame(grid)
    cov=[]
    for y in [2025,2026]:
        yy=ev[ev.year==y]; cov.append({'year':y,'events':len(yy),**{f+'_coverage':yy[f].notna().mean() for f in feats}})
    cov=pd.DataFrame(cov)
    ev.to_parquet(out/'true_dtc_events.parquet',index=False); qt.to_csv(out/'true_dtc_quintiles.csv',index=False); summ.to_csv(out/'true_dtc_squeeze_summary.csv',index=False); grid.to_csv(out/'short_interest_x_true_dtc_5x5.csv',index=False); cov.to_csv(out/'true_dtc_event_coverage.csv',index=False)
    print('\n=== TRUE DTC EVENT COVERAGE ===');print(cov.to_string(index=False));print('\n=== TRUE DTC QUINTILE DIAGNOSTIC ===');print(qt.to_string(index=False));print('\n=== TRUE DTC SQUEEZE SUMMARY ===');print(summ.to_string(index=False));print('\n=== IMPORTANT ===');print(f'- KRX reported balance lagged {a.publication_lag_bars} trading observations.');print('- true_dtc = lagged KRX-reported short-balance shares / prior-20d ADV.');print('- short_interest_to_float = lagged KRX-reported short-balance shares / free-float shares.');print('- target30/50/100 = future 20-bar upside from signal close.');print('- tp_before_sl = next-open entry, TP +30%, SL -15%.');print('\nwrote ->',out)
if __name__=='__main__':main()
