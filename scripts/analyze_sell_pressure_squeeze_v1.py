#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet')
    p.add_argument('--panel-dir',default='data/pit_kosdaq/sell_pressure')
    p.add_argument('--out-dir',default='data/pit_kosdaq/analysis/sell_pressure_v1')
    p.add_argument('--panel-lag-bars',type=int,default=1)
    p.add_argument('--max-hold',type=int,default=20); p.add_argument('--tp',type=float,default=.30); p.add_argument('--sl',type=float,default=-.15)
    return p.parse_args()

def qbucket(s): return np.ceil(s*5).clip(1,5).astype('Int64')

def add_c5(x):
    x=x.sort_values(['ticker','date']).copy(); g=x.groupby('ticker')
    x['ret1']=x['close']/g['close'].shift(1)-1
    x['fm_pct']=x.groupby('date')['free_float_market_cap'].rank(pct=True)
    x['to_pct']=x.groupby('date')['float_turnover_1d'].rank(pct=True)
    x['base']=(x.fm_pct<=.2)&(x.to_pct>=.8); m=x.base.fillna(False)
    for c,o in [('float_turnover_1d','t'),('free_float_market_cap','f'),('ret1','r')]:
        x[o+'_pct']=np.nan; rr=x.loc[m].groupby('date')[c].rank(pct=True); x.loc[rr.index,o+'_pct']=rr
    x['c5']=x.base&(qbucket(x.t_pct)<=2)&(qbucket(x.f_pct)<=2)&(qbucket(x.r_pct)<5)
    return x

def loadp(p):
    if not p.exists(): return pd.DataFrame()
    d=pd.read_parquet(p); d['ticker']=d.ticker.astype(str).str.zfill(6); d['date']=pd.to_datetime(d.date); return d.sort_values(['ticker','date'])

def enrich(x,credit,loan,short,lag):
    if 'free_float_shares' not in x: x['free_float_shares']=x.free_float_market_cap/x.close.replace(0,np.nan)
    g=x.groupby('ticker',group_keys=False)
    x['adv20_lag1']=g['volume'].apply(lambda s:s.shift(1).rolling(20,min_periods=10).mean()).reset_index(level=0,drop=True)
    def lagp(d,cols):
        if d.empty:return d
        z=d.copy().sort_values(['ticker','date'])
        for c in cols:
            if c in z: z[c]=z.groupby('ticker')[c].shift(lag)
        return z
    cc=['credit_balance_shares','credit_new_shares','credit_repay_shares','retail_stockloan_balance_shares']
    lc=['lending_balance_shares','lending_new_shares','lending_return_shares','lending_balance_change_shares']
    sc=['short_sale_qty','short_sale_volume_ratio_api','short_sale_value','short_sale_value_ratio_api']
    credit,loan,short=lagp(credit,cc),lagp(loan,lc),lagp(short,sc)
    for d,cols in [(credit,cc),(loan,lc),(short,sc)]:
        if not d.empty:
            keep=['ticker','date']+[c for c in cols if c in d]; x=x.merge(d[keep],on=['ticker','date'],how='left')
    if 'credit_balance_shares' in x:
        x['credit_to_float']=x.credit_balance_shares/x.free_float_shares.replace(0,np.nan)
        x['credit_change_5d']=x.groupby('ticker').credit_balance_shares.pct_change(5)
    if 'lending_balance_shares' in x:
        x['lending_to_float']=x.lending_balance_shares/x.free_float_shares.replace(0,np.nan)
        x['lending_change_5d']=x.groupby('ticker').lending_balance_shares.pct_change(5)
        x['loan_dtc_proxy']=x.lending_balance_shares/x.adv20_lag1.replace(0,np.nan)
    if 'short_sale_qty' in x: x['short_flow_to_adv20']=x.short_sale_qty/x.adv20_lag1.replace(0,np.nan)
    return x

def events(x,hold,tp,sl):
    rs=[]
    for t,g in x.groupby('ticker',sort=False):
        g=g.sort_values('date').reset_index(drop=True); n=len(g)
        for pos in np.flatnonzero(g.c5.fillna(False).to_numpy()):
            pos=int(pos); ep=pos+1
            if ep+hold-1>=n: continue
            s=g.iloc[pos]; e=g.iloc[ep]; px=float(e.open)
            if not np.isfinite(px) or px<=0: continue
            w=g.iloc[ep:ep+hold]; tpx=px*(1+tp); spx=px*(1+sl); reason='TIME'; ex=float(w.iloc[-1].close); maxh=float(w.high.max())
            for r in w.itertuples(index=False):
                o,h,l=float(r.open),float(r.high),float(r.low)
                if o<=spx: reason='SL'; ex=o; break
                if l<=spx: reason='SL'; ex=spx; break
                if h>=tpx: reason='TP'; ex=tpx; break
            sigc=float(s.close); mx=maxh/sigc-1 if sigc>0 else np.nan
            row={'ticker':t,'signal_date':pd.Timestamp(s.date),'trade_return_gross':ex/px-1,'tp_before_sl':int(reason=='TP'),'sl_before_tp':int(reason=='SL'),'target30':int(mx>=.30),'target50':int(mx>=.50),'target100':int(mx>=1.0),'max_signal_return_20d':mx}
            for c in ['credit_to_float','credit_change_5d','lending_to_float','lending_change_5d','loan_dtc_proxy','short_flow_to_adv20','short_sale_volume_ratio_api']:
                if c in s.index: row[c]=s[c]
            rs.append(row)
    e=pd.DataFrame(rs)
    if e.empty: raise SystemExit('No C5 events generated')
    e['year']=e.signal_date.dt.year; return e

def stats(g):
    if g.empty:return {'n':0,'target30':np.nan,'target50':np.nan,'target100':np.nan,'tp_before_sl':np.nan,'avg_trade_return':np.nan,'median_trade_return':np.nan,'avg_max20':np.nan,'pf':np.nan}
    r=g.trade_return_gross; pos=r[r>0].sum(); neg=-r[r<0].sum()
    return {'n':len(g),'target30':g.target30.mean(),'target50':g.target50.mean(),'target100':g.target100.mean(),'tp_before_sl':g.tp_before_sl.mean(),'avg_trade_return':r.mean(),'median_trade_return':r.median(),'avg_max20':g.max_signal_return_20d.mean(),'pf':pos/neg if neg>0 else np.nan}

def add_q(e,c):
    qc=c+'_q'; e[qc]=pd.NA; m=e[c].notna(); r=e.loc[m].groupby('signal_date')[c].rank(pct=True); e.loc[m,qc]=np.ceil(r*5).clip(1,5).astype(int); e[qc]=e[qc].astype('Int64'); return e

def quintiles(e,fs):
    rows=[]
    for f in fs:
        if f not in e: continue
        e=add_q(e,f); qc=f+'_q'
        for y in [2025,2026]:
            z=e[e.year==y]
            for q in range(1,6): rows.append({'feature':f,'year':y,'quintile':q,**stats(z[z[qc]==q])})
    return e,pd.DataFrame(rows)

def overhang(e):
    rows=[]
    for f in ['credit_to_float','lending_to_float']:
        qc=f+'_q'
        if qc not in e: continue
        for y in [2025,2026]:
            z=e[e.year==y]; lo=stats(z[z[qc]<=2]); hi=stats(z[z[qc]>=4])
            rows.append({'feature':f,'year':y,'low_n':lo['n'],'low_avg_return':lo['avg_trade_return'],'low_pf':lo['pf'],'low_tp':lo['tp_before_sl'],'high_n':hi['n'],'high_avg_return':hi['avg_trade_return'],'high_pf':hi['pf'],'high_tp':hi['tp_before_sl'],'high_minus_low_avg_return':hi['avg_trade_return']-lo['avg_trade_return'],'high_minus_low_tp':hi['tp_before_sl']-lo['tp_before_sl']})
    return pd.DataFrame(rows)

def squeeze(e):
    if not {'lending_to_float_q','loan_dtc_proxy_q'}.issubset(e.columns): return pd.DataFrame(),pd.DataFrame()
    rows=[]; cells=[]
    for y in [2025,2026]:
        z=e[e.year==y]
        for lq in range(1,6):
            for dq in range(1,6): cells.append({'year':y,'lending_q':lq,'dtc_proxy_q':dq,**stats(z[(z.lending_to_float_q==lq)&(z.loan_dtc_proxy_q==dq)])})
        b=stats(z); hh=stats(z[(z.lending_to_float_q>=4)&(z.loan_dtc_proxy_q>=4)])
        if 'short_flow_to_adv20_q' in z: th=stats(z[(z.lending_to_float_q>=4)&(z.loan_dtc_proxy_q>=4)&(z.short_flow_to_adv20_q>=4)])
        else: th=stats(z.iloc[0:0])
        rows.append({'year':y,'baseline_n':b['n'],'baseline_target30':b['target30'],'baseline_target50':b['target50'],'baseline_target100':b['target100'],'baseline_tp':b['tp_before_sl'],'baseline_avg_return':b['avg_trade_return'],'high_lending_high_dtc_n':hh['n'],'high_lending_high_dtc_target30':hh['target30'],'high_lending_high_dtc_target50':hh['target50'],'high_lending_high_dtc_target100':hh['target100'],'high_lending_high_dtc_tp':hh['tp_before_sl'],'high_lending_high_dtc_avg_return':hh['avg_trade_return'],'triple_high_n':th['n'],'triple_high_target30':th['target30'],'triple_high_target50':th['target50'],'triple_high_target100':th['target100'],'triple_high_tp':th['tp_before_sl'],'triple_high_avg_return':th['avg_trade_return']})
    return pd.DataFrame(rows),pd.DataFrame(cells)

def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); pd0=Path(a.panel_dir)
    m=pd.read_parquet(a.model); m['ticker']=m.ticker.astype(str).str.zfill(6); m['date']=pd.to_datetime(m.date)
    if 'volume' not in m: raise SystemExit('Model needs volume')
    m=add_c5(m); c=loadp(pd0/'credit_daily.parquet'); l=loadp(pd0/'lending_daily.parquet'); s=loadp(pd0/'short_sale_daily.parquet'); x=enrich(m,c,l,s,a.panel_lag_bars); e=events(x,a.max_hold,a.tp,a.sl)
    fs=['credit_to_float','credit_change_5d','lending_to_float','lending_change_5d','loan_dtc_proxy','short_flow_to_adv20','short_sale_volume_ratio_api']; e,qt=quintiles(e,fs); oh=overhang(e); sq,grid=squeeze(e)
    cov=[]
    for y in [2025,2026]:
        z=e[e.year==y]; row={'year':y,'events':len(z)}
        for f in fs:
            if f in z: row[f+'_coverage']=z[f].notna().mean()
        cov.append(row)
    cv=pd.DataFrame(cov); e.to_parquet(out/'sell_pressure_events.parquet',index=False); qt.to_csv(out/'feature_quintiles.csv',index=False); oh.to_csv(out/'overhang_high_vs_low.csv',index=False); sq.to_csv(out/'squeeze_proxy_summary.csv',index=False); grid.to_csv(out/'lending_x_dtc_5x5.csv',index=False); cv.to_csv(out/'event_feature_coverage.csv',index=False)
    print('\n=== SELL PRESSURE EVENT COVERAGE ==='); print(cv.to_string(index=False)); print('\n=== 3A OVERHANG: HIGH vs LOW ==='); print(oh.to_string(index=False) if not oh.empty else 'No overhang data'); print('\n=== 3B SQUEEZE PROXY SUMMARY ==='); print(sq.to_string(index=False) if not sq.empty else 'No squeeze proxy data')
    print('\n=== IMPORTANT ==='); print('- loan_dtc_proxy = lending balance / prior 20-day ADV.'); print('- This is NOT true short-interest DTC; KIS short-sale daily is flow, not short-interest balance.'); print(f'- Sell-pressure panels are lagged by {a.panel_lag_bars} trading observation(s).'); print('- target30/50/100 = future 20-bar upside from signal close.'); print('- tp_before_sl = next-open entry, TP +30%, SL -15%.'); print('\nwrote ->',out)
if __name__=='__main__': main()
