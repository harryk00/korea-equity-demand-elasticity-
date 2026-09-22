#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from pykrxauth import stock
from pykrxauth.website.comm.session import login


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet')
    p.add_argument('--out-dir',default='data/pit_kosdaq/true_short_balance')
    p.add_argument('--cache-dir',default=str(Path.home()/'.cache/korea-stock-true-short-balance-v1'))
    p.add_argument('--lookback-calendar-days',type=int,default=70)
    p.add_argument('--chunk-calendar-days',type=int,default=180)
    p.add_argument('--pause',type=float,default=.35)
    p.add_argument('--retries',type=int,default=4)
    p.add_argument('--max-tickers',type=int,default=0)
    return p.parse_args()


def c5(df):
    x=df.sort_values(['ticker','date']).copy()
    x['ret_1d']=x['close']/x.groupby('ticker')['close'].shift(1)-1
    x['fm_pct']=x.groupby('date')['free_float_market_cap'].rank(pct=True,ascending=True)
    x['to_pct']=x.groupby('date')['float_turnover_1d'].rank(pct=True,ascending=True)
    x['base']=(x.fm_pct<=.2)&(x.to_pct>=.8)
    m=x.base.fillna(False)
    for c,o in [('float_turnover_1d','t'),('free_float_market_cap','f'),('ret_1d','r')]:
        x[o]=np.nan
        q=x.loc[m].groupby('date')[c].rank(pct=True,ascending=True)
        x.loc[q.index,o]=np.ceil(q*5).clip(1,5)
    x['c5']=x.base&(x.t<=2)&(x.f<=2)&(x.r<5)
    return x.loc[x.c5,['ticker','date']].drop_duplicates().sort_values(['ticker','date'])


def norm(raw,ticker):
    if raw is None or raw.empty:return pd.DataFrame()
    d=raw.reset_index().copy(); d['ticker']=ticker
    dc='날짜' if '날짜' in d else d.columns[0]
    d['date']=pd.to_datetime(d[dc],errors='coerce')
    mp={'공매도잔고':'short_balance_shares','상장주식수':'listed_shares_krx','공매도금액':'short_balance_value','시가총액':'market_cap_krx','비중':'short_balance_ratio_listed_api'}
    for a,b in mp.items():
        if a in d:d[b]=pd.to_numeric(d[a],errors='coerce')
    keep=['ticker','date']+[b for b in mp.values() if b in d]
    return d[keep].dropna(subset=['date'])


def load(p):
    if not p.exists():return pd.DataFrame()
    try:
        d=pd.read_parquet(p); d['ticker']=d.ticker.astype(str).str.zfill(6); d['date']=pd.to_datetime(d.date); return d
    except:return pd.DataFrame()


def merge(a,b):
    if a.empty:x=b.copy()
    elif b.empty:x=a.copy()
    else:x=pd.concat([a,b],ignore_index=True)
    if x.empty:return x
    x['ticker']=x.ticker.astype(str).str.zfill(6); x['date']=pd.to_datetime(x.date)
    return x.drop_duplicates(['ticker','date'],keep='last').sort_values(['ticker','date']).reset_index(drop=True)


def chunks(s,e,n):
    s=pd.Timestamp(s); e=pd.Timestamp(e)
    while s<=e:
        z=min(s+pd.Timedelta(days=n-1),e); yield s,z; s=z+pd.Timedelta(days=1)


def fetch(ticker,s,e,retries,pause):
    last=None
    for i in range(retries):
        try:
            d=stock.get_shorting_balance_by_date(s.strftime('%Y%m%d'),e.strftime('%Y%m%d'),ticker)
            time.sleep(pause); return norm(d,ticker)
        except Exception as ex:
            last=ex; time.sleep(min(2**i,8))
    raise RuntimeError(f'{ticker} {s.date()}~{e.date()} {last!r}')


def main():
    a=args(); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    cache=Path(a.cache_dir).expanduser(); cache.mkdir(parents=True,exist_ok=True)
    m=pd.read_parquet(a.input); m['ticker']=m.ticker.astype(str).str.zfill(6); m['date']=pd.to_datetime(m.date)
    cand=c5(m); cand.to_csv(out/'c5_candidate_dates.csv',index=False)
    tickers=cand.ticker.drop_duplicates().tolist()
    if a.max_tickers>0: tickers=tickers[:a.max_tickers]; cand=cand[cand.ticker.isin(tickers)]
    print('=== TRUE DTC COLLECTION TARGET ==='); print('candidate_rows',len(cand)); print('distinct_tickers',len(tickers))
    errs=[]
    for i,t in enumerate(tickers,1):
        td=cand.loc[cand.ticker==t,'date']; s=td.min()-pd.Timedelta(days=a.lookback_calendar_days); e=td.max(); p=cache/f'{t}.parquet'; old=load(p)
        print(f'\n=== {i}/{len(tickers)} {t} {s.date()}..{e.date()} ===')
        for cs,ce in chunks(s,e,a.chunk_calendar_days):
            if not old.empty:
                hit=((old.date>=cs)&(old.date<=ce)).sum()
                if hit>=max(1,int((ce-cs).days*.45)): continue
            try:
                old=merge(old,fetch(t,cs,ce,a.retries,a.pause)); old.to_parquet(p,index=False)
            except Exception as ex:
                errs.append({'ticker':t,'start':cs,'end':ce,'error':repr(ex)}); print('[WARN]',ex)
    frames=[load(p) for p in cache.glob('*.parquet')]; frames=[d for d in frames if not d.empty]
    if not frames: raise SystemExit('No KRX short-balance data collected. pykrx/KRX authentication may have failed.')
    panel=merge(pd.DataFrame(),pd.concat(frames,ignore_index=True)); panel.to_parquet(out/'true_short_balance_daily.parquet',index=False)
    keys=set(zip(panel.ticker,panel.date.dt.normalize())); c=cand.copy(); c['year']=c.date.dt.year; c['ok']=[(t,pd.Timestamp(d).normalize()) in keys for t,d in zip(c.ticker,c.date)]
    rows=[]
    for sample,g in [('full',c)]+[(str(y),z) for y,z in c.groupby('year')]:
        rows.append({'sample':sample,'candidate_rows':len(g),'same_day_archive_rows':int(g.ok.sum()),'same_day_archive_coverage':float(g.ok.mean()),'distinct_tickers':g.ticker.nunique()})
    cov=pd.DataFrame(rows); cov.to_csv(out/'collection_coverage.csv',index=False)
    if errs: pd.DataFrame(errs).to_csv(out/'collection_errors.csv',index=False)
    meta={'candidate_rows':len(cand),'distinct_tickers':cand.ticker.nunique(),'panel_rows':len(panel),'errors':len(errs),'cache_dir':str(cache)}
    (out/'collection_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('\n=== TRUE SHORT BALANCE COLLECTION COVERAGE ==='); print(cov.to_string(index=False)); print('\n=== META ==='); print(json.dumps(meta,ensure_ascii=False,indent=2)); print('\nwrote ->',out)
if __name__=='__main__':main()
