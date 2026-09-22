#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import requests
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
API_BASE='https://openapi.koreainvestment.com:9443'
TOKEN_PATH='/oauth2/tokenP'
MINUTE_PATH='/uapi/domestic-stock/v1/quotations/inquire-time-dailychartprice'
TR_ID='FHKST03010230'

def getenv_any(names):
    for n in names:
        v=os.getenv(n)
        if v: return v.strip()
    return None

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet')
    p.add_argument('--out-dir',default='data/pit_kosdaq/entry_timing')
    p.add_argument('--cache-dir',default=str(Path.home()/'.cache'/'korea-stock-entry-timing-minute-v1'))
    p.add_argument('--start-date',default='2025-09-15')
    p.add_argument('--end-date',default='2026-08-29')
    p.add_argument('--pause',type=float,default=.12)
    p.add_argument('--max-pages',type=int,default=6)
    p.add_argument('--retries',type=int,default=4)
    return p.parse_args()

def qbucket(s): return np.ceil(s*5).clip(1,5).astype('Int64')

def build_c5_candidates(x):
    x=x.sort_values(['ticker','date']).copy()
    x['prev_close']=x.groupby('ticker')['close'].shift(1)
    x['ret_1d']=x['close']/x['prev_close']-1
    x['float_mcap_pct']=x.groupby('date')['free_float_market_cap'].rank(pct=True)
    x['turnover_pct']=x.groupby('date')['float_turnover_1d'].rank(pct=True)
    x['base_signal']=(x['float_mcap_pct']<=.20)&(x['turnover_pct']>=.80)
    sig=x['base_signal'].fillna(False)
    for col,out in [('float_turnover_1d','turn_signal_pct'),('free_float_market_cap','float_signal_pct'),('ret_1d','ret1_signal_pct')]:
        x[out]=np.nan
        r=x.loc[sig].groupby('date')[col].rank(method='average',pct=True)
        x.loc[r.index,out]=r
    x['turn_q']=qbucket(x['turn_signal_pct']); x['float_q']=qbucket(x['float_signal_pct']); x['ret1_q']=qbucket(x['ret1_signal_pct'])
    x['c5']=x['base_signal']&(x['turn_q']<=2)&(x['float_q']<=2)&(x['ret1_q']<5)
    x['entry_date']=x.groupby('ticker')['date'].shift(-1)
    x['entry_open_daily']=x.groupby('ticker')['open'].shift(-1)
    c=x.loc[x['c5'],['ticker','date','close','entry_date','entry_open_daily']].copy()
    c.columns=['ticker','signal_date','signal_close','entry_date','entry_open_daily']
    return c.dropna(subset=['entry_date','entry_open_daily'])

class KisClient:
    def __init__(self,pause=.12,retries=4):
        self.appkey=getenv_any(['KIS_APP_KEY','KIS_APPKEY','APP_KEY','appkey'])
        self.appsecret=getenv_any(['KIS_APP_SECRET','KIS_APPSECRET','APP_SECRET','appsecret'])
        if not self.appkey or not self.appsecret:
            raise SystemExit('KIS credentials not found. Set KIS_APP_KEY and KIS_APP_SECRET (or KIS_APPKEY/KIS_APPSECRET).')
        self.base=getenv_any(['KIS_BASE_URL']) or API_BASE
        self.pause=pause; self.retries=retries; self.s=requests.Session(); self.token=self._token()
    def _cache(self): return Path.home()/'.cache'/'kis-entry-timing-token-v1.json'
    def _token(self):
        p=self._cache(); p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            try:
                d=json.loads(p.read_text())
                if d.get('expires_at',0)>time.time()+120 and d.get('access_token'): return d['access_token']
            except Exception: pass
        r=self.s.post(self.base+TOKEN_PATH,json={'grant_type':'client_credentials','appkey':self.appkey,'appsecret':self.appsecret},timeout=30)
        r.raise_for_status(); d=r.json(); tok=d.get('access_token')
        if not tok: raise RuntimeError(f'KIS token failed: {d}')
        p.write_text(json.dumps({'access_token':tok,'expires_at':time.time()+int(d.get('expires_in',86400))}))
        return tok
    def headers(self):
        return {'content-type':'application/json; charset=utf-8','authorization':f'Bearer {self.token}','appkey':self.appkey,'appsecret':self.appsecret,'tr_id':TR_ID,'custtype':'P'}
    def get_page(self,ticker,ds,hhmmss):
        params={'FID_COND_MRKT_DIV_CODE':'J','FID_INPUT_ISCD':ticker,'FID_INPUT_HOUR_1':hhmmss,'FID_INPUT_DATE_1':ds,'FID_PW_DATA_INCU_YN':'Y','FID_FAKE_TICK_INCU_YN':''}
        last=None
        for attempt in range(self.retries):
            try:
                r=self.s.get(self.base+MINUTE_PATH,headers=self.headers(),params=params,timeout=30); r.raise_for_status(); d=r.json()
                if str(d.get('rt_cd',''))=='0': time.sleep(self.pause); return d.get('output2') or []
                if 'EGW00201' in str(d) or '초당 거래건수' in str(d.get('msg1','')): time.sleep(61); continue
                raise RuntimeError(f"rt_cd={d.get('rt_cd')} msg={d.get('msg1')}")
            except Exception as e:
                last=e; time.sleep(min(2**attempt,8))
        raise RuntimeError(repr(last))
    def full_day(self,ticker,ds,max_pages=6):
        rows=[]; cursor='153000'
        for _ in range(max_pages):
            page=self.get_page(ticker,ds,cursor)
            if not page: break
            rows.extend(page)
            ts=[str(r.get('stck_cntg_hour','')).zfill(6) for r in page if str(r.get('stck_bsop_date','')).replace('-','')==ds]
            ts=[t for t in ts if t.isdigit() and len(t)==6]
            if not ts: break
            mn=min(ts)
            if mn<='090000': break
            cursor=(datetime.strptime(mn,'%H%M%S')-timedelta(minutes=1)).strftime('%H%M%S')
        return rows

def normalize(rows,ticker,target_date):
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    for c in ['stck_bsop_date','stck_cntg_hour','stck_prpr','stck_oprc','stck_hgpr','stck_lwpr','cntg_vol']:
        if c not in d.columns: d[c]=np.nan
    d['date']=pd.to_datetime(d['stck_bsop_date'].astype(str).str.replace('-','',regex=False),format='%Y%m%d',errors='coerce')
    d['time']=d['stck_cntg_hour'].astype(str).str.zfill(6)
    d=d[(d['date']==pd.Timestamp(target_date))&d['time'].between('090000','153000')].copy()
    for src,dst in {'stck_oprc':'open','stck_hgpr':'high','stck_lwpr':'low','stck_prpr':'close','cntg_vol':'volume'}.items():
        d[dst]=pd.to_numeric(d[src].astype(str).str.replace(',','',regex=False),errors='coerce')
    d['ticker']=ticker
    d['datetime']=pd.to_datetime(d['date'].dt.strftime('%Y%m%d')+d['time'],format='%Y%m%d%H%M%S',errors='coerce')
    return d[['ticker','date','time','datetime','open','high','low','close','volume']].dropna(subset=['datetime','open','high','low','close']).drop_duplicates(['ticker','datetime']).sort_values('datetime')

def main():
    a=parse_args(); cols=['ticker','date','open','close','free_float_market_cap','float_turnover_1d']
    m=pd.read_parquet(a.model,columns=cols); m['ticker']=m['ticker'].astype(str).str.zfill(6); m['date']=pd.to_datetime(m['date'])
    c=build_c5_candidates(m); c['ticker']=c['ticker'].astype(str).str.zfill(6); c['signal_date']=pd.to_datetime(c['signal_date']); c['entry_date']=pd.to_datetime(c['entry_date'])
    c=c[c['entry_date'].between(pd.Timestamp(a.start_date),pd.Timestamp(a.end_date))].drop_duplicates(['ticker','signal_date','entry_date']).reset_index(drop=True)
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); c.to_parquet(out/'entry_timing_candidates.parquet',index=False)
    pairs=c[['ticker','entry_date']].drop_duplicates().sort_values(['entry_date','ticker'])
    cache=Path(a.cache_dir).expanduser(); cache.mkdir(parents=True,exist_ok=True)
    print('=== ENTRY TIMING MINUTE COLLECTION TARGET ==='); print('candidate_events',len(c)); print('unique_ticker_dates',len(pairs)); print('start',c.entry_date.min()); print('end',c.entry_date.max())
    client=KisClient(a.pause,a.retries); errors=[]
    for i,r in enumerate(pairs.itertuples(index=False),1):
        t=str(r.ticker).zfill(6); d=pd.Timestamp(r.entry_date); ds=d.strftime('%Y%m%d'); p=cache/f'{ds}_{t}.parquet'
        if p.exists():
            try:
                q=pd.read_parquet(p)
                if not q.empty:
                    if i%50==0 or i in (1,len(pairs)): print(f'{i}/{len(pairs)} {t} {ds} cache rows={len(q)}')
                    continue
            except Exception: pass
        try:
            q=normalize(client.full_day(t,ds,a.max_pages),t,d)
            if q.empty: errors.append({'ticker':t,'date':ds,'error':'empty'}); print(f'[EMPTY] {i}/{len(pairs)} {t} {ds}'); continue
            q.to_parquet(p,index=False)
            if i%20==0 or i in (1,len(pairs)): print(f'{i}/{len(pairs)} {t} {ds} rows={len(q)} {q.time.min()}..{q.time.max()}')
        except Exception as e:
            errors.append({'ticker':t,'date':ds,'error':repr(e)}); print(f'[WARN] {i}/{len(pairs)} {t} {ds}: {e}')
    frames=[]
    for p in sorted(cache.glob('*.parquet')):
        try:
            q=pd.read_parquet(p)
            if not q.empty: frames.append(q)
        except Exception: pass
    if not frames: raise SystemExit('No KIS minute bars collected.')
    panel=pd.concat(frames,ignore_index=True); panel['ticker']=panel['ticker'].astype(str).str.zfill(6); panel['date']=pd.to_datetime(panel['date']); panel=panel.drop_duplicates(['ticker','datetime']).sort_values(['ticker','datetime'])
    panel.to_parquet(out/'entry_timing_minute_bars.parquet',index=False)
    if errors: pd.DataFrame(errors).to_csv(out/'entry_timing_collection_errors.csv',index=False)
    cov=pairs.merge(panel.groupby(['ticker','date']).agg(minute_rows=('datetime','size'),min_time=('time','min'),max_time=('time','max')).reset_index(),left_on=['ticker','entry_date'],right_on=['ticker','date'],how='left')
    cov['has_0905_or_later']=cov['max_time'].fillna('')>='090500'; cov['has_0910_or_later']=cov['max_time'].fillna('')>='091000'; cov['fullish_day']=(cov['min_time'].fillna('')<='090100')&(cov['max_time'].fillna('')>='152900')
    cov.to_csv(out/'entry_timing_minute_coverage.csv',index=False)
    print('\n=== ENTRY TIMING MINUTE COVERAGE ==='); print('unique_ticker_dates',len(cov)); print('collected',int(cov.minute_rows.notna().sum())); print('coverage',float(cov.minute_rows.notna().mean())); print('has_0905_or_later',float(cov.has_0905_or_later.mean())); print('has_0910_or_later',float(cov.has_0910_or_later.mean())); print('fullish_day',float(cov.fullish_day.mean())); print('errors',len(errors)); print('wrote ->',out/'entry_timing_minute_bars.parquet')
if __name__=='__main__': main()
