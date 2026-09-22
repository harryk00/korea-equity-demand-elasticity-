#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd, requests

BASE='https://openapi.koreainvestment.com:9443'
TOKEN='/oauth2/tokenP'
CREDIT=('/uapi/domestic-stock/v1/quotations/daily-credit-balance','FHPST04760000')
LOAN=('/uapi/domestic-stock/v1/quotations/daily-loan-trans','HHPST074500C0')
SHORT=('/uapi/domestic-stock/v1/quotations/daily-short-sale','FHPST04830000')

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet')
    p.add_argument('--out-dir',default='data/pit_kosdaq/sell_pressure')
    p.add_argument('--cache-dir',default=str(Path.home()/'.cache/korea-stock-sell-pressure-v1'))
    p.add_argument('--pause',type=float,default=1.0); p.add_argument('--retries',type=int,default=5)
    p.add_argument('--lookback-days',type=int,default=70); p.add_argument('--chunk-days',type=int,default=90)
    p.add_argument('--max-tickers',type=int,default=0)
    return p.parse_args()

def qbucket(s): return np.ceil(s*5).clip(1,5).astype('Int64')

def c5_candidates(df):
    x=df.sort_values(['ticker','date']).copy(); g=x.groupby('ticker')
    x['ret1']=x['close']/g['close'].shift(1)-1
    x['fm_pct']=x.groupby('date')['free_float_market_cap'].rank(pct=True)
    x['to_pct']=x.groupby('date')['float_turnover_1d'].rank(pct=True)
    x['base']=(x.fm_pct<=.2)&(x.to_pct>=.8); m=x.base.fillna(False)
    for c,o in [('float_turnover_1d','t'),('free_float_market_cap','f'),('ret1','r')]:
        x[o+'_pct']=np.nan
        rr=x.loc[m].groupby('date')[c].rank(pct=True)
        x.loc[rr.index,o+'_pct']=rr
    x['c5']=x.base&(qbucket(x.t_pct)<=2)&(qbucket(x.f_pct)<=2)&(qbucket(x.r_pct)<5)
    return x.loc[x.c5,['ticker','date']].drop_duplicates().sort_values(['ticker','date']).reset_index(drop=True)

class KIS:
    def __init__(self,key,secret,cache,pause,retries):
        self.key=key; self.secret=secret; self.cache=Path(cache).expanduser(); self.cache.mkdir(parents=True,exist_ok=True)
        self.pause=pause; self.retries=retries; self.s=requests.Session(); self.last=0.; self.tf=self.cache/'token.json'; self.token=self._token()
    def _token(self):
        if self.tf.exists():
            try:
                o=json.loads(self.tf.read_text())
                if o.get('expires_at',0)>time.time()+300: return o['access_token']
            except Exception: pass
        r=self.s.post(BASE+TOKEN,json={'grant_type':'client_credentials','appkey':self.key,'appsecret':self.secret},headers={'content-type':'application/json'},timeout=30); r.raise_for_status(); o=r.json(); tok=o['access_token']
        try: exp=float(o.get('expires_in',23*3600))
        except Exception: exp=23*3600
        self.tf.write_text(json.dumps({'access_token':tok,'expires_at':time.time()+max(3600,exp-300)})); return tok
    def get(self,path,tr,params):
        for a in range(self.retries):
            w=self.pause-(time.time()-self.last)
            if w>0: time.sleep(w)
            try:
                r=self.s.get(BASE+path,headers={'authorization':f'Bearer {self.token}','appkey':self.key,'appsecret':self.secret,'tr_id':tr,'custtype':'P'},params=params,timeout=30); self.last=time.time()
                if r.status_code==401:
                    self.tf.unlink(missing_ok=True); self.token=self._token(); continue
                r.raise_for_status(); o=r.json()
                if str(o.get('rt_cd','0'))=='0': return o
                if a<self.retries-1: time.sleep(min(2**a,16)); continue
                raise RuntimeError(f"{o.get('msg_cd')} {o.get('msg1')}")
            except (requests.RequestException,ValueError):
                self.last=time.time()
                if a==self.retries-1: raise
                time.sleep(min(2**a,16))
        raise RuntimeError('unreachable')

def num(s): return pd.to_numeric(s.replace('',np.nan),errors='coerce')

def norm_credit(rows,t):
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows); dc='deal_date' if 'deal_date' in d else ('stlm_date' if 'stlm_date' in d else None)
    if dc is None: return pd.DataFrame()
    d['date']=pd.to_datetime(d[dc],format='%Y%m%d',errors='coerce'); d['ticker']=t
    mp={'whol_loan_new_stcn':'credit_new_shares','whol_loan_rdmp_stcn':'credit_repay_shares','whol_loan_rmnd_stcn':'credit_balance_shares','whol_loan_rmnd_amt':'credit_balance_amount','whol_loan_rmnd_rate':'credit_balance_rate_api','whol_stln_rmnd_stcn':'retail_stockloan_balance_shares'}
    for c in mp:
        if c in d: d[c]=num(d[c])
    d=d.rename(columns=mp); keep=['ticker','date']+[v for v in mp.values() if v in d]; return d[keep].dropna(subset=['date'])

def norm_loan(rows,t):
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    if 'bsop_date' not in d: return pd.DataFrame()
    d['date']=pd.to_datetime(d.bsop_date,format='%Y%m%d',errors='coerce'); d['ticker']=t
    mp={'new_stcn':'lending_new_shares','rdmp_stcn':'lending_return_shares','prdy_rmnd_vrss':'lending_balance_change_shares','rmnd_stcn':'lending_balance_shares','rmnd_amt':'lending_balance_amount'}
    for c in mp:
        if c in d: d[c]=num(d[c])
    d=d.rename(columns=mp); keep=['ticker','date']+[v for v in mp.values() if v in d]; return d[keep].dropna(subset=['date'])

def norm_short(rows,t):
    if not rows: return pd.DataFrame()
    d=pd.DataFrame(rows)
    if 'stck_bsop_date' not in d: return pd.DataFrame()
    d['date']=pd.to_datetime(d.stck_bsop_date,format='%Y%m%d',errors='coerce'); d['ticker']=t
    mp={'ssts_cntg_qty':'short_sale_qty','ssts_vol_rlim':'short_sale_volume_ratio_api','ssts_tr_pbmn':'short_sale_value','ssts_tr_pbmn_rlim':'short_sale_value_ratio_api','acml_vol':'api_volume'}
    for c in mp:
        if c in d: d[c]=num(d[c])
    d=d.rename(columns=mp); keep=['ticker','date']+[v for v in mp.values() if v in d]; return d[keep].dropna(subset=['date'])

def load(p):
    try: return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()

def merge(a,b):
    x=b.copy() if a.empty else (a.copy() if b.empty else pd.concat([a,b],ignore_index=True))
    if x.empty: return x
    x['ticker']=x.ticker.astype(str).str.zfill(6); x['date']=pd.to_datetime(x.date)
    return x.drop_duplicates(['ticker','date'],keep='last').sort_values(['ticker','date']).reset_index(drop=True)

def chunks(s,e,n):
    s=pd.Timestamp(s); e=pd.Timestamp(e)
    while s<=e:
        z=min(s+pd.Timedelta(days=n-1),e); yield s,z; s=z+pd.Timedelta(days=1)

def fetch_credit(k,t,s,e):
    fs=[]; a=pd.Timestamp(e); s=pd.Timestamp(s); seen=set()
    while a>=s:
        k0=a.strftime('%Y%m%d')
        if k0 in seen: break
        seen.add(k0); o=k.get(*CREDIT,{'FID_COND_MRKT_DIV_CODE':'J','FID_COND_SCR_DIV_CODE':'20476','FID_INPUT_ISCD':t,'FID_INPUT_DATE_1':k0}); d=norm_credit(o.get('output') or [],t)
        if d.empty: break
        fs.append(d); mn=d.date.min()
        if mn<=s: break
        a=mn-pd.Timedelta(days=1)
    return pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()

def fetch_loan(k,t,s,e,n):
    fs=[]
    for a,b in chunks(s,e,n):
        o=k.get(*LOAN,{'MRKT_DIV_CLS_CODE':'3','MKSC_SHRN_ISCD':t,'START_DATE':a.strftime('%Y%m%d'),'END_DATE':b.strftime('%Y%m%d'),'CTS':''}); d=norm_loan(o.get('output1') or o.get('output') or [],t)
        if not d.empty: fs.append(d)
    return pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()

def fetch_short(k,t,s,e,n):
    fs=[]
    for a,b in chunks(s,e,n):
        o=k.get(*SHORT,{'FID_COND_MRKT_DIV_CODE':'J','FID_INPUT_ISCD':t,'FID_INPUT_DATE_1':a.strftime('%Y%m%d'),'FID_INPUT_DATE_2':b.strftime('%Y%m%d')}); d=norm_short(o.get('output2') or [],t)
        if not d.empty: fs.append(d)
    return pd.concat(fs,ignore_index=True) if fs else pd.DataFrame()

def collect_dir(d):
    fs=[load(p) for p in sorted(d.glob('*.parquet'))]; fs=[x for x in fs if not x.empty]
    return merge(pd.DataFrame(),pd.concat(fs,ignore_index=True)) if fs else pd.DataFrame()

def cov(c,p,name):
    keys=set(zip(p.ticker,p.date.dt.normalize())) if not p.empty else set(); z=c.copy(); z['year']=z.date.dt.year; z['covered']=[(t,pd.Timestamp(d).normalize()) in keys for t,d in zip(z.ticker,z.date)]
    rs=[]
    for sm,g in [('full',z)]+[(str(y),q) for y,q in z.groupby('year')]: rs.append({'panel':name,'sample':sm,'candidate_rows':len(g),'covered_rows':int(g.covered.sum()),'coverage_rate_same_day':float(g.covered.mean()),'distinct_tickers':g.ticker.nunique()})
    return pd.DataFrame(rs)

def main():
    a=args(); key=os.getenv('KIS_APP_KEY'); sec=os.getenv('KIS_APP_SECRET')
    if not key or not sec: raise SystemExit('Set KIS_APP_KEY and KIS_APP_SECRET')
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); cache=Path(a.cache_dir).expanduser(); dirs={n:cache/n for n in ['credit','loan','short']}; [d.mkdir(parents=True,exist_ok=True) for d in dirs.values()]
    m=pd.read_parquet(a.input); m['ticker']=m.ticker.astype(str).str.zfill(6); m['date']=pd.to_datetime(m.date); c=c5_candidates(m); c.to_csv(out/'c5_candidate_dates.csv',index=False)
    ts=c.ticker.drop_duplicates().tolist(); ts=ts[:a.max_tickers] if a.max_tickers>0 else ts; c=c[c.ticker.isin(ts)]
    print('=== SELL PRESSURE COLLECTION TARGET ==='); print('candidate_rows',len(c)); print('distinct_tickers',len(ts))
    k=KIS(key,sec,cache,a.pause,a.retries); errs=[]
    for i,t in enumerate(ts,1):
        d=c.loc[c.ticker==t,'date']; s=d.min()-pd.Timedelta(days=a.lookback_days); e=d.max(); print(f'\n=== {i}/{len(ts)} {t} {s.date()}..{e.date()} ===')
        for name,fn in [('credit',lambda:fetch_credit(k,t,s,e)),('loan',lambda:fetch_loan(k,t,s,e,a.chunk_days)),('short',lambda:fetch_short(k,t,s,e,a.chunk_days))]:
            p=dirs[name]/f'{t}.parquet'; old=load(p)
            try:
                new=fn(); z=merge(old,new)
                if not z.empty: z.to_parquet(p,index=False)
            except Exception as ex:
                errs.append({'panel':name,'ticker':t,'error':repr(ex)}); print(f'[WARN {name}]',t,ex)
    panels={n:collect_dir(d) for n,d in dirs.items()}
    names={'credit':'credit_daily.parquet','loan':'lending_daily.parquet','short':'short_sale_daily.parquet'}
    for n,p in panels.items():
        if not p.empty: p.to_parquet(out/names[n],index=False)
    cv=pd.concat([cov(c,panels['credit'],'credit'),cov(c,panels['loan'],'lending'),cov(c,panels['short'],'short_sale_flow')],ignore_index=True); cv.to_csv(out/'coverage_summary.csv',index=False)
    if errs: pd.DataFrame(errs).to_csv(out/'collection_errors.csv',index=False)
    meta={'candidate_rows':len(c),'distinct_tickers':c.ticker.nunique(),'credit_rows':len(panels['credit']),'lending_rows':len(panels['loan']),'short_rows':len(panels['short']),'errors':len(errs),'cache_dir':str(cache)}; (out/'collection_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print('\n=== SELL PRESSURE COVERAGE ==='); print(cv.to_string(index=False)); print('\n=== META ==='); print(json.dumps(meta,ensure_ascii=False,indent=2)); print('\nwrote ->',out)
if __name__=='__main__': main()
