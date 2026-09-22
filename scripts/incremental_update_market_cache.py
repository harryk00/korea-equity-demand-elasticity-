#!/usr/bin/env python3
from __future__ import annotations

import argparse, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from korea_sd.data.kis_client import KisOpenApiClient
from korea_sd.data.kis_market import KisMarketProvider
from korea_sd.data.pit_universe import PykrxOhlcvOnlyProvider
from korea_sd.utils import atomic_write_parquet


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--universe", default="data/pit_kosdaq/universe/pit_pipeline_universe.csv")
    p.add_argument("--membership", default="data/pit_kosdaq/universe/membership_daily.parquet")
    p.add_argument("--core-root", default="data/pit_kosdaq/core")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", required=True)
    p.add_argument("--pause-seconds", type=float, default=1.2)
    p.add_argument("--pykrx-fallback-pause", type=float, default=0.5)
    return p.parse_args()


def main():
    a=parse_args()
    start=pd.Timestamp(a.start).normalize()
    end=pd.Timestamp(a.end).normalize()

    u=pd.read_csv(a.universe,dtype={"ticker":str})
    u["ticker"]=u["ticker"].astype(str).str.zfill(6)

    m=pd.read_parquet(a.membership)
    m["ticker"]=m["ticker"].astype(str).str.zfill(6)
    m["date"]=pd.to_datetime(m["date"]).dt.normalize()
    m=m[(m["date"]>=start)&(m["date"]<=end)].copy()

    cache=Path(a.core_root)/"cache"/"market"
    cache.mkdir(parents=True,exist_ok=True)

    kis=KisOpenApiClient.from_env(pause_seconds=a.pause_seconds)
    kp=KisMarketProvider(client=kis)
    pp=PykrxOhlcvOnlyProvider(pause_seconds=a.pykrx_fallback_pause)

    groups={t:g.sort_values("date") for t,g in m.groupby("ticker")}
    rows=[]

    tickers=[t for t in u["ticker"].drop_duplicates().tolist() if t in groups]
    for i,t in enumerate(tickers,1):
        mg=groups[t]
        t0=max(start,mg["date"].min())
        t1=min(end,mg["date"].max())
        p=cache/f"{t}.parquet"

        old=None
        if p.exists():
            try:
                old=pd.read_parquet(p)
                old["date"]=pd.to_datetime(old["date"]).dt.normalize()
            except Exception:
                old=None

        if old is not None and not old.empty:
            fetch_start=max(t0,old["date"].max()+pd.Timedelta(days=1))
        else:
            fetch_start=t0

        if fetch_start>t1:
            rows.append({"ticker":t,"status":"up_to_date","from":None,"to":None,"added":0})
            continue

        print(f"{i}/{len(tickers)} {t} {fetch_start.date()}..{t1.date()}")

        try:
            new=kp.daily_stock_ohlcv(t,fetch_start,t1)
            new["market_provider"]="KIS"
            provider="KIS"
        except Exception as e1:
            try:
                new=pp.daily_stock_ohlcv(t,fetch_start,t1)
                new["market_provider"]="pykrx_ohlcv_fallback"
                provider="pykrx"
            except Exception as e2:
                rows.append({"ticker":t,"status":"fail","from":fetch_start,"to":t1,"added":0,"detail":f"KIS={e1}; pykrx={e2}"})
                print(f"[WARN] {t}: {e2}")
                continue

        if new is None or new.empty:
            rows.append({"ticker":t,"status":"empty","from":fetch_start,"to":t1,"added":0,"detail":provider})
            continue

        new["ticker"]=new["ticker"].astype(str).str.zfill(6)
        new["date"]=pd.to_datetime(new["date"]).dt.normalize()

        if old is not None and not old.empty:
            out=pd.concat([old,new],ignore_index=True,sort=False)
        else:
            out=new.copy()

        out=(out.sort_values("date")
                .drop_duplicates(["ticker","date"],keep="last")
                .reset_index(drop=True))
        atomic_write_parquet(out,p)
        rows.append({"ticker":t,"status":"ok","from":fetch_start,"to":t1,"added":len(new),"detail":provider})

    q=Path(a.core_root)/"quality"
    q.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(q/"incremental_market_update_status.csv",index=False,encoding="utf-8-sig")

    s=pd.DataFrame(rows)
    print("\n=== INCREMENTAL MARKET UPDATE ===")
    print("tickers",len(s))
    print(s["status"].value_counts(dropna=False).to_string())
    print("added_rows",int(pd.to_numeric(s.get("added",0),errors="coerce").fillna(0).sum()))
    print("wrote ->",q/"incremental_market_update_status.csv")

if __name__=="__main__":
    main()
