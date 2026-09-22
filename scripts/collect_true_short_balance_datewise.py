#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from pykrxauth import stock
from pykrxauth.website.comm.session import login


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet")
    p.add_argument("--c5-dates", default="data/pit_kosdaq/true_short_balance/c5_candidate_dates.csv")
    p.add_argument("--out-dir", default="data/pit_kosdaq/true_short_balance")
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "korea-stock-true-short-balance-datewise-v1"))
    p.add_argument("--lookback-calendar-days", type=int, default=30)
    p.add_argument("--pause", type=float, default=0.35)
    p.add_argument("--retries", type=int, default=4)
    return p.parse_args()


def normalize_market_balance(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy().reset_index()
    ticker_col = None
    for c in ["티커", "종목코드", "ISU_SRT_CD", "index"]:
        if c in d.columns:
            ticker_col = c
            break
    if ticker_col is None:
        ticker_col = d.columns[0]

    d["ticker"] = d[ticker_col].astype(str).str.extract(r"(\d{6})", expand=False)

    mapping = {
        "공매도잔고": "short_balance_shares",
        "상장주식수": "listed_shares_krx",
        "공매도금액": "short_balance_value",
        "시가총액": "market_cap_krx",
        "비중": "short_balance_ratio_listed_api",
    }

    for src, dst in mapping.items():
        if src in d.columns:
            d[dst] = pd.to_numeric(
                d[src].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

    keep = ["ticker"] + [dst for dst in mapping.values() if dst in d.columns]
    out = d[keep].dropna(subset=["ticker"]).copy()
    out["ticker"] = out["ticker"].str.zfill(6)
    return out


def fetch_date(date: pd.Timestamp, retries: int, pause: float) -> pd.DataFrame:
    ds = pd.Timestamp(date).strftime("%Y%m%d")
    last = None
    for attempt in range(retries):
        try:
            raw = stock.get_shorting_balance_by_ticker(ds, market="KOSDAQ")
            out = normalize_market_balance(raw)
            time.sleep(pause)
            return out
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"KOSDAQ short-balance by ticker failed {ds}: {last!r}")


def main():
    args = parse_args()

    if not login():
        raise SystemExit("KRX login failed. Check KRX_ID / KRX_PW.")

    model = pd.read_parquet(args.model)
    model["ticker"] = model["ticker"].astype(str).str.zfill(6)
    model["date"] = pd.to_datetime(model["date"])

    c5_path = Path(args.c5_dates)
    if not c5_path.exists():
        raise SystemExit(f"Missing {c5_path}")

    c5 = pd.read_csv(c5_path, dtype={"ticker": str})
    c5["ticker"] = c5["ticker"].astype(str).str.zfill(6)
    c5["date"] = pd.to_datetime(c5["date"])

    target_tickers = sorted(c5["ticker"].unique().tolist())
    start = c5["date"].min() - pd.Timedelta(days=args.lookback_calendar_days)
    end = c5["date"].max()

    trading_dates = (
        model.loc[(model["date"] >= start) & (model["date"] <= end), "date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=== DATEWISE TRUE DTC COLLECTION TARGET ===")
    print("target_tickers", len(target_tickers))
    print("trading_dates", len(trading_dates))
    print("start", pd.Timestamp(start).date())
    print("end", pd.Timestamp(end).date())

    errors = []
    successful_market_dates = 0

    for i, date in enumerate(trading_dates, 1):
        date = pd.Timestamp(date)
        ds = date.strftime("%Y%m%d")
        p = cache_dir / f"{ds}.parquet"

        if p.exists():
            try:
                old = pd.read_parquet(p)
                if not old.empty:
                    successful_market_dates += 1
                    if i % 25 == 0 or i == 1 or i == len(trading_dates):
                        print(f"{i}/{len(trading_dates)} {ds} cache rows={len(old)}")
                    continue
            except Exception:
                pass

        try:
            market = fetch_date(date, args.retries, args.pause)
            if market.empty:
                errors.append({"date": ds, "error": "whole KOSDAQ balance endpoint returned empty"})
                print(f"[EMPTY MARKET] {i}/{len(trading_dates)} {ds}")
                continue

            successful_market_dates += 1

            base = pd.DataFrame({"ticker": target_tickers})
            base["date"] = date
            x = base.merge(market, on="ticker", how="left")

            x["is_reported_balance"] = x["short_balance_shares"].notna()

            for c in ["short_balance_shares", "short_balance_value", "short_balance_ratio_listed_api"]:
                if c in x.columns:
                    x[c] = x[c].fillna(0.0)

            x.to_parquet(p, index=False)

            if i % 10 == 0 or i == 1 or i == len(trading_dates):
                nrep = int(x["is_reported_balance"].sum())
                print(f"{i}/{len(trading_dates)} {ds} market_rows={len(market)} target_reported={nrep}/{len(target_tickers)}")

        except Exception as exc:
            errors.append({"date": ds, "error": repr(exc)})
            print(f"[WARN] {i}/{len(trading_dates)} {ds}: {exc}")

    frames = []
    for p in sorted(cache_dir.glob("*.parquet")):
        try:
            d = pd.read_parquet(p)
            if not d.empty:
                frames.append(d)
        except Exception:
            pass

    if not frames:
        raise SystemExit("No datewise KOSDAQ short-balance data collected.")

    panel = pd.concat(frames, ignore_index=True)
    panel["ticker"] = panel["ticker"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = (
        panel.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    out_file = out_dir / "true_short_balance_daily.parquet"
    panel.to_parquet(out_file, index=False)
    panel.to_csv(out_dir / "true_short_balance_daily.csv", index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "datewise_collection_errors.csv", index=False)

    event = c5.merge(
        panel[["ticker", "date", "short_balance_shares", "is_reported_balance"]],
        on=["ticker", "date"],
        how="left",
    )
    event["year"] = event["date"].dt.year

    rows = []
    for sample, g in [("full", event)] + [(str(y), z) for y, z in event.groupby("year")]:
        rows.append({
            "sample": sample,
            "c5_rows": len(g),
            "panel_coverage": float(g["short_balance_shares"].notna().mean()),
            "reported_balance_rate": float(g["is_reported_balance"].fillna(False).mean()),
            "positive_reported_balance_rate": float((g["short_balance_shares"].fillna(0) > 0).mean()),
        })

    diag = pd.DataFrame(rows)
    diag.to_csv(out_dir / "true_short_balance_event_diagnostic.csv", index=False)

    meta = {
        "target_tickers": len(target_tickers),
        "trading_dates": len(trading_dates),
        "successful_market_dates": successful_market_dates,
        "panel_rows": len(panel),
        "reported_rows": int(panel["is_reported_balance"].sum()),
        "errors": len(errors),
        "cache_dir": str(cache_dir),
    }
    (out_dir / "datewise_collection_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== DATEWISE TRUE SHORT BALANCE EVENT DIAGNOSTIC ===")
    print(diag.to_string(index=False))
    print("\n=== META ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\nwrote ->", out_file)


if __name__ == "__main__":
    main()
