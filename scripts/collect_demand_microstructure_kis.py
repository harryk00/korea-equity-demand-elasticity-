#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_URL = "/oauth2/tokenP"

PROGRAM_URL = "/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"
PROGRAM_TR_ID = "FHPPG04650201"

DAILY_TRADE_VOLUME_URL = "/uapi/domestic-stock/v1/quotations/inquire-daily-trade-volume"
DAILY_TRADE_VOLUME_TR_ID = "FHKST03010800"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Collect KIS program-trading and daily buy/sell execution-volume data "
            "only for strict-causal C5 candidate tickers/dates."
        )
    )
    p.add_argument(
        "--input",
        default="data/pit_kosdaq/core/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/demand_microstructure",
    )
    p.add_argument(
        "--cache-dir",
        default=str(Path.home() / ".cache" / "korea-stock-demand-microstructure-v1"),
    )
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--retries", type=int, default=5)
    p.add_argument(
        "--execution-lookback-days",
        type=int,
        default=14,
        help="Calendar-day range fetched before each missing signal date.",
    )
    p.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Debug only. 0 = all C5 tickers.",
    )
    p.add_argument(
        "--only",
        choices=["all", "program", "execution"],
        default="all",
    )
    return p.parse_args()


def qbucket(pct: pd.Series, q: int = 5) -> pd.Series:
    return np.ceil(pct * q).clip(1, q).astype("Int64")


def extract_c5_candidates(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["ticker", "date"]).copy()
    x["prev_close"] = x.groupby("ticker")["close"].shift(1)
    x["ret_1d"] = x["close"] / x["prev_close"] - 1.0

    x["float_mcap_pct"] = (
        x.groupby("date")["free_float_market_cap"]
        .rank(method="average", pct=True, ascending=True)
    )
    x["turnover_pct"] = (
        x.groupby("date")["float_turnover_1d"]
        .rank(method="average", pct=True, ascending=True)
    )

    x["base_signal"] = (
        (x["float_mcap_pct"] <= 0.20)
        & (x["turnover_pct"] >= 0.80)
    )

    sig = x["base_signal"].fillna(False)

    for col, out in [
        ("float_turnover_1d", "turn_signal_pct"),
        ("free_float_market_cap", "float_signal_pct"),
        ("ret_1d", "ret1_signal_pct"),
    ]:
        x[out] = np.nan
        ranked = (
            x.loc[sig]
            .groupby("date")[col]
            .rank(method="average", pct=True, ascending=True)
        )
        x.loc[ranked.index, out] = ranked

    x["turn_signal_q"] = qbucket(x["turn_signal_pct"])
    x["float_signal_q"] = qbucket(x["float_signal_pct"])
    x["ret1_signal_q"] = qbucket(x["ret1_signal_pct"])

    x["c5"] = (
        x["base_signal"]
        & (x["turn_signal_q"] <= 2)
        & (x["float_signal_q"] <= 2)
        & (x["ret1_signal_q"] < 5)
    )

    out = (
        x.loc[x["c5"], ["ticker", "date"]]
        .drop_duplicates()
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    return out


class KISClient:
    def __init__(
        self,
        app_key: str,
        app_secret: str,
        cache_dir: Path,
        pause: float,
        retries: int,
    ):
        self.app_key = app_key
        self.app_secret = app_secret
        self.cache_dir = cache_dir
        self.pause = pause
        self.retries = retries
        self.session = requests.Session()
        self.token_path = cache_dir / "token.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self.token = self._get_token()

    def _get_token(self) -> str:
        if self.token_path.exists():
            try:
                obj = json.loads(self.token_path.read_text(encoding="utf-8"))
                if float(obj.get("expires_at", 0)) > time.time() + 300:
                    return str(obj["access_token"])
            except Exception:
                pass

        url = BASE_URL + TOKEN_URL
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        r = self.session.post(
            url,
            headers={"content-type": "application/json"},
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        obj = r.json()

        token = obj.get("access_token")
        if not token:
            raise RuntimeError(f"KIS token error: {obj}")

        expires_in = obj.get("expires_in")
        try:
            expires_sec = float(expires_in)
        except Exception:
            expires_sec = 23 * 3600

        save = {
            "access_token": token,
            "expires_at": time.time() + max(3600, expires_sec - 300),
        }
        self.token_path.write_text(
            json.dumps(save, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return token

    def _throttle(self):
        elapsed = time.time() - self._last_request
        if elapsed < self.pause:
            time.sleep(self.pause - elapsed)

    def get(self, path: str, tr_id: str, params: dict) -> dict:
        url = BASE_URL + path

        for attempt in range(self.retries):
            self._throttle()

            headers = {
                "authorization": f"Bearer {self.token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": tr_id,
                "custtype": "P",
            }

            try:
                r = self.session.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=30,
                )
                self._last_request = time.time()

                # Refresh once on auth problem.
                if r.status_code == 401:
                    try:
                        self.token_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self.token = self._get_token()
                    continue

                r.raise_for_status()
                obj = r.json()

                if str(obj.get("rt_cd", "0")) == "0":
                    return obj

                msg = f"{obj.get('msg_cd')} {obj.get('msg1')}"
                # Retry temporary/rate-limit style responses.
                if attempt < self.retries - 1:
                    time.sleep(min(2 ** attempt, 16))
                    continue
                raise RuntimeError(f"KIS API error: {msg}; params={params}")

            except (requests.RequestException, ValueError) as e:
                self._last_request = time.time()
                if attempt >= self.retries - 1:
                    raise
                time.sleep(min(2 ** attempt, 16))

        raise RuntimeError("unreachable")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def normalize_program(rows: list[dict], ticker: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)
    if "stck_bsop_date" not in d.columns:
        return pd.DataFrame()

    d["date"] = pd.to_datetime(d["stck_bsop_date"], format="%Y%m%d", errors="coerce")
    d["ticker"] = ticker

    raw_cols = [
        "stck_clpr",
        "acml_vol",
        "acml_tr_pbmn",
        "whol_smtn_seln_vol",
        "whol_smtn_shnu_vol",
        "whol_smtn_ntby_qty",
        "whol_smtn_seln_tr_pbmn",
        "whol_smtn_shnu_tr_pbmn",
        "whol_smtn_ntby_tr_pbmn",
        "whol_ntby_vol_icdc",
        "whol_ntby_tr_pbmn_icdc2",
    ]
    for c in raw_cols:
        if c in d.columns:
            d[c] = to_num(d[c])

    rename = {
        "whol_smtn_seln_vol": "program_sell_qty",
        "whol_smtn_shnu_vol": "program_buy_qty",
        "whol_smtn_ntby_qty": "program_net_buy_qty",
        "whol_smtn_seln_tr_pbmn": "program_sell_value",
        "whol_smtn_shnu_tr_pbmn": "program_buy_value",
        "whol_smtn_ntby_tr_pbmn": "program_net_buy_value",
        "whol_ntby_vol_icdc": "program_net_buy_qty_change_api",
        "whol_ntby_tr_pbmn_icdc2": "program_net_buy_value_change_api",
    }
    d = d.rename(columns=rename)

    keep = [
        "ticker",
        "date",
        "program_sell_qty",
        "program_buy_qty",
        "program_net_buy_qty",
        "program_sell_value",
        "program_buy_value",
        "program_net_buy_value",
        "program_net_buy_qty_change_api",
        "program_net_buy_value_change_api",
    ]
    keep = [c for c in keep if c in d.columns]
    d = d[keep].dropna(subset=["date"])
    return d


def normalize_execution(rows: list[dict], ticker: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    d = pd.DataFrame(rows)
    if "stck_bsop_date" not in d.columns:
        return pd.DataFrame()

    d["date"] = pd.to_datetime(d["stck_bsop_date"], format="%Y%m%d", errors="coerce")
    d["ticker"] = ticker

    for c in ["total_seln_qty", "total_shnu_qty"]:
        if c in d.columns:
            d[c] = to_num(d[c])

    if not {"total_seln_qty", "total_shnu_qty"}.issubset(d.columns):
        return pd.DataFrame()

    d = d.rename(
        columns={
            "total_seln_qty": "sell_execution_qty",
            "total_shnu_qty": "buy_execution_qty",
        }
    )

    den = d["sell_execution_qty"].replace(0, np.nan)
    d["execution_strength"] = 100.0 * d["buy_execution_qty"] / den

    total = d["buy_execution_qty"] + d["sell_execution_qty"]
    d["execution_imbalance"] = (
        (d["buy_execution_qty"] - d["sell_execution_qty"])
        / total.replace(0, np.nan)
    )

    return d[
        [
            "ticker",
            "date",
            "sell_execution_qty",
            "buy_execution_qty",
            "execution_strength",
            "execution_imbalance",
        ]
    ].dropna(subset=["date"])


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def merge_cache(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        x = new.copy()
    elif new.empty:
        x = old.copy()
    else:
        x = pd.concat([old, new], ignore_index=True)

    if x.empty:
        return x

    x["ticker"] = x["ticker"].astype(str).str.zfill(6)
    x["date"] = pd.to_datetime(x["date"])
    return (
        x.drop_duplicates(["ticker", "date"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def program_has_date(cache: pd.DataFrame, date: pd.Timestamp) -> bool:
    if cache.empty:
        return False
    dates = set(pd.to_datetime(cache["date"]).dt.normalize())
    date = pd.Timestamp(date).normalize()
    # Need t and at least one earlier observation to construct acceleration.
    return date in dates and any(d < date for d in dates)


def execution_has_date(cache: pd.DataFrame, date: pd.Timestamp) -> bool:
    if cache.empty:
        return False
    dates = set(pd.to_datetime(cache["date"]).dt.normalize())
    date = pd.Timestamp(date).normalize()
    return date in dates and any(d < date for d in dates)


def fetch_program_for_anchor(
    client: KISClient,
    ticker: str,
    anchor: pd.Timestamp,
) -> tuple[pd.DataFrame, Optional[str]]:
    # KIS documentation examples for this endpoint have appeared with an
    # "002" prefix. Try that form first, then plain YYYYMMDD.
    ymd = pd.Timestamp(anchor).strftime("%Y%m%d")
    formats = [f"002{ymd}", ymd]

    last_error = None
    for base_date in formats:
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": ticker,
            "FID_INPUT_DATE_1": base_date,
        }
        try:
            obj = client.get(PROGRAM_URL, PROGRAM_TR_ID, params)
            rows = obj.get("output") or obj.get("output1") or []
            d = normalize_program(rows, ticker)
            if not d.empty:
                return d, base_date
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error
    return pd.DataFrame(), None


def fetch_execution_window(
    client: KISClient,
    ticker: str,
    anchor: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    end = pd.Timestamp(anchor)
    start = end - pd.Timedelta(days=lookback_days)

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": ticker,
        "FID_INPUT_DATE_1": start.strftime("%Y%m%d"),
        "FID_INPUT_DATE_2": end.strftime("%Y%m%d"),
        "FID_PERIOD_DIV_CODE": "D",
    }
    obj = client.get(
        DAILY_TRADE_VOLUME_URL,
        DAILY_TRADE_VOLUME_TR_ID,
        params,
    )
    rows = obj.get("output2") or obj.get("output") or []
    return normalize_execution(rows, ticker)


def coverage(candidate: pd.DataFrame, panel: pd.DataFrame, label: str) -> pd.DataFrame:
    c = candidate.copy()
    c["year"] = c["date"].dt.year
    if panel.empty:
        c["covered"] = False
    else:
        keys = set(
            zip(
                panel["ticker"].astype(str).str.zfill(6),
                pd.to_datetime(panel["date"]).dt.normalize(),
            )
        )
        c["covered"] = [
            (t, pd.Timestamp(d).normalize()) in keys
            for t, d in zip(c["ticker"], c["date"])
        ]

    rows = []
    for sample, g in [("full", c)] + [(str(y), g) for y, g in c.groupby("year")]:
        rows.append(
            {
                "panel": label,
                "sample": sample,
                "candidate_rows": len(g),
                "covered_rows": int(g["covered"].sum()),
                "coverage_rate": float(g["covered"].mean()) if len(g) else np.nan,
                "distinct_tickers": int(g["ticker"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()

    app_key = os.getenv("KIS_APP_KEY")
    app_secret = os.getenv("KIS_APP_SECRET")
    if not app_key or not app_secret:
        raise SystemExit(
            "Set KIS_APP_KEY and KIS_APP_SECRET before running."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path(args.cache_dir).expanduser()
    program_cache_dir = cache_root / "program"
    execution_cache_dir = cache_root / "execution"
    program_cache_dir.mkdir(parents=True, exist_ok=True)
    execution_cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input)
    req = [
        "ticker", "date", "close",
        "free_float_market_cap", "float_turnover_1d",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing input columns: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])

    candidates = extract_c5_candidates(df)
    candidates.to_csv(out_dir / "c5_candidate_dates.csv", index=False)

    tickers = candidates["ticker"].drop_duplicates().tolist()
    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]
        candidates = candidates[candidates["ticker"].isin(tickers)].copy()

    print("=== C5 COLLECTION TARGET ===")
    print("candidate_rows", len(candidates))
    print("distinct_tickers", candidates["ticker"].nunique())
    print("start", candidates["date"].min())
    print("end", candidates["date"].max())

    client = KISClient(
        app_key=app_key,
        app_secret=app_secret,
        cache_dir=cache_root,
        pause=args.pause,
        retries=args.retries,
    )

    errors = []
    program_formats_used = {}

    for i, ticker in enumerate(tickers, start=1):
        dates = (
            candidates.loc[candidates["ticker"] == ticker, "date"]
            .drop_duplicates()
            .sort_values(ascending=False)
            .tolist()
        )

        print(f"\n=== {i}/{len(tickers)} {ticker} candidate_dates={len(dates)} ===")

        if args.only in {"all", "program"}:
            ppath = program_cache_dir / f"{ticker}.parquet"
            pcache = load_cache(ppath)

            for date in dates:
                if program_has_date(pcache, date):
                    continue
                try:
                    new, fmt = fetch_program_for_anchor(client, ticker, date)
                    if fmt:
                        program_formats_used[fmt[:3] == "002" and "002YYYYMMDD" or "YYYYMMDD"] = (
                            program_formats_used.get(
                                fmt[:3] == "002" and "002YYYYMMDD" or "YYYYMMDD",
                                0,
                            )
                            + 1
                        )
                    pcache = merge_cache(pcache, new)
                    if not pcache.empty:
                        pcache.to_parquet(ppath, index=False)
                except Exception as e:
                    errors.append(
                        {
                            "panel": "program",
                            "ticker": ticker,
                            "date": pd.Timestamp(date),
                            "error": repr(e),
                        }
                    )
                    print(f"[WARN program] {ticker} {pd.Timestamp(date).date()}: {e}")

        if args.only in {"all", "execution"}:
            epath = execution_cache_dir / f"{ticker}.parquet"
            ecache = load_cache(epath)

            for date in dates:
                if execution_has_date(ecache, date):
                    continue
                try:
                    new = fetch_execution_window(
                        client,
                        ticker,
                        date,
                        args.execution_lookback_days,
                    )
                    ecache = merge_cache(ecache, new)
                    if not ecache.empty:
                        ecache.to_parquet(epath, index=False)
                except Exception as e:
                    errors.append(
                        {
                            "panel": "execution",
                            "ticker": ticker,
                            "date": pd.Timestamp(date),
                            "error": repr(e),
                        }
                    )
                    print(f"[WARN execution] {ticker} {pd.Timestamp(date).date()}: {e}")

    def collect_cache(dir_: Path) -> pd.DataFrame:
        frames = []
        for p in sorted(dir_.glob("*.parquet")):
            d = load_cache(p)
            if not d.empty:
                frames.append(d)
        if not frames:
            return pd.DataFrame()
        x = pd.concat(frames, ignore_index=True)
        x["ticker"] = x["ticker"].astype(str).str.zfill(6)
        x["date"] = pd.to_datetime(x["date"])
        return (
            x.drop_duplicates(["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )

    program_panel = collect_cache(program_cache_dir)
    execution_panel = collect_cache(execution_cache_dir)

    if not program_panel.empty:
        # Compute our own acceleration after collecting all available days.
        program_panel = program_panel.sort_values(["ticker", "date"])
        program_panel["program_net_buy_value_lag1"] = (
            program_panel.groupby("ticker")["program_net_buy_value"].shift(1)
        )
        program_panel["program_net_buy_accel_1d"] = (
            program_panel["program_net_buy_value"]
            - program_panel["program_net_buy_value_lag1"]
        )
        program_panel.to_parquet(out_dir / "program_daily.parquet", index=False)
        program_panel.to_csv(out_dir / "program_daily.csv", index=False, encoding="utf-8-sig")

    if not execution_panel.empty:
        execution_panel = execution_panel.sort_values(["ticker", "date"])
        execution_panel["execution_strength_lag1"] = (
            execution_panel.groupby("ticker")["execution_strength"].shift(1)
        )
        execution_panel["execution_strength_accel_1d"] = (
            execution_panel["execution_strength"]
            - execution_panel["execution_strength_lag1"]
        )
        execution_panel.to_parquet(out_dir / "execution_strength_daily.parquet", index=False)
        execution_panel.to_csv(
            out_dir / "execution_strength_daily.csv",
            index=False,
            encoding="utf-8-sig",
        )

    cov_frames = []
    cov_frames.append(coverage(candidates, program_panel, "program"))
    cov_frames.append(coverage(candidates, execution_panel, "execution_strength"))
    cov = pd.concat(cov_frames, ignore_index=True)
    cov.to_csv(out_dir / "coverage_summary.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "collection_errors.csv", index=False)

    meta = {
        "c5_candidate_rows": int(len(candidates)),
        "c5_distinct_tickers": int(candidates["ticker"].nunique()),
        "program_rows": int(len(program_panel)),
        "execution_rows": int(len(execution_panel)),
        "program_request_formats_used": program_formats_used,
        "cache_dir": str(cache_root),
        "errors": len(errors),
    }
    (out_dir / "collection_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== DEMAND MICROSTRUCTURE COVERAGE ===")
    print(cov.to_string(index=False))
    print("\n=== META ===")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
