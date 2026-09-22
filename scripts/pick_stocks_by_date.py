#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Final research-frozen main strategy
DEFAULT_THRESHOLDS = {
    "kq_ret20_q40_2025": 0.021656050955414008,
    "breadth_above_ma20_q40_2025": 0.3493981526355278,
    "kq_drawdown60_q40_2025": -0.03420061531932961,
}
MAX_POSITIONS = 10
TP = 0.30
SL = -0.15
MAX_HOLD = 20
SHORT_PUBLICATION_LAG_BARS = 2


def parse_args():
    p = argparse.ArgumentParser(
        description="Date -> frozen C5 + Market Weakness buy candidates"
    )
    p.add_argument("--date", help="Signal date YYYY-MM-DD. Use 'latest' for last model date.")
    p.add_argument(
        "--model",
        default="data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet",
    )
    p.add_argument(
        "--market-state",
        default="data/pit_kosdaq/analysis/market_state_v1/market_state_daily.parquet",
    )
    p.add_argument(
        "--market-thresholds",
        default="data/pit_kosdaq/analysis/portfolio_construction_v1/portfolio_market_thresholds_2025.json",
    )
    p.add_argument(
        "--short-balance",
        default="data/pit_kosdaq/true_short_balance/true_short_balance_daily.parquet",
    )
    p.add_argument(
        "--universe",
        default="data/pit_kosdaq/universe/pit_pipeline_universe.csv",
        help="Optional ticker-name mapping CSV.",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Optional account capital in KRW. Position sleeve remains capital/10.",
    )
    p.add_argument(
        "--adv-cap",
        type=float,
        default=0.005,
        help="Execution capacity cap as fraction of signal-date ADV20. Default 0.005 = 0.5%%.",
    )
    p.add_argument(
        "--out-dir",
        default="data/pit_kosdaq/signals/date_picker",
    )
    return p.parse_args()


def qbucket(s: pd.Series) -> pd.Series:
    return np.ceil(s * 5).clip(1, 5).astype("Int64")


def load_thresholds(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            needed = set(DEFAULT_THRESHOLDS)
            if needed.issubset(d):
                return {k: float(d[k]) for k in DEFAULT_THRESHOLDS}
        except Exception:
            pass
    return DEFAULT_THRESHOLDS.copy()


def build_c5(model: pd.DataFrame) -> pd.DataFrame:
    x = model.sort_values(["ticker", "date"]).copy()

    x["prev_close"] = x.groupby("ticker")["close"].shift(1)
    x["ret_1d"] = x["close"] / x["prev_close"] - 1

    # Cross-sectional PIT ranks on each date.
    x["float_mcap_pct"] = x.groupby("date")["free_float_market_cap"].rank(pct=True)
    x["turnover_pct"] = x.groupby("date")["float_turnover_1d"].rank(pct=True)

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
        r = (
            x.loc[sig]
            .groupby("date")[col]
            .rank(method="average", pct=True)
        )
        x.loc[r.index, out] = r

    x["turn_q"] = qbucket(x["turn_signal_pct"])
    x["float_q"] = qbucket(x["float_signal_pct"])
    x["ret1_q"] = qbucket(x["ret1_signal_pct"])

    x["c5"] = (
        x["base_signal"]
        & (x["turn_q"] <= 2)
        & (x["float_q"] <= 2)
        & (x["ret1_q"] < 5)
    )

    # Same priority used in portfolio construction:
    # smaller free float + less-overheated turnover within C5 is preferred.
    x["priority_score"] = (
        x["float_signal_pct"].fillna(1.0)
        + x["turn_signal_pct"].fillna(1.0)
    )

    if "trading_value" in x.columns:
        x["adv20_value_t"] = x.groupby("ticker")["trading_value"].transform(
            lambda z: z.rolling(20, min_periods=10).mean()
        )
    else:
        x["adv20_value_t"] = np.nan

    return x


def resolve_date(raw: str | None, model_dates: pd.Series) -> pd.Timestamp:
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(model_dates.dropna().unique())))
    if len(dates) == 0:
        raise SystemExit("Model contains no dates.")

    if raw is None:
        raw = input("신호 기준일을 입력하세요 (YYYY-MM-DD, 또는 latest): ").strip()

    if raw.lower() == "latest":
        return pd.Timestamp(dates[-1])

    try:
        d = pd.Timestamp(raw).normalize()
    except Exception:
        raise SystemExit("날짜 형식이 잘못되었습니다. 예: 2026-08-28")

    if d not in dates:
        prevs = dates[dates < d]
        nexts = dates[dates > d]
        msg = [f"{d.date()}는 모델의 거래일 데이터에 없습니다."]
        if len(prevs):
            msg.append(f"직전 데이터 거래일: {pd.Timestamp(prevs[-1]).date()}")
        if len(nexts):
            msg.append(f"다음 데이터 거래일: {pd.Timestamp(nexts[0]).date()}")
        raise SystemExit("\n".join(msg))

    return d


def next_market_date(signal_date: pd.Timestamp, model_dates: pd.Series):
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(model_dates.dropna().unique())))
    future = dates[dates > signal_date]
    return pd.Timestamp(future[0]) if len(future) else None


def add_market_gate(
    signal_date: pd.Timestamp,
    market: pd.DataFrame,
    thresholds: dict,
):
    row = market[market["date"] == signal_date]
    if row.empty:
        raise SystemExit(f"Market-state data missing for {signal_date.date()}.")

    r = row.iloc[0]
    checks = {
        "weak_trend": bool(
            pd.notna(r.get("kq_ret20"))
            and r["kq_ret20"] <= thresholds["kq_ret20_q40_2025"]
        ),
        "weak_breadth": bool(
            pd.notna(r.get("breadth_above_ma20"))
            and r["breadth_above_ma20"]
            <= thresholds["breadth_above_ma20_q40_2025"]
        ),
        "deep_drawdown": bool(
            pd.notna(r.get("kq_drawdown60"))
            and r["kq_drawdown60"]
            <= thresholds["kq_drawdown60_q40_2025"]
        ),
    }
    score = sum(checks.values())
    return r, checks, score, score >= 2


def load_name_map(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        u = pd.read_csv(p, dtype=str)
    except Exception:
        return {}

    ticker_col = None
    for c in ["ticker", "code", "stock_code", "종목코드"]:
        if c in u.columns:
            ticker_col = c
            break

    name_col = None
    for c in ["name", "stock_name", "corp_name", "company_name", "종목명", "한글종목명"]:
        if c in u.columns:
            name_col = c
            break

    if ticker_col is None or name_col is None:
        return {}

    u[ticker_col] = u[ticker_col].astype(str).str.zfill(6)
    return dict(zip(u[ticker_col], u[name_col]))


def load_lagged_short(path: str, signal_date: pd.Timestamp) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["ticker", "reported_short_balance_shares"])

    s = pd.read_parquet(
        p,
        columns=["ticker", "date", "short_balance_shares"],
    )
    s["ticker"] = s["ticker"].astype(str).str.zfill(6)
    s["date"] = pd.to_datetime(s["date"])
    s = s.sort_values(["ticker", "date"])
    s["reported_short_balance_shares"] = (
        s.groupby("ticker")["short_balance_shares"]
        .shift(SHORT_PUBLICATION_LAG_BARS)
    )
    return s.loc[
        s["date"] == signal_date,
        ["ticker", "reported_short_balance_shares"],
    ].copy()


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.{digits}f}%"


def fmt_krw(x):
    if pd.isna(x):
        return "N/A"
    return f"{int(round(x)):,}원"


def main():
    a = parse_args()

    model_path = Path(a.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {model_path}")

    required = [
        "ticker", "date", "close",
        "free_float_market_cap", "float_turnover_1d",
    ]
    optional = ["trading_value"]

    import pyarrow.parquet as pq
    schema_cols = set(pq.ParquetFile(model_path).schema_arrow.names)
    missing = [c for c in required if c not in schema_cols]
    if missing:
        raise SystemExit(f"Missing model columns: {missing}")

    cols = required + [c for c in optional if c in schema_cols]
    print("모델 데이터 읽는 중:", cols)
    model = pd.read_parquet(model_path, columns=cols)
    model["ticker"] = model["ticker"].astype(str).str.zfill(6)
    model["date"] = pd.to_datetime(model["date"])

    signal_date = resolve_date(a.date, model["date"])
    buy_date = next_market_date(signal_date, model["date"])

    market = pd.read_parquet(a.market_state)
    market["date"] = pd.to_datetime(market["date"])
    thresholds = load_thresholds(a.market_thresholds)

    market_row, checks, market_score, market_ok = add_market_gate(
        signal_date, market, thresholds
    )

    print("\n" + "="*72)
    print("FROZEN STRATEGY SIGNAL")
    print("="*72)
    print("신호 기준일 :", signal_date.date())
    print("예상 매수일 :", buy_date.date() if buy_date is not None else "모델 범위 밖")
    print("매수 시점   : 다음 거래일 시가(09:00)")
    print("익절/손절   : +30% / -15%")
    print("최대 보유   : 20 거래일")
    print("최대 종목수 :", MAX_POSITIONS)

    print("\n[시장 상태]")
    print(
        f"KOSDAQ 20d return = {fmt_pct(market_row.get('kq_ret20'))} "
        f"(기준 <= {fmt_pct(thresholds['kq_ret20_q40_2025'])}) "
        f"=> {'YES' if checks['weak_trend'] else 'NO'}"
    )
    print(
        f"MA20 breadth      = {fmt_pct(market_row.get('breadth_above_ma20'))} "
        f"(기준 <= {fmt_pct(thresholds['breadth_above_ma20_q40_2025'])}) "
        f"=> {'YES' if checks['weak_breadth'] else 'NO'}"
    )
    print(
        f"60d drawdown      = {fmt_pct(market_row.get('kq_drawdown60'))} "
        f"(기준 <= {fmt_pct(thresholds['kq_drawdown60_q40_2025'])}) "
        f"=> {'YES' if checks['deep_drawdown'] else 'NO'}"
    )
    print(f"Market Weak Score = {market_score}/3")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not market_ok:
        print("\n[결론] 오늘은 MARKET 조건이 2/3 미만이므로 매수하지 않습니다.")
        no_buy = pd.DataFrame([{
            "signal_date": signal_date,
            "buy_date": buy_date,
            "market_weak_score": market_score,
            "action": "NO_BUY_MARKET_FILTER",
        }])
        out_file = out_dir / f"signal_{signal_date:%Y%m%d}.csv"
        no_buy.to_csv(out_file, index=False, encoding="utf-8-sig")
        print("저장 ->", out_file)
        return

    print("\nC5 계산 중...")
    x = build_c5(model)
    day = x[(x["date"] == signal_date) & x["c5"]].copy()

    if day.empty:
        print("\n[결론] MARKET 조건은 통과했지만 C5 매수 후보가 없습니다.")
        no_buy = pd.DataFrame([{
            "signal_date": signal_date,
            "buy_date": buy_date,
            "market_weak_score": market_score,
            "action": "NO_BUY_NO_C5",
        }])
        out_file = out_dir / f"signal_{signal_date:%Y%m%d}.csv"
        no_buy.to_csv(out_file, index=False, encoding="utf-8-sig")
        print("저장 ->", out_file)
        return

    day = day.sort_values(
        ["priority_score", "float_signal_pct", "turn_signal_pct", "ticker"]
    ).head(MAX_POSITIONS).copy()

    name_map = load_name_map(a.universe)
    day["name"] = day["ticker"].map(name_map)

    short = load_lagged_short(a.short_balance, signal_date)
    day = day.merge(short, on="ticker", how="left")
    day["short_overhang_flag"] = np.where(
        day["reported_short_balance_shares"].isna(),
        "UNKNOWN",
        np.where(day["reported_short_balance_shares"] > 0, "YES", "NO"),
    )

    day["rank"] = np.arange(1, len(day)+1)
    day["signal_date"] = signal_date
    day["buy_date"] = buy_date
    day["market_weak_score"] = market_score
    day["tp_pct"] = TP
    day["sl_pct"] = SL
    day["max_hold_days"] = MAX_HOLD
    day["adv_capacity_krw"] = day["adv20_value_t"] * a.adv_cap

    if a.capital is not None:
        sleeve = float(a.capital) / MAX_POSITIONS
        day["account_sleeve_krw"] = sleeve
        day["planned_order_krw"] = np.minimum(
            sleeve,
            day["adv_capacity_krw"].fillna(sleeve),
        )
        day["planned_cash_if_full"] = float(a.capital) - day["planned_order_krw"].sum()

    show_cols = [
        "rank", "ticker", "name", "close",
        "free_float_market_cap", "float_turnover_1d",
        "ret_1d", "priority_score",
        "adv20_value_t", "adv_capacity_krw",
        "short_overhang_flag",
    ]
    if a.capital is not None:
        show_cols += ["planned_order_krw"]

    print("\n[매수 후보]")
    print(f"총 {len(day)}개 — 다음 거래일 시가 매수 후보")
    print("-"*72)

    for r in day[show_cols].itertuples(index=False):
        nm = getattr(r, "name")
        label = f"{r.ticker} {nm}" if pd.notna(nm) and str(nm).strip() else r.ticker
        print(
            f"#{r.rank:>2} {label:<20} "
            f"종가 {getattr(r,'close'):>10,.0f} | "
            f"turnover {fmt_pct(getattr(r,'float_turnover_1d')):>8} | "
            f"당일 {fmt_pct(getattr(r,'ret_1d')):>8} | "
            f"Short {getattr(r,'short_overhang_flag')}"
        )
        print(
            f"     ADV20 {fmt_krw(getattr(r,'adv20_value_t'))} | "
            f"{a.adv_cap*100:.2f}% ADV 한도 {fmt_krw(getattr(r,'adv_capacity_krw'))}"
            + (
                f" | 계획 주문 {fmt_krw(getattr(r,'planned_order_krw'))}"
                if a.capital is not None else ""
            )
        )

    print("\n[실행 규칙]")
    print("1) 이 신호는 신호일 장 마감 후에만 확정")
    print("2) 다음 거래일 08:30~09:00 동시호가에 주문 제출")
    print("3) 시가 체결을 목표로 하고, 미체결 수량은 추격매수하지 않음")
    print("4) 체결가 기준 +30% 익절 / -15% 손절")
    print("5) 둘 다 미도달하면 20번째 거래일 종가 청산")
    print("6) Short YES는 연구상 위험표시일 뿐 현재 hard exclusion은 아님")

    save_cols = [
        "rank","signal_date","buy_date","ticker","name","close",
        "free_float_market_cap","float_turnover_1d","ret_1d",
        "float_signal_pct","turn_signal_pct","ret1_signal_pct","priority_score",
        "market_weak_score","adv20_value_t","adv_capacity_krw",
        "reported_short_balance_shares","short_overhang_flag",
        "tp_pct","sl_pct","max_hold_days",
    ]
    if a.capital is not None:
        save_cols += ["account_sleeve_krw","planned_order_krw","planned_cash_if_full"]

    out_file = out_dir / f"signal_{signal_date:%Y%m%d}.csv"
    day[save_cols].to_csv(out_file, index=False, encoding="utf-8-sig")

    print("\n저장 ->", out_file)
    if buy_date is None:
        print("\n주의: 신호일 이후 거래일이 모델 데이터에 없어 실제 매수일은 계산할 수 없습니다.")
    print("\n주의: 모델 데이터의 마지막 날짜 이후를 조회하려면 먼저 일별 데이터/market-state를 업데이트해야 합니다.")


if __name__ == "__main__":
    main()
