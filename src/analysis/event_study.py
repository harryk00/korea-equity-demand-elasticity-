"""
이벤트 스터디 엔진.

지수 편입/편출을 외생적 수요 충격으로 사용하여 수요탄력성(1/zeta)을 추정한다.

가설 (docs/01_hypothesis.md 참조):
    H1: beta0 > 0   수요곡선 우하향
    H2: beta1 < 0   유통비율이 높을수록 증폭이 작다 (희소성 증폭)
    H3:             장기 되돌림 여부

설계 원칙:
    - 추정구간과 이벤트구간을 엄격히 분리한다 (룩어헤드 방지).
    - 이벤트 날짜가 겹치므로 횡단면 상관을 클러스터 표준오차로 처리한다.
    - Shock의 절대 크기는 AUM 추정에 의존하므로, 순위 기반 강건성을 병기한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------------------
# 이벤트 정의
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IndexEvent:
    """지수 편입/편출 이벤트 하나.

    Attributes
    ----------
    ticker : 종목코드 (6자리)
    announcement_date : 발표일 (AD)
    effective_date : 실행일 (ED)
    index_name : 지수명 (KOSPI200, MSCI_KOREA, FTSE_GEIS 등)
    direction : +1 편입, -1 편출
    target_weight : 편입 시 지수 내 목표 비중 (0~1). 편출이면 편출 전 비중.
    """

    ticker: str
    announcement_date: pd.Timestamp
    effective_date: pd.Timestamp
    index_name: str
    direction: int
    target_weight: float

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")
        if self.effective_date < self.announcement_date:
            raise ValueError(
                f"{self.ticker}: effective_date precedes announcement_date"
            )


# 사전등록된 이벤트 윈도우 (docs/01_hypothesis.md 5.1)
# (시작 앵커, 시작 오프셋, 종료 앵커, 종료 오프셋)
WINDOWS: dict[str, tuple[str, int, str, int]] = {
    "pre":          ("AD", -20, "AD",  -1),
    "announcement": ("AD",  -1, "AD",  +1),
    "runup":        ("AD",  +1, "ED",  -1),
    "effective":    ("ED",  -1, "ED",  +1),
    "post":         ("ED",  +1, "ED", +60),
    # 주 검정 윈도우 — 사전 고정
    "main":         ("AD",  -1, "ED",  +1),
}

PRIMARY_WINDOW = "main"


# --------------------------------------------------------------------------
# 수요 충격
# --------------------------------------------------------------------------

def compute_shock(
    events: Sequence[IndexEvent],
    free_float: pd.Series,
    price: pd.Series,
    tracking_aum: dict[str, float],
) -> pd.DataFrame:
    """예상 수요 충격을 유통주식수 대비로 계산한다.

        Shock = direction * (weight * AUM / price) / FreeFloat

    Parameters
    ----------
    free_float : 종목코드 -> 유통가능주식수 (이벤트 시점 기준)
    price : 종목코드 -> 발표일 전일 종가
    tracking_aum : 지수명 -> 추종 자금 규모 (원)

    Notes
    -----
    tracking_aum은 정확한 공개치가 없어 추정값이다. 따라서 Shock의
    절대 크기는 신뢰하지 않고, 종목 간 상대 순위에 의존한다
    (docs/01_hypothesis.md 7.1). 순위 기반 강건성 점검을 위해
    shock_rank를 함께 반환한다.
    """
    rows = []
    for ev in events:
        if ev.ticker not in free_float.index or ev.ticker not in price.index:
            continue
        ff = free_float[ev.ticker]
        px = price[ev.ticker]
        aum = tracking_aum.get(ev.index_name)
        if aum is None or ff <= 0 or px <= 0:
            continue

        demanded_shares = ev.target_weight * aum / px
        rows.append(
            {
                "ticker": ev.ticker,
                "index_name": ev.index_name,
                "announcement_date": ev.announcement_date,
                "effective_date": ev.effective_date,
                "direction": ev.direction,
                "shock": ev.direction * demanded_shares / ff,
                "free_float": ff,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # 순위 기반 강건성용 (AUM 추정 불확실성에 둔감)
    df["shock_rank"] = df["shock"].rank(pct=True)
    return df


# --------------------------------------------------------------------------
# 초과수익률
# --------------------------------------------------------------------------

def _market_model_params(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    est_start: pd.Timestamp,
    est_end: pd.Timestamp,
    min_obs: int = 120,
) -> tuple[float, float] | None:
    """추정구간에서 시장모형 (alpha, beta)를 OLS로 적합한다.

    추정구간은 이벤트 윈도우와 겹치지 않아야 한다 (룩어헤드 방지).
    """
    mask = (stock_ret.index >= est_start) & (stock_ret.index <= est_end)
    y = stock_ret.loc[mask].dropna()
    x = market_ret.reindex(y.index).dropna()
    y = y.reindex(x.index)

    if len(y) < min_obs:
        return None

    X = sm.add_constant(x.values)
    res = sm.OLS(y.values, X).fit()
    return float(res.params[0]), float(res.params[1])


def compute_car(
    returns: pd.DataFrame,
    market_ret: pd.Series,
    events: Sequence[IndexEvent],
    window: str = PRIMARY_WINDOW,
    est_offset_start: int = -250,
    est_offset_end: int = -30,
    method: str = "market_model",
) -> pd.DataFrame:
    """이벤트별 누적초과수익률(CAR)을 계산한다.

    Parameters
    ----------
    returns : DatetimeIndex x 종목코드 일별 수익률
    market_ret : 시장 수익률 (KOSPI)
    window : WINDOWS의 키
    method : 'market_model' 또는 'market_adjusted'
             강건성 점검을 위해 두 방식 모두 산출할 것
             (docs/01_hypothesis.md 6절 기각조건 3)

    Returns
    -------
    ticker, event_date, car, n_days, alpha, beta
    """
    if window not in WINDOWS:
        raise KeyError(f"unknown window {window!r}; choose from {list(WINDOWS)}")

    start_anchor, start_off, end_anchor, end_off = WINDOWS[window]
    cal = returns.index
    rows = []

    for ev in events:
        if ev.ticker not in returns.columns:
            continue

        anchors = {"AD": ev.announcement_date, "ED": ev.effective_date}

        try:
            i_start = _shift_trading_day(cal, anchors[start_anchor], start_off)
            i_end = _shift_trading_day(cal, anchors[end_anchor], end_off)
            i_ad = cal.searchsorted(ev.announcement_date)
        except (IndexError, ValueError):
            continue
        if i_start is None or i_end is None or i_end < i_start:
            continue

        stock_ret = returns[ev.ticker]

        alpha = beta = np.nan
        if method == "market_model":
            i_est_s = max(0, i_ad + est_offset_start)
            i_est_e = i_ad + est_offset_end
            if i_est_e <= i_est_s:
                continue
            params = _market_model_params(
                stock_ret, market_ret, cal[i_est_s], cal[i_est_e]
            )
            if params is None:
                continue
            alpha, beta = params
            expected = alpha + beta * market_ret.iloc[i_start : i_end + 1]
        elif method == "market_adjusted":
            expected = market_ret.iloc[i_start : i_end + 1]
        else:
            raise ValueError(f"unknown method {method!r}")

        actual = stock_ret.iloc[i_start : i_end + 1]
        abnormal = (actual - expected).dropna()
        if abnormal.empty:
            continue

        rows.append(
            {
                "ticker": ev.ticker,
                "index_name": ev.index_name,
                "announcement_date": ev.announcement_date,
                "effective_date": ev.effective_date,
                # 클러스터링 키: 같은 날 발표된 이벤트는 상관됨
                "event_cluster": ev.announcement_date,
                "car": float(abnormal.sum()),
                "n_days": int(len(abnormal)),
                "alpha": alpha,
                "beta": beta,
            }
        )

    return pd.DataFrame(rows)


def _shift_trading_day(
    calendar: pd.DatetimeIndex, anchor: pd.Timestamp, offset: int
) -> int | None:
    """거래일 기준으로 anchor에서 offset일 이동한 위치 인덱스를 반환한다."""
    pos = calendar.searchsorted(anchor)
    if pos >= len(calendar):
        return None
    target = pos + offset
    if target < 0 or target >= len(calendar):
        return None
    return int(target)


# --------------------------------------------------------------------------
# 횡단면 회귀 (H1, H2)
# --------------------------------------------------------------------------

def run_cross_sectional(
    car_df: pd.DataFrame,
    shock_df: pd.DataFrame,
    characteristics: pd.DataFrame,
    interactions: Sequence[str] = ("ff_ratio", "log_adv"),
    controls: Sequence[str] = ("log_mktcap", "ret_60d", "vol_20d"),
    use_rank_shock: bool = False,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """H1/H2 검정을 위한 횡단면 회귀.

        CAR = a + b0*Shock + b1*(Shock x FF) + b2*(Shock x logADV) + g'X + e

    검정 대상:
        b0 > 0  (H1) 수요곡선 우하향
        b1 < 0  (H2) 유통비율이 높을수록 증폭 작음  <- 핵심

    표준오차는 이벤트 날짜 클러스터 기준 (docs/01_hypothesis.md 5.4).
    같은 날 발표된 이벤트들은 공통 충격을 공유하므로 독립이 아니다.

    Parameters
    ----------
    use_rank_shock : True이면 shock 대신 shock_rank 사용.
                     AUM 추정 불확실성에 대한 강건성 점검.
    """
    df = car_df.merge(
        shock_df, on=["ticker", "announcement_date"], how="inner",
        suffixes=("", "_shk"),
    )
    df = df.merge(characteristics, on="ticker", how="inner")

    shock_col = "shock_rank" if use_rank_shock else "shock"
    df = df.dropna(subset=["car", shock_col, *interactions, *controls])

    if len(df) < 30:
        raise ValueError(
            f"표본 부족: {len(df)}건. 사전등록 기각조건 2 (n<100) 확인 필요."
        )

    X = pd.DataFrame(index=df.index)
    X["shock"] = df[shock_col]

    # 상호작용항은 중심화하여 주효과 해석을 보존한다
    for var in interactions:
        centered = df[var] - df[var].mean()
        X[f"shock_x_{var}"] = df[shock_col] * centered
        X[var] = centered

    for var in controls:
        X[var] = df[var]

    X = sm.add_constant(X)
    model = sm.OLS(df["car"].values, X.values)

    return model.fit(
        cov_type="cluster",
        cov_kwds={"groups": df["event_cluster"].values},
    ), list(X.columns), df


def summarize(result, colnames: Sequence[str]) -> pd.DataFrame:
    """회귀 결과를 가설 검정 관점으로 정리한다."""
    out = pd.DataFrame(
        {
            "coef": result.params,
            "std_err": result.bse,
            "t": result.tvalues,
            "p": result.pvalues,
        },
        index=colnames,
    )

    hypothesis = {}
    for name in colnames:
        if name == "shock":
            hypothesis[name] = "H1: >0"
        elif name == "shock_x_ff_ratio":
            hypothesis[name] = "H2: <0  <<< 핵심"
        else:
            hypothesis[name] = ""
    out["hypothesis"] = pd.Series(hypothesis)
    return out
