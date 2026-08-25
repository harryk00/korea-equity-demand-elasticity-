"""
합성 데이터 검증.

정답(true beta)을 아는 데이터를 생성하여 엔진이 그것을 회수하는지 확인한다.
실데이터에 적용하기 전에 반드시 통과해야 한다.

핵심 검증 항목:
    1. 알려진 beta0를 회수하는가 (H1 엔진 정확성)
    2. 알려진 beta1 (상호작용)을 회수하는가 (H2 엔진 정확성)
    3. 효과가 없는 데이터에서 유의성을 만들어내지 않는가 (위양성 통제)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.event_study import (  # noqa: E402
    IndexEvent,
    compute_car,
    compute_shock,
    run_cross_sectional,
    summarize,
)


TRUE_BETA0 = 0.50    # 수요곡선 기울기
TRUE_BETA1 = -0.80   # 희소성 증폭: FF가 높을수록 효과 감쇠
NOISE_SD = 0.010


def build_synthetic(n_events=400, n_days=600, seed=42, null=False):
    """정답을 아는 합성 패널을 만든다.

    null=True이면 shock 효과를 0으로 설정 (위양성 검정용).
    """
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2019-01-01", periods=n_days)
    tickers = [f"{i:06d}" for i in range(n_events)]

    market_ret = pd.Series(rng.normal(0.0003, 0.010, n_days), index=cal)

    # 종목 특성
    ff_ratio = rng.uniform(0.10, 0.90, n_events)
    log_adv = rng.normal(22.0, 1.5, n_events)
    log_mktcap = rng.normal(26.0, 1.2, n_events)
    true_beta_mkt = rng.normal(1.0, 0.3, n_events)

    # 이벤트는 여러 날짜에 클러스터링되어 발생 (현실 반영)
    cluster_dates = cal[300:340:4]
    ad_idx = rng.choice(len(cluster_dates), n_events)

    shock_true = rng.normal(0, 1.0, n_events) * 0.02

    returns = pd.DataFrame(
        rng.normal(0, 0.001, (n_days, n_events)), index=cal, columns=tickers
    )
    # 시장 요인 부여
    for j, tk in enumerate(tickers):
        returns[tk] += true_beta_mkt[j] * market_ret

    events, ff_map, px_map = [], {}, {}

    for j, tk in enumerate(tickers):
        ad = cluster_dates[ad_idx[j]]
        ed = cal[cal.searchsorted(ad) + 10]

        # 진짜 효과: main 윈도우(AD-1 ~ ED+1)에 균등 주입
        i_s = cal.searchsorted(ad) - 1
        i_e = cal.searchsorted(ed) + 1
        span = i_e - i_s + 1

        effect = 0.0 if null else (
            TRUE_BETA0 * shock_true[j]
            + TRUE_BETA1 * shock_true[j] * (ff_ratio[j] - ff_ratio.mean())
        )
        returns.iloc[i_s : i_e + 1, j] += effect / span
        returns.iloc[i_s : i_e + 1, j] += rng.normal(0, NOISE_SD / np.sqrt(span), span)

        # shock_true를 재현하도록 weight/ff/price 역산
        ff = 1e7
        px = 10000.0
        aum = 1e12
        weight = shock_true[j] * ff * px / aum

        events.append(
            IndexEvent(
                ticker=tk,
                announcement_date=ad,
                effective_date=ed,
                index_name="SYNTH",
                direction=1,
                target_weight=weight,
            )
        )
        ff_map[tk] = ff
        px_map[tk] = px

    characteristics = pd.DataFrame(
        {
            "ticker": tickers,
            "ff_ratio": ff_ratio,
            "log_adv": log_adv,
            "log_mktcap": log_mktcap,
            "ret_60d": rng.normal(0, 0.15, n_events),
            "vol_20d": rng.uniform(0.01, 0.05, n_events),
        }
    )

    return {
        "returns": returns,
        "market_ret": market_ret,
        "events": events,
        "free_float": pd.Series(ff_map),
        "price": pd.Series(px_map),
        "tracking_aum": {"SYNTH": 1e12},
        "characteristics": characteristics,
    }


def run_pipeline(data, method="market_model"):
    shock_df = compute_shock(
        data["events"], data["free_float"], data["price"], data["tracking_aum"]
    )
    car_df = compute_car(
        data["returns"], data["market_ret"], data["events"], method=method
    )
    result, cols, merged = run_cross_sectional(
        car_df, shock_df, data["characteristics"]
    )
    return summarize(result, cols), len(merged)


def main() -> int:
    failures = []

    # --- 검정 1: 효과가 있는 데이터에서 정답 회수 -----------------------
    print("=" * 66)
    print("검정 1 — 알려진 계수 회수 (market_model)")
    print(f"정답: beta0 = {TRUE_BETA0:+.2f}, beta1 = {TRUE_BETA1:+.2f}")
    print("=" * 66)

    data = build_synthetic()
    tbl, n = run_pipeline(data)
    print(tbl.round(4).to_string())
    print(f"\n유효 이벤트: {n}건")

    b0 = tbl.loc["shock", "coef"]
    b1 = tbl.loc["shock_x_ff_ratio", "coef"]
    p1 = tbl.loc["shock_x_ff_ratio", "p"]

    if abs(b0 - TRUE_BETA0) > 0.15:
        failures.append(f"beta0 회수 실패: {b0:.3f} vs {TRUE_BETA0}")
    if abs(b1 - TRUE_BETA1) > 0.30:
        failures.append(f"beta1 회수 실패: {b1:.3f} vs {TRUE_BETA1}")
    if p1 >= 0.05:
        failures.append(f"beta1 유의성 검출 실패: p={p1:.4f}")

    print(f"\nbeta0: {b0:+.3f} (정답 {TRUE_BETA0:+.2f})")
    print(f"beta1: {b1:+.3f} (정답 {TRUE_BETA1:+.2f}), p={p1:.4f}")

    # --- 검정 2: 강건성 (market_adjusted에서도 부호 유지) ---------------
    print("\n" + "=" * 66)
    print("검정 2 — 강건성: 초과수익률 산출 방식 변경")
    print("사전등록 기각조건 3: 방식에 따라 beta1 부호가 뒤집히면 신뢰 불가")
    print("=" * 66)

    tbl_adj, _ = run_pipeline(data, method="market_adjusted")
    b1_adj = tbl_adj.loc["shock_x_ff_ratio", "coef"]
    print(f"market_model    beta1: {b1:+.3f}")
    print(f"market_adjusted beta1: {b1_adj:+.3f}")

    if np.sign(b1) != np.sign(b1_adj):
        failures.append("강건성 실패: 방식별 beta1 부호 불일치")
    else:
        print("→ 부호 일치. 강건성 통과.")

    # --- 검정 3: 위양성 통제 ---------------------------------------------
    print("\n" + "=" * 66)
    print("검정 3 — 위양성 통제 (효과 없는 데이터)")
    print("효과가 0인 데이터에서 유의한 결과를 만들어내면 안 된다")
    print("=" * 66)

    false_pos = 0
    trials = 20
    for s in range(trials):
        null_data = build_synthetic(seed=1000 + s, null=True)
        tbl_n, _ = run_pipeline(null_data)
        if tbl_n.loc["shock_x_ff_ratio", "p"] < 0.05:
            false_pos += 1

    rate = false_pos / trials
    print(f"위양성률: {false_pos}/{trials} = {rate:.1%}  (기대: ~5%)")
    if rate > 0.25:
        failures.append(f"위양성률 과다: {rate:.1%}")
    else:
        print("→ 통제 범위 내.")

    # --- 결론 -------------------------------------------------------------
    print("\n" + "=" * 66)
    if failures:
        print("검증 실패:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("전체 통과. 엔진을 실데이터에 적용 가능.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
