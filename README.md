# Korea Stock Supply-Demand Elasticity Strategy

## 1. Project Overview

이 프로젝트의 목적은 **KOSDAQ에서 유통물량이 제한된 종목에 수요가 집중될 때 발생하는 급등 가능성**을 정량적으로 검증하고, 이를 실제 매매 가능한 규칙으로 발전시키는 것이다.

핵심 가설:

> **Supply Tightness × Demand Pressure × Sell Pressure × Market State → 급등 확률 상승**

초기 목표는 **향후 20거래일 내 +30% 이상 상승할 종목을 사전에 탐지**하는 것이었으며, 단순 예측모델이 아니라 실제 체결, 비용, 포트폴리오 제약까지 포함한 실행 가능한 전략을 만드는 것을 목표로 했다.

---

## 2. Research Universe / PIT Data

- 연구 시작: 2025-01-01
- Universe: KOSDAQ PIT universe
- MSCI 편입 등 외생적 index-provider free float 효과는 제외
- 주요 소스: KIS Open API, OpenDART, FinanceDataReader, KRX/pykrxauth

핵심 core 결과:
- Requested tickers: 1,723
- Complete market + DART: 1,712
- Rows: 657,370
- `target_20d_30pct = 1`: 89,401
- 독립 +30% surge event: 8,188

주요 파일:

```text
data/pit_kosdaq/core/processed/stock_master_model.parquet
data/pit_kosdaq/core/processed/stock_master_model_demand_complete.parquet
data/pit_kosdaq/universe/membership_daily.parquet
data/pit_kosdaq/universe/pit_pipeline_universe.csv
```

---

## 3. Initial Target

기본 target:

> 신호일 이후 20거래일 안에 주가가 +30% 이상 상승

독립 급등 event는 cooldown을 적용하여 동일 상승구간의 중복 계산을 줄였다.

---

## 4. Supply Tightness Hypothesis

초기 핵심 신호:

```text
free_float_market_cap 하위 20%
AND
float_turnover_1d 상위 20%
```

전체 PIT:
- overall target rate ≈ 14.35%
- signal target rate ≈ 25.36%
- RR vs overall ≈ 1.77x

따라서 **작은 유통물량 + 높은 turnover** 조합은 급등 확률을 높였다.

하지만 신호 내부에서는 turnover가 가장 극단적으로 높은 종목의 성과가 오히려 악화됐다.

---

## 5. Final C5 Signal

C5 조건:

1. `free_float_market_cap` 전체시장 하위 20%
2. `float_turnover_1d` 전체시장 상위 20%
3. Base Signal 안에서 turnover 상대순위 Q1~Q2
4. Base Signal 안에서 free-float mcap Q1~Q2
5. 당일 수익률 최상위 20% (`ret1 Q5`) 제외

해석:

> **작은 유통물량 + 충분한 turnover는 필요하지만, 이미 극단적으로 과열된 종목은 제외**

C5 내부 우선순위:

```text
priority_score = float_signal_pct + turn_signal_pct
```

낮을수록 우선순위가 높다.

---

## 6. Demand Pressure

검증:
- Turnover acceleration
- Program net-buy level
- Program net-buy acceleration
- Execution strength
- Execution-strength acceleration

결론:

| 변수 | 결론 |
|---|---|
| Turnover acceleration | 부분 지지 |
| Program net-buy acceleration | 부분 지지 |
| Program net-buy level | 불안정 |
| Execution-strength acceleration | 기각 |
| Demand hard filter | 기각 |

Demand 변수는 현재 hard filter에 쓰지 않는다. Program acceleration은 향후 ranking challenger로만 유지한다.

---

## 7. Sell Pressure / Short

검증:
- Credit balance
- Securities lending balance
- Short-sale flow
- Lending-to-float
- Lending DTC proxy
- KRX reported short balance
- True DTC
- Short build-up

핵심 결론:
- Lending overhang: ✅ penalty 후보
- KRX reported short balance presence: ✅ penalty 후보
- Credit: ⚪ 불확실
- Lending-DTC proxy: ❌
- True DTC squeeze: ❌
- High SI × DTC squeeze: ❌
- Short build-up squeeze: ❌

즉 공매도/대차는 현재 **hard exclusion이 아니라 ranking penalty 후보**다.

---

## 8. Market State

중요한 발견:

> C5는 강한 장보다 **시장 전체가 약한 상태에서 일부 저유통주로 수요가 집중될 때** 더 강했다.

Frozen threshold:

```text
KOSDAQ 20d return <= +2.1656%
MA20 breadth      <= 34.9398%
60d drawdown      <= -3.4201%
```

3개 중 2개 이상 충족:

```text
MARKET_WEAK = True
```

현재 채택:

```text
C5 + Market Weakness 2-of-3
```

---

## 9. Entry Timing

검증:
- t+1 Open
- Gap filter
- 09:05
- 09:05 NO_CHASE
- 09:10
- 09:10 NO_CHASE

09:05 NO_CHASE는 이벤트 단위에서 일부 개선이 있었지만, 실제 포트폴리오에서는 t+1 Open보다 일관되게 우수하지 않았다.

최종:

```text
Entry = t+1 Open
```

실전:

```text
신호일 t 장 마감 후 후보 확정
→ t+1 08:30~09:00 동시호가 주문
→ 09:00 시가 체결 시도
→ 미체결 물량 추격매수하지 않음
```

---

## 10. Exit Optimization

검증 grid:

```text
TP:   +20%, +30%, +40%, +50%
SL:   -10%, -15%, -20%
Hold: 10, 20, 30 trading days
```

최종:

```text
Take Profit: +30%
Stop Loss:   -15%
Max Hold:    20 trading days
```

동일 bar에서 TP/SL 모두 발생 시 보수적으로 SL 우선. Gap-down이 stop 아래에서 시작하면 실제 시가 청산.

---

## 11. Portfolio Construction

실제 계좌 조건:
- 동일 종목 중복보유 금지
- 현금 100% 시작
- 미사용 슬롯은 현금 유지
- 최대 동시보유 3 / 5 / 10 테스트

### Main: MARKET + N10

Full:
- Total Return ≈ +140.8%
- Annualized ≈ +70.8%
- MDD ≈ -23.0%
- Sharpe ≈ 2.17
- Trades = 222

### Aggressive: MARKET + N3

Full:
- Total Return ≈ +247.8%
- Annualized ≈ +113.7%
- MDD ≈ -23.0%
- Sharpe ≈ 2.08
- Trades = 71

최종:

```text
N10 = Main
N3  = Aggressive challenger
```

---

## 12. Execution Robustness

MARKET 후보 1,034건 기준:
- delayed entry count = 0
- locked limit-up proxy = 0
- strict tradable rate = 100%

### N10 Slippage Stress

| Slippage / side | Annualized | MDD | PF |
|---|---:|---:|---:|
| 10bp | 70.8% | -23.0% | 1.77 |
| 30bp | 61.9% | -24.1% | 1.68 |
| 50bp | 53.4% | -25.1% | 1.59 |
| 100bp | 34.2% | -27.7% | 1.40 |

전략은 거래비용보다 **liquidity capacity**에 더 민감했다.

---

## 13. Liquidity / Capacity

실전 주문한도:

```text
order_size = min(account_equity / 10, ADV20 × participation_cap)
```

기본 research cap:

```text
participation_cap = 0.5% ADV20
```

N10 median capacity:

| ADV participation | Approx. account capacity |
|---|---:|
| 0.10% | 약 384만원 |
| 0.25% | 약 960만원 |
| 0.50% | 약 1,920만원 |
| 1.00% | 약 3,839만원 |

대략:
- 100만~1,000만원: 현실적
- 5,000만원: liquidity 영향 큼
- 1억원 이상: 확장성 부족

주의: ADV 0.5%는 시장충격 없이 반드시 체결 가능한 절대 기준이 아니라 research execution cap이다.

---

## 14. Current Frozen Main Strategy

### Signal

```text
C5 Supply Tightness
```

### Market Regime

3개 중 2개:

```text
KOSDAQ 20d return <= +2.1656%
MA20 breadth      <= 34.9398%
60d drawdown      <= -3.4201%
```

### Entry

```text
t close에서 신호 확정
→ t+1 open
```

### Portfolio

```text
Max positions = 10
Equal sleeves = account equity / 10
Unused sleeves remain cash
```

### Exit

```text
TP       = +30%
SL       = -15%
Max Hold = 20 trading days
```

### Liquidity

```text
planned_order = min(account_equity / 10, signal_date_ADV20 × ADV cap)
```

### Short / Lending

```text
risk flag / future ranking penalty
```

---

## 15. Date-Based Stock Picker

스크립트:

```text
scripts/pick_stocks_by_date.py
```

사용:

```bash
python scripts/pick_stocks_by_date.py   --date latest   --model data/pit_kosdaq/core/processed/stock_master_model.parquet   --market-state data/pit_kosdaq/analysis/market_state_v1/market_state_daily.parquet   --capital 10000000   --adv-cap 0.005
```

출력:

```text
입력 날짜 장 마감 기준 신호
→ 다음 거래일 09:00 매수 후보
```

저장:

```text
data/pit_kosdaq/signals/date_picker/signal_YYYYMMDD.csv
```

---

## 16. Latest Data Refresh

```bash
python scripts/refresh_to_latest.py --end latest
```

흐름:

```text
latest KOSDAQ trading date
→ PIT universe update
→ incremental KIS update
→ core model rebuild
→ KOSDAQ index refresh
→ Market State rebuild
```

---

## 17. Example: 2026-09-17

후보:

| Rank | Ticker | Name | Close | ADV20 | 0.5% ADV |
|---|---|---|---:|---:|---:|
| 1 | 308100 | 형지글로벌 | 283 | 약 1.65억 | 약 82.7만원 |
| 2 | 373170 | 엠아이큐브솔루션 | 1,904 | 약 9.17억 | 약 458.7만원 |
| 3 | 290560 | 파라택시스이더리움 | 1,204 | 약 12.49억 | 약 624.4만원 |
| 4 | 079950 | 인베니아 | 1,380 | 약 2.36억 | 약 117.8만원 |

1,000만원 계좌라면 sleeve는 100만원.

계획상:
- 형지글로벌: 약 82.7만원
- 엠아이큐브솔루션: 최대 100만원
- 파라택시스이더리움: 최대 100만원
- 인베니아: 최대 100만원

후보가 4개라고 250만원씩 나누지 않는다.

---

## 18. Can Max Hold Be Reduced To 5 Days?

기술적으로 가능하다.

현재:

```text
Max Hold = 20 trading days
```

을:

```text
Max Hold = 5 trading days
```

로 바꾸는 것은 코드상 어렵지 않다.

하지만 **현재 Main Strategy에 바로 반영하면 안 된다.**

기존 Exit Optimization에서 검증한 Hold는:

```text
10 / 20 / 30 days
```

뿐이므로:

```text
5 days = 미검증
```

이다.

### 새 challenger 권장

동일한:
- C5
- Market Weak
- t+1 Open
- TP +30%
- SL -15%
- N10
- 비용조건

을 유지하고 Hold만:

```text
5 / 10 / 20
```

으로 비교한다.

봐야 할 항목:
- CAGR
- Total Return
- MDD
- Sharpe
- PF
- Avg trade return
- Win rate
- Capital turnover
- Cash ratio
- Average holding period
- 2025 / 2026 연도별 안정성

### 5일 Hold의 잠재 장점

- 자본 회전율 증가
- 장기간 자금 묶임 감소
- 신규 신호에 더 자주 대응 가능

### 잠재 단점

- +30% 급등이 6~20일 사이에 나타나는 종목을 조기청산
- 큰 winner 일부 상실
- 전략 본래 target(20일 내 +30%)과 horizon mismatch

따라서:

> **5일은 충분히 실험할 가치가 있지만, 재백테스트 후에만 Main으로 채택한다.**

---

## 19. Research Status

| Stage | Status |
|---|---|
| PIT Universe | ✅ |
| Supply Tightness | ✅ |
| Demand Pressure | ✅ |
| Sell Pressure | ✅ |
| True DTC | ✅ |
| Market State | ✅ |
| Entry Timing | ✅ |
| Exit Optimization | ✅ |
| Portfolio Construction | ✅ |
| Execution Robustness | ✅ |
| Date Stock Picker | ✅ |
| Incremental Refresh | ✅ |
| 5-day Hold Challenger | ⏳ |
| Forward / Paper Test | ⏳ |

---

## 20. Research Discipline

2025와 2026 데이터는 이미 여러 차례 연구 의사결정에 사용되었다.

따라서 현재부터는:

```text
Frozen Main Strategy
→ 새 데이터 forward test
→ 실제 fill/slippage 기록
→ Main vs Challenger 비교
```

형태로 운영해야 한다.

새 규칙은 과거 수익률이 좋아졌다는 이유만으로 Main에 즉시 합치지 않는다.

---

## 21. Final Summary

현재 구조:

> **Low effective free float**
>
> + **moderately strong turnover**
>
> + **avoid extreme same-day overheat**
>
> + **weak broad market**
>
> → concentrated demand into constrained supply
>
> → outsized price response

현재 Main Strategy:

```text
C5
+
Market Weak 2-of-3
+
t+1 Open
+
Max 10 positions
+
TP +30%
+
SL -15%
+
Hold 20 days
+
Liquidity cap
```

다음 핵심 검증:

```text
1. Hold 5-day challenger
2. Forward / Paper Trading
3. Actual fill/slippage logging
4. Short/Lending ranking penalty
```
