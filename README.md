# Korea Stock Supply-Demand Model

## 1. 모델의 목적

이 모델은 KOSDAQ 종목 중에서 **유통 가능한 주식 수가 적고, 제한된 공급에 비해 수요가 강하게 들어오는 종목**을 찾아
향후 단기간에 큰 가격 상승이 발생할 가능성이 높은 종목을 선별하는 것을 목표로 한다.

핵심 아이디어는 다음과 같다.

> **Supply Tightness × Demand Concentration × Market Weakness → 큰 가격 반응**

즉 시장 전체가 강해서 모든 종목이 오르는 상황을 노리는 전략이 아니라,

> **시장 전체가 약한 상태에서도 특정 저유통 종목으로 수요가 집중될 때 발생하는 급등**

을 포착하는 구조다.

---

# 2. 전체 모델 구조

```text
[1] KOSDAQ PIT Universe
        ↓
[2] Supply Tightness 계산
        ↓
[3] C5 Signal 생성
        ↓
[4] Market Weakness Filter
        ↓
[5] Candidate Ranking
        ↓
[6] 최대 10종목 선정
        ↓
[7] 다음 거래일 시가 매수
        ↓
[8] TP / SL / Max Hold 적용
        ↓
[9] Liquidity / ADV 제한
```

---

# 3. 데이터 구조

## Universe

KOSDAQ PIT(Point-in-Time) Universe를 사용한다.

즉 현재 상장 종목 목록을 과거에 그대로 적용하지 않고,
각 날짜 당시 실제 존재했던 종목만 사용한다.

주요 데이터 소스:

```text
KIS Open API
OpenDART
FinanceDataReader
KRX / pykrxauth
```

주요 모델 파일:

```text
data/pit_kosdaq/core/processed/stock_master_model.parquet
```

---

# 4. 핵심 변수

## 4.1 Free Float Market Cap

```text
free_float_market_cap
```

실제로 시장에서 거래 가능한 주식의 가치가 작은 종목을 찾는다.

값이 작을수록:

```text
시장에 공급되는 주식이 적음
→ 같은 매수세에도 가격이 더 크게 움직일 가능성
```

---

## 4.2 Float Turnover

```text
float_turnover_1d
```

유통주식 대비 하루 거래량 수준을 나타낸다.

값이 높을수록:

```text
적은 유통물량에 강한 거래수요가 들어옴
```

을 의미한다.

단, turnover가 지나치게 높은 종목은 이미 과열된 경우가 많아 제외한다.

---

# 5. C5 Signal

현재 모델의 핵심 매수신호는 C5이다.

## 1단계: Base Signal

```text
free_float_market_cap 하위 20%
AND
float_turnover_1d 상위 20%
```

즉:

> **유통주식은 적고 거래수요는 강한 종목**

을 먼저 찾는다.

---

## 2단계: 과열 제거

Base Signal 내부에서:

```text
turnover 상대순위 Q1~Q2
free-float mcap 상대순위 Q1~Q2
당일 수익률 최상위 20% 제외
```

한다.

즉 최종 C5는:

> 작은 free float  
> + 충분히 높은 turnover  
> + 지나친 turnover 과열 제외  
> + 당일 급등 과열 제외

구조다.

---

# 6. Market Weakness Filter

C5가 발생했다고 무조건 매수하지 않는다.

연구 결과 C5는 강한 시장보다 **시장 전체가 약할 때 더 좋은 성과**를 보였다.

현재 Market Weak 조건:

```text
1. KOSDAQ 20일 수익률 <= +2.1656%
2. MA20 상회 종목 비율 <= 34.9398%
3. KOSDAQ 60일 고점 대비 Drawdown <= -3.4201%
```

위 세 조건 중:

```text
2개 이상 충족
```

해야 매수가 가능하다.

즉:

```text
C5 = True
AND
Market Weak Score >= 2
```

가 최종 매수환경이다.

---

# 7. 종목 Ranking

같은 날 여러 종목이 발생하면 다음 우선순위를 사용한다.

```text
priority_score
=
float_signal_pct
+
turn_signal_pct
```

값이 낮을수록 우선순위가 높다.

해석:

```text
더 작은 free float
+
C5 내부에서 상대적으로 덜 과열된 turnover
```

종목을 우선한다.

최대:

```text
10종목
```

까지 선정한다.

---

# 8. 매수 시점

신호는 반드시 **장 마감 이후** 확정한다.

예:

```text
9월 17일 종가 데이터
→ 9월 17일 신호 계산
→ 다음 거래일 매수
```

현재 Entry Rule:

```text
t+1 Open
```

즉 다음 거래일:

```text
08:30~09:00 동시호가 주문
→ 09:00 시가 체결 시도
```

한다.

현재 연구 규칙에서는:

```text
미체결 물량을 09:05 이후 추격매수하지 않음
```

으로 본다.

---

# 9. 포트폴리오 구조

현재 Main Portfolio:

```text
Max Positions = 10
```

계좌자산을 10개의 동일 sleeve로 나눈다.

예:

```text
계좌자산 = 10,000,000원

1개 종목 기본 sleeve
= 10,000,000 / 10
= 1,000,000원
```

후보가 4개라고 해서 250만원씩 투자하지 않는다.

```text
4종목 × 100만원
+
나머지 현금
```

형태로 운용한다.

---

# 10. Liquidity / ADV 제한

저유통 종목을 대상으로 하기 때문에
계좌규모보다 **시장 유동성**이 더 중요한 제약이다.

실제 주문금액:

```text
order_size
=
min(
    account_equity / 10,
    ADV20 × participation_cap
)
```

현재 기본 research cap:

```text
ADV20의 0.5%
```

예:

```text
ADV20 = 2억원

0.5% ADV
= 100만원
```

이면 해당 종목에는 최대 약 100만원까지만 주문한다.

주의:

> 0.5% ADV는 절대적인 시장 체결 한도가 아니라
> 현재 연구에서 사용하는 보수적 execution cap이다.

---

# 11. 매도 규칙

## 연구상 검증된 Main Exit

```text
Take Profit = +30%
Stop Loss   = -15%
Max Hold    = 20 trading days
```

즉:

```text
+30% 도달 → 익절
-15% 도달 → 손절
둘 다 미도달 → 최대 보유기간 종료 시 종가 청산
```

---

# 12. Max Hold 5일 설정

코드상 최대 보유기간은 변경 가능하다.

파일:

```text
scripts/pick_stocks_by_date.py
```

현재 기본:

```python
MAX_HOLD = 20
```

5일로 운영하려면:

```python
MAX_HOLD = 5
```

로 바꿀 수 있다.

하지만 중요한 점:

> **연구상 검증된 Main Strategy는 20일 보유이다.**

기존 Exit Optimization에서는:

```text
10 / 20 / 30일
```

을 비교했고,
5일은 아직 별도로 검증하지 않았다.

따라서 현재 구분:

```text
20일 = Research-Frozen Main
5일  = Operational Challenger / 미검증
```

이다.

---

# 13. Short / Lending 데이터

연구 결과:

```text
공매도잔고 존재
대차잔고 증가
```

는 대체로 성과에 부정적인 영향을 보였다.

하지만 현재 모델에서는:

```text
Hard Filter ❌
Risk Flag / Ranking Penalty 후보 ✅
```

로 사용한다.

따라서 출력의:

```text
Short YES
Short NO
Short UNKNOWN
```

은 참고 위험표시다.

현재 매수 후보 자체를 제거하지 않는다.

---

# 14. 날짜 입력형 모델

사용 스크립트:

```text
scripts/pick_stocks_by_date.py
```

예:

```bash
python scripts/pick_stocks_by_date.py   --date 2026-09-17   --model data/pit_kosdaq/core/processed/stock_master_model.parquet   --market-state data/pit_kosdaq/analysis/market_state_v1/market_state_daily.parquet   --capital 10000000   --adv-cap 0.005
```

또는 최신 데이터:

```bash
python scripts/pick_stocks_by_date.py   --date latest   --model data/pit_kosdaq/core/processed/stock_master_model.parquet   --market-state data/pit_kosdaq/analysis/market_state_v1/market_state_daily.parquet   --capital 10000000   --adv-cap 0.005
```

출력:

```text
신호 기준일
예상 매수일
Market Weak Score
매수 후보
우선순위
ADV20
ADV 주문한도
Short Risk Flag
익절 / 손절 / 보유기간
```

---

# 15. 최신 데이터 업데이트

최신화:

```bash
python scripts/refresh_to_latest.py --end latest
```

업데이트 흐름:

```text
최신 KOSDAQ 거래일 확인
        ↓
PIT Universe 최신화
        ↓
KIS 신규 일봉 추가
        ↓
Core Model 재생성
        ↓
KOSDAQ Market State 최신화
        ↓
latest 신호 계산 가능
```

그 다음:

```bash
python scripts/pick_stocks_by_date.py --date latest ...
```

를 실행한다.

---

# 16. 현재 모델의 핵심 논리

모델이 찾는 상황은 다음과 같다.

```text
시장 전체는 약함
        ↓
대부분 종목의 수요가 강하지 않음
        ↓
그런데 특정 종목에 거래수요 집중
        ↓
해당 종목의 유통가능 물량은 작음
        ↓
수요 대비 공급 부족
        ↓
가격 탄력성 확대
        ↓
큰 상승 가능성
```

이를 수식처럼 표현하면:

```text
Price Response
≈
Demand Shock
/
Effective Tradable Supply
```

즉:

> **같은 수준의 매수세라도 실제로 시장에 풀려 있는 물량이 적으면 가격 반응이 더 크게 나타난다.**

이것이 모델의 가장 핵심적인 경제적 가설이다.

---

# 17. 현재 Main Strategy

```text
KOSDAQ PIT Universe
        ↓
Small Free Float
        ↓
High but Not Extreme Turnover
        ↓
Avoid Same-Day Overheat
        ↓
C5
        ↓
Market Weak 2-of-3
        ↓
Rank Candidates
        ↓
Top 10
        ↓
t+1 Open
        ↓
Position Size = min(Equity/10, ADV cap)
        ↓
TP +30%
SL -15%
Hold 20 days
```

---

# 18. 현재 연구상 채택 / 기각

| 요소 | 상태 |
|---|---|
| Small Free Float | ✅ 채택 |
| High Turnover | ✅ 채택 |
| Extreme Turnover | ❌ 제외 |
| Same-day Overheat | ❌ 제외 |
| Market Weakness | ✅ 채택 |
| Demand hard filter | ❌ |
| Program acceleration | 🟡 Ranking 후보 |
| Lending overhang | 🟡 Penalty 후보 |
| Short balance | 🟡 Penalty 후보 |
| True DTC squeeze | ❌ |
| 09:05 entry | ❌ |
| t+1 Open | ✅ |
| TP +30% | ✅ |
| SL -15% | ✅ |
| Hold 20d | ✅ Research Main |
| Hold 5d | 🟡 Challenger |
| N10 | ✅ Main |
| N3 | 🟡 Aggressive |
| ADV liquidity cap | ✅ |

---


# 20. 가장 중요한 주의사항

2025~2026 데이터는 이미 연구와 전략 선택에 반복적으로 사용되었다.

따라서 앞으로는 과거 성과를 계속 보고 규칙을 수정하기보다:

```text
현재 규칙 Freeze
        ↓
새로운 날짜의 Signal 생성
        ↓
Paper / Live Forward Test
        ↓
실제 체결가 및 Slippage 기록
        ↓
성과 비교
```

과정으로 넘어가야 한다.

현재 모델은:

> **과거 데이터를 기반으로 만든 연구전략**

이며,

> **새 데이터에서의 forward 성과가 최종 검증 단계**

이다.
