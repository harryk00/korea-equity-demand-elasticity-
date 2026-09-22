import pandas as pd

from korea_sd.panels.pit_eligibility import filter_full_period_free_float_eligible


def test_filters_all_missing_start_snapshots_at_once():
    market = pd.DataFrame({
        "ticker": ["111111", "111111", "222222", "333333"],
        "date": ["2025-01-02", "2025-01-03", "2025-01-02", "2025-01-02"],
    })
    snapshots = pd.DataFrame({
        "ticker": ["111111", "222222", "333333"],
        "asof_date": ["2024-03-20", "2025-03-20", "2024-03-20"],
        "distributed_shares": [100, 200, 0],
    })

    eligible, report = filter_full_period_free_float_eligible(
        market, snapshots, tickers=["111111", "222222", "333333"]
    )

    assert eligible == ["111111"]
    excluded = report.loc[report["status"] == "excluded", "ticker"].tolist()
    assert excluded == ["222222", "333333"]


def test_accepts_snapshot_exactly_on_first_market_date():
    market = pd.DataFrame({"ticker": ["111111"], "date": ["2025-01-02"]})
    snapshots = pd.DataFrame({
        "ticker": ["111111"],
        "asof_date": ["2025-01-02"],
        "distributed_shares": [100],
    })
    eligible, report = filter_full_period_free_float_eligible(market, snapshots)
    assert eligible == ["111111"]
    assert report.iloc[0]["status"] == "eligible"
