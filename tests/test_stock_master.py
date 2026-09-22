import numpy as np
import pandas as pd
import pytest

from korea_sd.panels.stock_master import StockMasterError, build_stock_master


def market(n=25):
    dates = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "ticker": ["005930"] * n,
            "date": dates,
            "close": np.arange(n) + 100.0,
            "volume": np.arange(n) + 1000.0,
            "trading_value": (np.arange(n) + 1000.0) * (np.arange(n) + 100.0),
            "shares_outstanding": [1_000_000] * n,
        }
    )


def test_point_in_time_merge_never_uses_future_snapshot():
    m = market(25)
    ff = pd.DataFrame(
        {
            "ticker": ["005930", "005930"],
            "asof_date": [m.loc[0, "date"], m.loc[10, "date"]],
            "free_float_shares": [500_000, 400_000],
        }
    )
    out = build_stock_master(m, ff, ff_shares_col="free_float_shares")
    assert out.loc[9, "free_float_shares"] == 500_000
    assert out.loc[10, "free_float_shares"] == 400_000
    assert "float_days" in out.columns
    assert "amihud_20d" in out.columns


def test_ratio_input_builds_float_shares():
    m = market(3)
    ff = pd.DataFrame(
        {"ticker": ["005930"], "asof_date": [m.loc[0, "date"]], "free_float_ratio": [40.0]}
    )
    out = build_stock_master(m, ff, ff_ratio_col="free_float_ratio")
    assert out.loc[0, "free_float_shares"] == 400_000
    assert out.loc[0, "free_float_ratio"] == pytest.approx(0.4)


def test_strict_mode_rejects_rows_before_first_snapshot():
    m = market(3)
    ff = pd.DataFrame(
        {"ticker": ["005930"], "asof_date": [m.loc[1, "date"]], "free_float_shares": [500_000]}
    )
    with pytest.raises(StockMasterError):
        build_stock_master(m, ff, ff_shares_col="free_float_shares")


def test_dart_snapshot_supplies_historical_share_count_when_market_omits_it():
    m = market(3).drop(columns=["shares_outstanding"])
    ff = pd.DataFrame(
        {
            "ticker": ["005930"],
            "asof_date": [m.loc[0, "date"]],
            "shares_outstanding": [1_000_000],
            "distributed_shares": [400_000],
        }
    )
    out = build_stock_master(m, ff, distributed_shares_col="distributed_shares")
    assert out.loc[0, "shares_outstanding"] == 1_000_000
    assert out.loc[0, "free_float_shares"] == 400_000
    assert out.loc[0, "free_float_ratio"] == pytest.approx(0.4)
    assert out.loc[0, "market_cap"] == pytest.approx(out.loc[0, "close"] * 1_000_000)
