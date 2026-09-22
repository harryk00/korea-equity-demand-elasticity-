import numpy as np
import pandas as pd
import pytest

from korea_sd.data.kis_microstructure import KisMicrostructureProvider
from korea_sd.panels.microstructure import (
    PanelMergeError,
    add_microstructure_features,
    merge_microstructure_panels,
)


def test_lending_normalizer():
    raw = pd.DataFrame(
        {
            "bsop_date": ["20260828"],
            "new_stcn": ["1,000"],
            "rdmp_stcn": ["400"],
            "prdy_rmnd_vrss": ["600"],
            "rmnd_stcn": ["10,000"],
            "rmnd_amt": ["1000000"],
        }
    )
    out = KisMicrostructureProvider._normalise_lending(raw, "005930")
    assert out.loc[0, "lending_balance_shares"] == 10_000
    assert out.loc[0, "lending_net_change_shares"] == 600


def test_short_normalizer():
    raw = pd.DataFrame(
        {
            "stck_bsop_date": ["20260828"],
            "ssts_cntg_qty": ["100"],
            "ssts_tr_pbmn": ["1000000"],
            "ssts_vol_rlim": ["3.2"],
            "ssts_tr_pbmn_rlim": ["3.4"],
        }
    )
    out = KisMicrostructureProvider._normalise_short(raw, "005930")
    assert out.loc[0, "short_volume"] == 100
    assert out.loc[0, "short_value"] == 1_000_000


def test_credit_normalizer():
    raw = pd.DataFrame(
        {
            "deal_date": ["20260828"],
            "whol_loan_new_stcn": ["100"],
            "whol_loan_rdmp_stcn": ["40"],
            "whol_loan_rmnd_stcn": ["1000"],
        }
    )
    out = KisMicrostructureProvider._normalise_credit(raw, "005930")
    assert out.loc[0, "credit_net_change_shares"] == 60


def test_program_normalizer():
    raw = pd.DataFrame(
        {
            "stck_bsop_date": ["20260828"],
            "whol_smtn_ntby_qty": ["100"],
            "whol_smtn_ntby_tr_pbmn": ["1,000,000"],
        }
    )
    out = KisMicrostructureProvider._normalise_program(raw, "005930")
    assert out.loc[0, "program_netbuy_volume"] == 100
    assert out.loc[0, "program_netbuy_value"] == 1_000_000


def test_execution_strength_proxy():
    raw = pd.DataFrame(
        {"stck_bsop_date": ["20260828"], "total_seln_qty": ["100"], "total_shnu_qty": ["150"]}
    )
    out = KisMicrostructureProvider._normalise_execution(raw, "005930")
    assert out.loc[0, "execution_strength"] == 150.0


def test_merge_is_strict_and_missing_remains_nan():
    master = pd.DataFrame(
        {
            "ticker": ["005930", "005930"],
            "date": pd.to_datetime(["2026-08-27", "2026-08-28"]),
            "close": [100, 101],
            "free_float_shares": [1000, 1000],
            "free_float_market_cap": [100000, 101000],
        }
    )
    lending = pd.DataFrame(
        {"ticker": ["005930"], "date": [pd.Timestamp("2026-08-28")], "lending_balance_shares": [100]}
    )
    out = merge_microstructure_panels(master, {"lending": lending})
    assert len(out) == 2
    assert np.isnan(out.loc[0, "lending_balance_shares"])
    assert out.loc[1, "lending_balance_shares"] == 100


def test_duplicate_panel_rejected():
    master = pd.DataFrame({"ticker": ["005930"], "date": [pd.Timestamp("2026-08-28")]})
    dup = pd.DataFrame(
        {"ticker": ["005930", "005930"], "date": [pd.Timestamp("2026-08-28")] * 2, "x": [1, 2]}
    )
    with pytest.raises(PanelMergeError):
        merge_microstructure_panels(master, {"dup": dup})


def test_supply_normalized_features_and_rolling_no_future():
    dates = pd.bdate_range("2026-07-01", periods=25)
    df = pd.DataFrame(
        {
            "ticker": ["005930"] * 25,
            "date": dates,
            "close": [100.0] * 25,
            "trading_value": [1_000_000.0] * 25,
            "adv20_value": [900_000.0] * 25,
            "free_float_shares": [1000.0] * 25,
            "free_float_market_cap": [100_000.0] * 25,
            "lending_net_change_shares": np.arange(25),
            "lending_balance_shares": [100.0] * 25,
            "short_volume": [10.0] * 25,
            "short_value": [1000.0] * 25,
            "credit_net_change_shares": np.arange(25),
            "credit_balance_shares": [50.0] * 25,
            "program_netbuy_volume": np.arange(25),
            "program_netbuy_value": np.arange(25) * 100.0,
            "execution_strength": np.arange(25) + 100.0,
        }
    )
    out = add_microstructure_features(df)
    assert out.loc[0, "lending_balance_float"] == pytest.approx(0.1)
    # At row 19, ma20 can only use rows 0..19, never row 20+.
    assert out.loc[19, "execution_strength_ma20"] == pytest.approx(np.mean(np.arange(20) + 100.0))
