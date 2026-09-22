import numpy as np
import pandas as pd

from korea_sd.model.targets import add_cooldown_events, add_forward_targets


def test_forward_targets_have_exact_trailing_nans():
    n = 60
    df = pd.DataFrame({
        "ticker": ["005930"] * n,
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "close": np.arange(100, 100 + n, dtype=float),
        "high": np.arange(101, 101 + n, dtype=float),
    })
    out = add_forward_targets(df)
    assert out["fwd_max_return_5d"].isna().sum() == 5
    assert out["fwd_max_return_10d"].isna().sum() == 10
    assert out["fwd_max_return_20d"].isna().sum() == 20
    assert out["fwd_max_return_40d"].isna().sum() == 40
    assert out["target_20d_30pct"].isna().sum() == 20


def test_cooldown_removes_overlapping_events():
    df = pd.DataFrame({
        "ticker": ["000001"] * 50,
        "date": pd.date_range("2026-01-01", periods=50, freq="D"),
        "target_20d_30pct": [0] * 50,
    })
    df.loc[[2, 5, 22, 23, 44], "target_20d_30pct"] = 1
    out = add_cooldown_events(df, cooldown=20)
    event_idx = out.index[out["surge_event_30pct"] == 1].tolist()
    assert event_idx == [2, 23, 44]
