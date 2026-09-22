import pandas as pd

from korea_sd.data.pit_universe import (
    build_union_summary,
    resolve_corp_map,
    trim_market_to_observable_free_float,
)


def test_build_union_summary_marks_exited_names():
    membership = pd.DataFrame(
        {
            "date": pd.to_datetime([
                "2025-01-02", "2025-01-02",
                "2025-01-03", "2025-01-03",
                "2025-01-06",
            ]),
            "ticker": ["000001", "000002", "000001", "000002", "000001"],
        }
    )
    out = build_union_summary(membership).set_index("ticker")
    assert bool(out.loc["000001", "active_on_last_snapshot"])
    assert bool(out.loc["000002", "exited_before_end"])
    assert out.loc["000002", "last_seen"] == pd.Timestamp("2025-01-03")


def test_resolve_corp_map_uses_historical_for_delisted():
    union = pd.DataFrame(
        {
            "ticker": ["000001", "000002"],
            "first_seen": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "last_seen": pd.to_datetime(["2025-01-03", "2025-01-03"]),
            "membership_days": [2, 2],
            "active_on_last_snapshot": [True, False],
            "exited_before_end": [False, True],
            "last_snapshot_date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
        }
    )
    current = pd.DataFrame(
        {"ticker": ["000001"], "corp_code": ["12345678"], "corp_name": ["현재회사"]}
    )
    hist = pd.DataFrame(
        {
            "ticker": ["000002"],
            "corp_code": ["87654321"],
            "corp_name": ["과거회사"],
            "rcept_dt": pd.to_datetime(["2024-03-01"]),
            "report_nm": ["사업보고서"],
            "map_source": ["historical_disclosure"],
        }
    )
    out = resolve_corp_map(union, current, hist).set_index("ticker")
    assert out.loc["000001", "corp_code"] == "12345678"
    assert out.loc["000002", "corp_code"] == "87654321"
    assert out.loc["000002", "corp_map_status"] == "ok"


def test_trim_market_starts_only_when_free_float_is_observable():
    market = pd.DataFrame(
        {
            "ticker": ["000001"] * 4 + ["000002"] * 2,
            "date": pd.to_datetime([
                "2025-01-02", "2025-01-03", "2025-03-31", "2025-04-01",
                "2025-01-02", "2025-01-03",
            ]),
            "close": [1, 1, 1, 1, 1, 1],
            "volume": [1, 1, 1, 1, 1, 1],
        }
    )
    dart = pd.DataFrame(
        {
            "ticker": ["000001", "000002"],
            "asof_date": pd.to_datetime(["2025-03-31", "2025-02-01"]),
            "distributed_shares": [100, None],
        }
    )
    trimmed, report = trim_market_to_observable_free_float(market, dart)
    assert trimmed["ticker"].unique().tolist() == ["000001"]
    assert trimmed["date"].min() == pd.Timestamp("2025-03-31")
    rep = report.set_index("ticker")
    assert rep.loc["000001", "rows_trimmed"] == 2
    assert rep.loc["000002", "status"] == "no_valid_free_float"
