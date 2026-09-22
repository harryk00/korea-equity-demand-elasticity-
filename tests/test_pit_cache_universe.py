from __future__ import annotations

import pandas as pd

from korea_sd.data.pit_universe import build_membership_from_listing_tables


def test_interval_reconstruction_includes_active_and_delisted():
    current = pd.DataFrame(
        {
            "Code": ["111111", "222222", "999999"],
            "Name": ["ACTIVE", "NEW", "KOSPI"],
            "Market": ["KOSDAQ", "KOSDAQ", "KOSPI"],
            "ListingDate": ["2020-01-01", "2025-01-06", "2020-01-01"],
        }
    )
    delisted = pd.DataFrame(
        {
            "Symbol": ["333333", "444444", "888888"],
            "Name": ["DELIST", "OLD", "KOSPI_D"],
            "Market": ["KOSDAQ", "KOSDAQ", "KOSPI"],
            "ListingDate": ["2024-01-01", "2020-01-01", "2020-01-01"],
            "DelistingDate": ["2025-01-08", "2024-12-31", "2025-01-08"],
        }
    )

    membership, intervals = build_membership_from_listing_tables(
        current, delisted, start="2025-01-02", end="2025-01-10"
    )

    # Active KOSDAQ name is present throughout requested business-day window.
    active_dates = membership.loc[membership.ticker == "111111", "date"]
    assert active_dates.min() == pd.Timestamp("2025-01-02")
    assert active_dates.max() == pd.Timestamp("2025-01-10")

    # New listing starts only on its listing date.
    new_dates = membership.loc[membership.ticker == "222222", "date"]
    assert new_dates.min() == pd.Timestamp("2025-01-06")

    # Delisted name disappears after delisting date.
    de_dates = membership.loc[membership.ticker == "333333", "date"]
    assert de_dates.max() == pd.Timestamp("2025-01-08")

    # Already-delisted-before-window and KOSPI names are excluded.
    assert "444444" not in set(membership.ticker)
    assert "999999" not in set(membership.ticker)
    assert "888888" not in set(membership.ticker)

    assert set(intervals.ticker) == {"111111", "222222", "333333"}


def test_ticker_date_dedup_for_overlapping_records():
    current = pd.DataFrame(
        {
            "Code": ["111111"],
            "Name": ["ACTIVE"],
            "Market": ["KOSDAQ"],
            "ListingDate": ["2020-01-01"],
        }
    )
    # Artificial overlapping record: dedup should still guarantee one ticker-date.
    delisted = pd.DataFrame(
        {
            "Symbol": ["111111"],
            "Name": ["OLD_RECORD"],
            "Market": ["KOSDAQ"],
            "ListingDate": ["2020-01-01"],
            "DelistingDate": ["2025-01-08"],
        }
    )
    membership, _ = build_membership_from_listing_tables(
        current, delisted, start="2025-01-06", end="2025-01-10"
    )
    assert not membership.duplicated(["date", "ticker"]).any()
