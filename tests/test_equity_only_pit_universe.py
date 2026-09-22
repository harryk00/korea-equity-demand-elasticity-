from __future__ import annotations

import pandas as pd

from korea_sd.data.pit_universe import build_membership_from_listing_tables


def test_delisted_warrants_and_rights_are_excluded():
    current = pd.DataFrame(
        {
            "Code": ["067630", "109960"],
            "Name": ["HLB생명과학", "아프로젠H&G"],
            "Market": ["KOSDAQ", "KOSDAQ"],
            "ListingDate": ["2008-09-16", "2009-11-13"],
        }
    )

    delisted = pd.DataFrame(
        {
            "Symbol": [
                "067630",    # real stock
                "0676321C",  # warrant / right-like KRX security
                "1099621D",  # warrant
                "0543021A",  # warrant/right
            ],
            "Name": [
                "HLB생명과학",
                "HLB생명과학 9WR",
                "아프로젠H&G WR",
                "팬스타엔터프라이즈 WR",
            ],
            "Market": ["KOSDAQ"] * 4,
            "SecuGroup": ["주권", "신주인수권증권", "신주인수권증권", "신주인수권증서"],
            "ListingDate": [
                "2008-09-16",
                "2025-01-01",
                "2025-01-01",
                "2025-01-01",
            ],
            "DelistingDate": [
                "2025-06-30",
                "2026-06-26",
                "2026-06-26",
                "2025-05-09",
            ],
        }
    )

    membership, intervals = build_membership_from_listing_tables(
        current,
        delisted,
        start="2025-01-01",
        end="2026-08-29",
        market="KOSDAQ",
    )

    tickers = set(intervals["ticker"])
    assert "067630" in tickers
    assert "0676321C" not in tickers
    assert "1099621D" not in tickers
    assert "0543021A" not in tickers

    assert membership["ticker"].str.fullmatch(r"\d{6}").all()


def test_non_stock_six_digit_security_group_is_excluded():
    current = pd.DataFrame(
        {
            "Code": ["123456"],
            "Name": ["ACTIVE"],
            "Market": ["KOSDAQ"],
            "ListingDate": ["2020-01-01"],
        }
    )
    delisted = pd.DataFrame(
        {
            "Symbol": ["654321"],
            "Name": ["NONSTOCK"],
            "Market": ["KOSDAQ"],
            "SecuGroup": ["기타증권"],
            "ListingDate": ["2025-01-01"],
            "DelistingDate": ["2025-02-01"],
        }
    )

    _, intervals = build_membership_from_listing_tables(
        current,
        delisted,
        start="2025-01-01",
        end="2025-03-01",
    )
    assert set(intervals["ticker"]) == {"123456"}
