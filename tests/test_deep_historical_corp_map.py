from __future__ import annotations

import pandas as pd

from korea_sd.data.pit_universe import OpenDartHistoricalCorpMapProvider


class FakeProvider(OpenDartHistoricalCorpMapProvider):
    def __init__(self):
        super().__init__(api_key="x")
        self.calls = []

    def _request_page_flexible(
        self,
        start,
        end,
        page_no,
        *,
        corp_cls=None,
        pblntf_ty="A",
        last_reprt_at="Y",
    ):
        self.calls.append((pd.Timestamp(start), pd.Timestamp(end), page_no, corp_cls))
        # Newest window finds ticker 111111, next older window finds 222222.
        if pd.Timestamp(end) >= pd.Timestamp("2025-03-01"):
            items = [
                {
                    "stock_code": "111111",
                    "corp_code": "00111111",
                    "corp_name": "A",
                    "rcept_dt": "20250301",
                    "report_nm": "사업보고서",
                    "corp_cls": "E",
                },
                {
                    "stock_code": "999999",
                    "corp_code": "00999999",
                    "corp_name": "NOT TARGET",
                    "rcept_dt": "20250301",
                    "report_nm": "사업보고서",
                    "corp_cls": "K",
                },
            ]
        else:
            items = [
                {
                    "stock_code": "222222",
                    "corp_code": "00222222",
                    "corp_name": "B",
                    "rcept_dt": "20241201",
                    "report_nm": "분기보고서",
                    "corp_cls": "K",
                }
            ]
        return {
            "status": "000",
            "list": items,
            "total_page": 1,
            "total_count": len(items),
        }


def test_targeted_historical_search_recovers_only_wanted():
    p = FakeProvider()
    out = p.search_target_tickers(
        ["111111", "222222"],
        start="2024-01-01",
        end="2025-03-31",
        corp_cls=None,
        pblntf_ty="A",
        window_days=80,
        stop_when_all_found=True,
    )
    assert set(out["ticker"]) == {"111111", "222222"}
    assert "999999" not in set(out["ticker"])
    assert set(out["corp_cls"]) == {"E", "K"}
