from __future__ import annotations

import pandas as pd

from korea_sd.data.pit_universe import (
    normalize_corp_name,
    recover_missing_corp_map_by_name,
)


def test_normalize_corp_name_legal_forms():
    assert normalize_corp_name("(주)테스트 기업") == normalize_corp_name("주식회사 테스트기업")
    assert normalize_corp_name("㈜ABC-테크") == normalize_corp_name("ABC 테크")


def test_recover_unique_delisted_name():
    resolved = pd.DataFrame(
        [{
            "ticker": "123456",
            "interval_name": "테스트기업",
            "corp_name": "",
            "corp_code": "",
            "corp_code_candidates": "",
            "corp_code_count": 0,
            "corp_map_status": "missing",
            "corp_map_source": "",
        }]
    )
    allc = pd.DataFrame(
        [
            {
                "ticker": "",
                "corp_code": "00123456",
                "corp_name": "주식회사 테스트기업",
                "modify_date": "20250101",
            },
            {
                "ticker": "999999",
                "corp_code": "00999999",
                "corp_name": "다른회사",
                "modify_date": "20260101",
            },
        ]
    )
    out, audit = recover_missing_corp_map_by_name(resolved, allc)

    assert out.loc[0, "corp_map_status"] == "ok"
    assert out.loc[0, "corp_code"] == "00123456"
    assert "corpCode_name_exact" in out.loc[0, "corp_map_source"]
    assert audit.loc[0, "recovery_status"] == "recovered_unique_exact_name"


def test_ambiguous_name_is_not_guessed():
    resolved = pd.DataFrame(
        [{
            "ticker": "123456",
            "interval_name": "동일회사",
            "corp_name": "",
            "corp_code": "",
            "corp_code_candidates": "",
            "corp_code_count": 0,
            "corp_map_status": "missing",
            "corp_map_source": "",
        }]
    )
    allc = pd.DataFrame(
        [
            {"ticker": "", "corp_code": "00111111", "corp_name": "동일회사"},
            {"ticker": "", "corp_code": "00222222", "corp_name": "(주) 동일회사"},
        ]
    )
    out, audit = recover_missing_corp_map_by_name(resolved, allc)

    assert out.loc[0, "corp_map_status"] == "missing"
    assert audit.loc[0, "candidate_count"] == 2
    assert audit.loc[0, "recovery_status"] == "ambiguous_name_match"
