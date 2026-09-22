import pandas as pd

from korea_sd.data.dart_client import OpenDartClient


class MultiResponse:
    status_code = 200
    text = "ok"

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class DirectSession:
    def get(self, url, params=None, timeout=None):
        return MultiResponse(
            {
                "status": "000",
                "message": "정상",
                "list": [
                    {
                        "rcept_no": "20250318000001",
                        "se": "합계",
                        "istc_totqy": "1,000,000",
                        "tesstk_co": "100,000",
                        "distb_stock_co": "900,000",
                        "stlm_dt": "2024-12-31",
                    }
                ],
            }
        )


def test_dart_share_status_normalises_official_fields_and_uses_receipt_date():
    c = OpenDartClient("x", session=DirectSession())
    out = c.stock_total_status("00126380", 2024, "FY", ticker="005930")
    row = out.iloc[0]
    assert row["ticker"] == "005930"
    assert row["shares_outstanding"] == 1_000_000
    assert row["treasury_shares"] == 100_000
    assert row["distributed_shares"] == 900_000
    assert row["report_period_end"] == pd.Timestamp("2024-12-31")
    assert row["asof_date"] == pd.Timestamp("2025-03-18")
    assert row["share_status_source"] == "stockTotqySttus"


class ReconstructionSession:
    """Synthetic Samsung-like case: Q3 stock-total rows are all '-' values."""

    def get(self, url, params=None, timeout=None):
        endpoint = url.rsplit("/", 1)[-1]
        year = str(params.get("bsns_year"))
        report = str(params.get("reprt_code"))

        if endpoint == "stockTotqySttus.json" and year == "2025" and report == "11014":
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {
                            "rcept_no": "20251114000001",
                            "se": "합계",
                            "istc_totqy": "-",
                            "tesstk_co": "-",
                            "distb_stock_co": "-",
                            "now_to_isu_stock_totqy": "-",
                            "now_to_dcrs_stock_totqy": "-",
                            "stlm_dt": "2025-09-30",
                        },
                        {
                            "rcept_no": "20251114000001",
                            "se": "비고",
                            "istc_totqy": "-",
                            "tesstk_co": "-",
                            "distb_stock_co": "-",
                            "now_to_isu_stock_totqy": "-",
                            "now_to_dcrs_stock_totqy": "-",
                            "stlm_dt": "2025-09-30",
                        },
                    ],
                }
            )

        # Earlier 2025 interim reports unavailable/unusable, then 2024 FY is valid.
        if endpoint == "stockTotqySttus.json" and year == "2025" and report in {"11012", "11013"}:
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {
                            "rcept_no": "20250814000001",
                            "se": "합계",
                            "istc_totqy": "-",
                            "tesstk_co": "-",
                            "distb_stock_co": "-",
                            "now_to_isu_stock_totqy": "-",
                            "now_to_dcrs_stock_totqy": "-",
                            "stlm_dt": "2025-06-30" if report == "11012" else "2025-03-31",
                        }
                    ],
                }
            )

        if endpoint == "stockTotqySttus.json" and year == "2024" and report == "11011":
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {
                            "rcept_no": "20250318000001",
                            "se": "합계",
                            "istc_totqy": "6,792,669,250",
                            "tesstk_co": "33,750,000",
                            "distb_stock_co": "6,758,919,250",
                            "stlm_dt": "2024-12-31",
                        }
                    ],
                }
            )

        if endpoint == "irdsSttus.json" and year == "2025" and report == "11014":
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {
                            "rcept_no": "20251114000001",
                            "isu_dcrs_de": "2025-03-05",
                            "isu_dcrs_stle": "주식소각",
                            "isu_dcrs_stock_knd": "보통주",
                            "isu_dcrs_qy": "50,144,628",
                            "stlm_dt": "2025-09-30",
                        },
                        {
                            "rcept_no": "20251114000001",
                            "isu_dcrs_de": "2025-03-05",
                            "isu_dcrs_stle": "주식소각",
                            "isu_dcrs_stock_knd": "우선주",
                            "isu_dcrs_qy": "6,912,036",
                            "stlm_dt": "2025-09-30",
                        },
                    ],
                }
            )

        if endpoint == "tesstkAcqsDspsSttus.json" and year == "2025" and report == "11014":
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {
                            "rcept_no": "20251114000001",
                            "acqs_mth1": "총계",
                            "acqs_mth2": "총계",
                            "acqs_mth3": "총계",
                            "stock_knd": "보통주",
                            "trmend_qy": "91,413,157",
                            "stlm_dt": "2025-09-30",
                        },
                        {
                            "rcept_no": "20251114000001",
                            "acqs_mth1": "총계",
                            "acqs_mth2": "총계",
                            "acqs_mth3": "총계",
                            "stock_knd": "우선주",
                            "trmend_qy": "13,536,988",
                            "stlm_dt": "2025-09-30",
                        },
                    ],
                }
            )

        raise AssertionError((endpoint, year, report))


def test_dart_reconstructs_when_target_stock_total_is_dash_only():
    c = OpenDartClient("x", session=ReconstructionSession())
    out = c.stock_total_status("00126380", 2025, "Q3", ticker="005930")
    row = out.iloc[0]

    # 2024 FY issued shares minus 2025 retirements.
    assert row["shares_outstanding"] == 6_735_612_586
    assert row["treasury_shares"] == 104_950_145
    assert row["distributed_shares"] == 6_630_662_441
    assert row["report_period_end"] == pd.Timestamp("2025-09-30")
    assert row["asof_date"] == pd.Timestamp("2025-11-14")
    assert row["share_status_source"] == "reconstructed:baseline+irds+tesstk"
    assert row["baseline_report_year"] == 2024
    assert row["baseline_report_code"] == "11011"
    assert row["issued_share_delta"] == -57_056_664

class TreasuryNoGrandTotalSession(ReconstructionSession):
    def get(self, url, params=None, timeout=None):
        endpoint = url.rsplit("/", 1)[-1]
        year = str(params.get("bsns_year"))
        report = str(params.get("reprt_code"))
        if endpoint == "tesstkAcqsDspsSttus.json" and year == "2025" and report == "11014":
            return MultiResponse(
                {
                    "status": "000",
                    "list": [
                        {"rcept_no": "20251114000001", "acqs_mth1": "배당가능이익범위 이내 취득", "acqs_mth2": "직접취득", "acqs_mth3": "소계", "stock_knd": "보통주", "trmend_qy": "90,000,000", "stlm_dt": "2025-09-30"},
                        {"rcept_no": "20251114000001", "acqs_mth1": "기타취득", "acqs_mth2": "기타취득", "acqs_mth3": "소계", "stock_knd": "보통주", "trmend_qy": "1,413,157", "stlm_dt": "2025-09-30"},
                        {"rcept_no": "20251114000001", "acqs_mth1": "배당가능이익범위 이내 취득", "acqs_mth2": "직접취득", "acqs_mth3": "장내직접취득", "stock_knd": "보통주", "trmend_qy": "90,000,000", "stlm_dt": "2025-09-30"},
                        {"rcept_no": "20251114000001", "acqs_mth1": "기타취득", "acqs_mth2": "기타취득", "acqs_mth3": "기타취득", "stock_knd": "보통주", "trmend_qy": "1,413,157", "stlm_dt": "2025-09-30"},
                        {"rcept_no": "20251114000001", "acqs_mth1": "기타취득", "acqs_mth2": "기타취득", "acqs_mth3": "소계", "stock_knd": "우선주", "trmend_qy": "13,536,988", "stlm_dt": "2025-09-30"},
                        {"rcept_no": "20251114000001", "acqs_mth1": "기타취득", "acqs_mth2": "기타취득", "acqs_mth3": "기타취득", "stock_knd": "우선주", "trmend_qy": "13,536,988", "stlm_dt": "2025-09-30"},
                    ],
                }
            )
        return super().get(url, params=params, timeout=timeout)


def test_dart_reconstructs_treasury_from_subtotals_when_no_total_rows():
    c = OpenDartClient("x", session=TreasuryNoGrandTotalSession())
    out = c.stock_total_status("00126380", 2025, "Q3", ticker="005930")
    row = out.iloc[0]
    assert row["treasury_shares"] == 104_950_145
    assert row["distributed_shares"] == 6_630_662_441
