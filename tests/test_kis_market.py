import pandas as pd

from korea_sd.data.kis_market import KisMarketProvider


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, api_path, tr_id, params, tr_cont=""):
        self.calls.append((api_path, tr_id, params))
        start = pd.Timestamp(params["FID_INPUT_DATE_1"])
        end = pd.Timestamp(params["FID_INPUT_DATE_2"])
        rows = []
        for d in pd.bdate_range(start, end):
            rows.append(
                {
                    "stck_bsop_date": d.strftime("%Y%m%d"),
                    "stck_oprc": "100",
                    "stck_hgpr": "110",
                    "stck_lwpr": "90",
                    "stck_clpr": "105",
                    "acml_vol": "1,000",
                    "acml_tr_pbmn": "105000",
                }
            )
        return {"rt_cd": "0", "output2": rows}, {}


def test_kis_market_provider_normalises_and_chunks():
    c = FakeClient()
    p = KisMarketProvider(client=c, chunk_calendar_days=30)
    out = p.daily_stock_ohlcv("5930", "2026-01-01", "2026-03-31")
    assert set(["ticker", "date", "open", "high", "low", "close", "volume", "trading_value"]).issubset(out.columns)
    assert out["ticker"].eq("005930").all()
    assert out["close"].eq(105).all()
    assert out["volume"].eq(1000).all()
    assert len(c.calls) == 3
    assert all(call[1] == "FHKST03010100" for call in c.calls)
    assert all(call[2]["FID_ORG_ADJ_PRC"] == "0" for call in c.calls)


def test_kis_market_provider_does_not_backfill_current_listed_shares():
    c = FakeClient()
    p = KisMarketProvider(client=c)
    out = p.daily_stock_ohlcv("005930", "2026-01-01", "2026-01-15")
    assert "shares_outstanding" not in out.columns
    assert "market_cap" not in out.columns
