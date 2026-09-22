import pandas as pd

from korea_sd.data.kis_universe import KisMarketCapUniverseProvider


def test_select_pilot_systematically_samples_rank_band():
    ranking = pd.DataFrame({
        "ticker": [str(i).zfill(6) for i in range(1, 501)],
        "market_cap_rank": list(range(1, 501)),
        "name": [f"종목{i}" for i in range(1, 501)],
        "current_price": [1000] * 500,
        "current_volume": [100000] * 500,
        "listed_shares_current": [1000000] * 500,
        "current_market_cap": list(range(500, 0, -1)),
    })
    out = KisMarketCapUniverseProvider.select_pilot(
        ranking, count=50, rank_start=31, rank_end=400, min_volume=30000
    )
    assert len(out) == 50
    assert out["ticker"].is_unique
    assert out["market_cap_rank"].min() >= 31
    assert out["market_cap_rank"].max() <= 400
    assert out["pilot_order"].tolist() == list(range(1, 51))
