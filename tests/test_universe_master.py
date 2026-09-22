from __future__ import annotations

import io
import zipfile

import pandas as pd

from korea_sd.data.kis_universe import KisKosdaqMasterProvider, _FIELD_NAMES, _FIELD_WIDTHS


def _tail(**values: str) -> bytes:
    chunks = []
    for name, width in zip(_FIELD_NAMES, _FIELD_WIDTHS):
        text = str(values.get(name, ""))[:width].rjust(width)
        chunks.append(text.encode("ascii"))
    return b"".join(chunks)


def _line(ticker: str, name: str, mcap_100m: int, volume: int) -> bytes:
    head = ticker.ljust(9).encode() + ("KR" + ticker + "0000")[:12].ljust(12).encode() + name.encode("cp949")
    return head + _tail(
        previous_market_cap_100m=str(mcap_100m),
        previous_volume=str(volume),
        reference_price="10000",
        listed_shares_thousand="10000",
        spac_yn="N",
    )


def test_parse_zip_and_select_50():
    raw = b"\n".join(
        _line(f"{i:06d}", f"테스트{i}", 5000 - i, 100000)
        for i in range(1, 601)
    )
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        zf.writestr("kosdaq_code.mst", raw)
    df = KisKosdaqMasterProvider.parse_zip_bytes(bio.getvalue())
    assert len(df) == 600
    selected = KisKosdaqMasterProvider.select_pilot(
        df, count=50, rank_start=31, rank_end=400, min_volume=30000
    )
    assert len(selected) == 50
    assert selected["market_cap_rank"].min() >= 31
    assert selected["market_cap_rank"].max() <= 400
