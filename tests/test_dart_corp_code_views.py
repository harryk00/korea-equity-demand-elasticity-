from __future__ import annotations

import io
import zipfile

from korea_sd.data.dart_corp_codes import OpenDartCorpCodeProvider


class DummyResponse:
    status_code = 200
    text = ""

    def __init__(self, content: bytes):
        self.content = content


class DummySession:
    def __init__(self, content: bytes):
        self.content = content

    def get(self, *args, **kwargs):
        return DummyResponse(self.content)


def make_zip():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <result>
      <list>
        <corp_code>00111111</corp_code>
        <corp_name>LISTED</corp_name>
        <corp_eng_name>LISTED CO</corp_eng_name>
        <stock_code>123456</stock_code>
        <modify_date>20260101</modify_date>
      </list>
      <list>
        <corp_code>00222222</corp_code>
        <corp_name>DELISTED</corp_name>
        <corp_eng_name>DELISTED CO</corp_eng_name>
        <stock_code></stock_code>
        <modify_date>20260102</modify_date>
      </list>
    </result>"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("CORPCODE.xml", xml.encode("utf-8"))
    return buf.getvalue()


def test_download_keeps_real_six_digit_listed_ticker():
    provider = OpenDartCorpCodeProvider(
        api_key="x",
        session=DummySession(make_zip()),
    )
    listed = provider.download()
    assert listed["ticker"].tolist() == ["123456"]
    assert listed["corp_code"].tolist() == ["00111111"]


def test_download_all_keeps_delisted_blank_stock_code():
    provider = OpenDartCorpCodeProvider(
        api_key="x",
        session=DummySession(make_zip()),
    )
    allc = provider.download_all()
    assert set(allc["corp_code"]) == {"00111111", "00222222"}
    row = allc.loc[allc["corp_code"] == "00222222"].iloc[0]
    assert row["ticker"] == ""
