from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import os
from typing import Any
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pandas as pd
import requests


class CorpCodeError(RuntimeError):
    pass


def _ticker(value: Any) -> str:
    s = str(value or "").strip()
    return s.zfill(6) if s else ""


@dataclass
class OpenDartCorpCodeProvider:
    api_key: str
    base_url: str = "https://opendart.fss.or.kr/api"
    timeout_seconds: float = 30.0
    session: requests.Session = field(default_factory=requests.Session)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenDartCorpCodeProvider":
        key = os.environ.get("OPENDART_API_KEY", "").strip()
        if not key:
            raise CorpCodeError("Set OPENDART_API_KEY before downloading DART corp codes.")
        return cls(api_key=key, **kwargs)

    def _download_rows(self) -> pd.DataFrame:
        """Download the complete OpenDART corpCode.xml population.

        Important:
        OpenDART only populates stock_code for currently listed companies.
        Delisted filing companies can still retain corp_code/corp_name with a blank
        stock_code. Therefore this internal table deliberately keeps blank stock codes.
        """
        url = f"{self.base_url.rstrip('/')}/corpCode.xml"
        try:
            response = self.session.get(
                url,
                params={"crtfc_key": self.api_key},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise CorpCodeError(f"OpenDART corpCode request failed: {exc}") from exc
        if response.status_code != 200:
            raise CorpCodeError(
                f"OpenDART corpCode HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            with ZipFile(BytesIO(response.content)) as zf:
                xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
                if not xml_names:
                    raise CorpCodeError("OpenDART corpCode ZIP contained no XML file.")
                xml_bytes = zf.read(xml_names[0])
        except CorpCodeError:
            raise
        except Exception as exc:
            raise CorpCodeError("Could not decode OpenDART corpCode ZIP.") from exc

        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            raise CorpCodeError("Could not parse OpenDART CORPCODE.xml.") from exc

        rows: list[dict[str, str]] = []
        for node in root.findall("list"):
            corp_code = (node.findtext("corp_code") or "").strip()
            corp_name = (node.findtext("corp_name") or "").strip()
            stock_code = (node.findtext("stock_code") or "").strip()
            if not corp_code or not corp_name:
                continue
            rows.append(
                {
                    "ticker": _ticker(stock_code),
                    "stock_code": _ticker(stock_code),
                    "corp_code": corp_code.zfill(8),
                    "corp_name": corp_name,
                    "corp_eng_name": (node.findtext("corp_eng_name") or "").strip(),
                    "modify_date": (node.findtext("modify_date") or "").strip(),
                }
            )

        if not rows:
            raise CorpCodeError("OpenDART corpCode file contained no companies.")

        out = pd.DataFrame(rows)
        return (
            out.sort_values(["corp_code", "modify_date"])
            .drop_duplicates("corp_code", keep="last")
            .reset_index(drop=True)
        )

    def download_all(self) -> pd.DataFrame:
        """Return all DART filing corporations, including blank stock_code rows."""
        return self._download_rows()

    def download(self) -> pd.DataFrame:
        """Backward-compatible currently-listed ticker -> corp_code map."""
        out = self._download_rows()
        out = out[out["ticker"].astype(str).str.fullmatch(r"\d{6}", na=False)].copy()
        out = out[out["ticker"] != ""]
        if out.empty:
            raise CorpCodeError("OpenDART corpCode file contained no listed companies.")
        return (
            out[["ticker", "corp_code", "corp_name", "modify_date"]]
            .drop_duplicates("ticker", keep="last")
            .sort_values("ticker")
            .reset_index(drop=True)
        )
