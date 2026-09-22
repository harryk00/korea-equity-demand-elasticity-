from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import zipfile

import numpy as np
import pandas as pd
import requests


MASTER_URL = "https://new.real.download.dws.co.kr/common/master/kosdaq_code.mst.zip"

# Official KIS kosdaq_code.mst fixed-width tail layout.  The official sample
# slices 222 characters including the newline; after splitlines() the payload
# itself is 221 bytes wide.
_FIELD_WIDTHS = [
    2,1,4,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,9,
    5,5,1,1,1,2,1,1,1,2,2,2,3,1,3,12,12,8,15,21,2,7,1,1,1,1,
    9,9,9,5,9,8,9,3,1,1,1,
]
_FIELD_NAMES = [
    "security_group_code", "market_cap_size_code", "sector_large", "sector_mid",
    "sector_small", "venture_yn", "low_liquidity_yn", "krx_security_yn",
    "etp_product_code", "krx100_yn", "krx_auto_yn", "krx_semicon_yn",
    "krx_bio_yn", "krx_bank_yn", "spac_yn", "krx_energy_chem_yn",
    "krx_steel_yn", "short_overheat_code", "krx_media_telecom_yn",
    "krx_construction_yn", "kosdaq_investment_attention_yn", "krx_securities_code",
    "krx_ship_code", "krx_insurance_yn", "krx_transport_yn", "kosdaq150_yn",
    "reference_price", "regular_lot", "afterhours_lot", "trading_halt_yn",
    "liquidation_trade_yn", "managed_stock_yn", "market_warning_code",
    "market_warning_pre_yn", "unfaithful_disclosure_yn", "backdoor_listing_yn",
    "lock_code", "par_value_change_code", "capital_increase_code", "margin_rate",
    "credit_order_yn", "credit_period", "previous_volume", "par_value",
    "listing_date", "listed_shares_thousand", "capital", "fiscal_month",
    "ipo_price", "preferred_stock_code", "short_sale_overheat_yn",
    "abnormal_surge_yn", "krx300_yn", "sales", "operating_profit",
    "ordinary_profit", "net_income", "roe", "base_yyyymm",
    "previous_market_cap_100m", "group_code", "company_credit_limit_yn",
    "collateral_loan_yn", "stock_lending_yn",
]
_TAIL_LEN = sum(_FIELD_WIDTHS)
if _TAIL_LEN != 221 or len(_FIELD_WIDTHS) != len(_FIELD_NAMES):
    raise RuntimeError("KOSDAQ master field layout is inconsistent")


class UniverseDataError(RuntimeError):
    pass


def _text(b: bytes) -> str:
    return b.decode("cp949", errors="ignore").strip()


def _number(value: object) -> float:
    s = str(value).replace(",", "").strip()
    if not s or s in {"-", "nan", "None"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _tail_values(tail: bytes) -> dict[str, str]:
    out: dict[str, str] = {}
    pos = 0
    for name, width in zip(_FIELD_NAMES, _FIELD_WIDTHS):
        out[name] = _text(tail[pos:pos + width])
        pos += width
    return out


@dataclass
class KisKosdaqMasterProvider:
    """Download and parse the official KIS KOSDAQ security master.

    This avoids relying on the market-cap ranking endpoint returning pages
    beyond its first response.  The master already contains previous-day
    volume and market capitalization, which is sufficient for deterministic
    pilot-universe construction without hundreds of REST calls.
    """

    url: str = MASTER_URL
    timeout_seconds: float = 60.0

    def download(self) -> pd.DataFrame:
        try:
            response = requests.get(self.url, timeout=self.timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UniverseDataError(f"KIS KOSDAQ master download failed: {exc}") from exc
        return self.parse_zip_bytes(response.content)

    @staticmethod
    def parse_zip_bytes(content: bytes) -> pd.DataFrame:
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".mst")]
                if not names:
                    names = zf.namelist()
                if not names:
                    raise UniverseDataError("KIS KOSDAQ master ZIP is empty")
                raw = zf.read(names[0])
        except zipfile.BadZipFile as exc:
            raise UniverseDataError("KIS KOSDAQ master download is not a valid ZIP") from exc
        return KisKosdaqMasterProvider.parse_mst_bytes(raw)

    @staticmethod
    def parse_mst_bytes(raw: bytes) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for line in raw.splitlines():
            line = line.rstrip(b"\r")
            if len(line) <= _TAIL_LEN + 21:
                continue
            head = line[:-_TAIL_LEN]
            tail = line[-_TAIL_LEN:]
            ticker = _text(head[0:9])[-6:].zfill(6)
            isin = _text(head[9:21])
            name = _text(head[21:])
            if not re.fullmatch(r"\d{6}", ticker) or not name:
                continue
            fields = _tail_values(tail)
            mcap_100m = _number(fields["previous_market_cap_100m"])
            volume = _number(fields["previous_volume"])
            listed_thousand = _number(fields["listed_shares_thousand"])
            rows.append(
                {
                    "ticker": ticker,
                    "isin": isin,
                    "name": name,
                    "current_price": _number(fields["reference_price"]),
                    # Master values are previous-business-day snapshots.
                    "current_volume": volume,
                    "listed_shares_current": listed_thousand * 1000.0 if np.isfinite(listed_thousand) else np.nan,
                    "current_market_cap": mcap_100m * 100_000_000.0 if np.isfinite(mcap_100m) else np.nan,
                    "market_cap_100m": mcap_100m,
                    "spac_yn": fields["spac_yn"],
                    "etp_product_code": fields["etp_product_code"],
                    "preferred_stock_code": fields["preferred_stock_code"],
                    "trading_halt_yn": fields["trading_halt_yn"],
                    "liquidation_trade_yn": fields["liquidation_trade_yn"],
                    "managed_stock_yn": fields["managed_stock_yn"],
                    "low_liquidity_yn": fields["low_liquidity_yn"],
                    "base_yyyymm": fields["base_yyyymm"],
                }
            )
        if len(rows) < 100:
            raise UniverseDataError(
                f"Parsed only {len(rows)} KOSDAQ master rows; layout/download may have changed"
            )
        out = pd.DataFrame(rows).drop_duplicates("ticker", keep="first")
        return out.reset_index(drop=True)

    @staticmethod
    def select_pilot(
        master: pd.DataFrame,
        *,
        count: int = 50,
        rank_start: int = 31,
        rank_end: int = 400,
        min_volume: int = 30_000,
    ) -> pd.DataFrame:
        if count < 1:
            raise ValueError("count must be >= 1")
        if rank_start < 1 or rank_end < rank_start:
            raise ValueError("invalid market-cap rank range")

        pool = master.copy()
        pool["current_market_cap"] = pd.to_numeric(pool["current_market_cap"], errors="coerce")
        pool["current_volume"] = pd.to_numeric(pool["current_volume"], errors="coerce")
        pool = pool.loc[pool["current_market_cap"].fillna(0) > 0].copy()

        # Remove non-operating/common-equity-like securities conservatively.
        name_bad = pool["name"].str.contains(
            r"스팩|SPAC|ETF|ETN|리츠|REIT",
            case=False,
            regex=True,
            na=False,
        )
        spac_series = pool["spac_yn"] if "spac_yn" in pool.columns else pd.Series("", index=pool.index)
        spac_bad = spac_series.astype(str).str.upper().eq("Y")
        # Preferred-stock code conventions can vary; name-based protection is
        # added so a changed code table cannot wipe out the universe.
        pref_name_bad = pool["name"].str.contains(r"우(?:B|C)?$", regex=True, na=False)
        pool = pool.loc[~(name_bad | spac_bad | pref_name_bad)].copy()

        pool = pool.sort_values(["current_market_cap", "ticker"], ascending=[False, True])
        pool["market_cap_rank"] = np.arange(1, len(pool) + 1)
        pool = pool.loc[pool["market_cap_rank"].between(rank_start, rank_end)].copy()
        pool = pool.loc[pool["current_volume"].fillna(0) >= min_volume].copy()
        pool = pool.sort_values("market_cap_rank").reset_index(drop=True)

        if len(pool) < count:
            raise UniverseDataError(
                f"Only {len(pool)} eligible KOSDAQ names after master filters; need {count}. "
                f"rank_range={rank_start}-{rank_end}, min_volume={min_volume}. "
                "Widen --rank-end or lower --min-volume."
            )

        positions = np.linspace(0, len(pool) - 1, num=count)
        idx = np.rint(positions).astype(int)
        selected = pool.iloc[idx].drop_duplicates("ticker").copy()
        if len(selected) != count:
            missing = count - len(selected)
            extras = pool.loc[~pool["ticker"].isin(selected["ticker"])].head(missing)
            selected = pd.concat([selected, extras], ignore_index=True)
        selected = selected.sort_values("market_cap_rank").reset_index(drop=True)
        selected["pilot_order"] = np.arange(1, len(selected) + 1)
        return selected


# Backward-compatible alias so older imports fail gracefully less often.
KisMarketCapUniverseProvider = KisKosdaqMasterProvider
