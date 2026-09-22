from __future__ import annotations

import numpy as np
import pandas as pd


class StockMasterError(ValueError):
    pass


def _norm_ticker(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.zfill(6)


def _normalise_market(market: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "date", "close", "volume"}
    missing = required - set(market.columns)
    if missing:
        raise StockMasterError(f"market missing columns: {sorted(missing)}")
    out = market.copy()
    out["ticker"] = _norm_ticker(out["ticker"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out[["ticker", "date"]].isna().any().any():
        raise StockMasterError("market contains invalid ticker/date keys.")
    if out.duplicated(["ticker", "date"]).any():
        raise StockMasterError("market contains duplicate ticker-date rows.")
    for c in [
        "open", "high", "low", "close", "volume", "trading_value",
        "market_cap", "shares_outstanding",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _normalise_free_float(
    free_float: pd.DataFrame,
    *,
    ff_ratio_col: str | None,
    ff_shares_col: str | None,
    distributed_shares_col: str | None,
) -> pd.DataFrame:
    required = {"ticker", "asof_date"}
    missing = required - set(free_float.columns)
    if missing:
        raise StockMasterError(f"free_float missing columns: {sorted(missing)}")
    out = free_float.copy()
    out["ticker"] = _norm_ticker(out["ticker"])
    out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce").dt.normalize()
    if out[["ticker", "asof_date"]].isna().any().any():
        raise StockMasterError("free_float contains invalid ticker/asof_date keys.")
    if out.duplicated(["ticker", "asof_date"]).any():
        raise StockMasterError("free_float contains duplicate ticker-asof_date rows.")

    explicit = [x for x in [ff_ratio_col, ff_shares_col, distributed_shares_col] if x]
    if len(explicit) > 1:
        raise StockMasterError("Choose only one primary free-float input.")
    if not explicit:
        if "free_float_shares" in out.columns:
            ff_shares_col = "free_float_shares"
        elif "free_float_ratio" in out.columns:
            ff_ratio_col = "free_float_ratio"
        elif "distributed_shares" in out.columns:
            distributed_shares_col = "distributed_shares"
        else:
            raise StockMasterError(
                "No free-float measure found. Provide free_float_shares, free_float_ratio, or distributed_shares."
            )

    result = out[["ticker", "asof_date"]].copy()
    if ff_shares_col:
        if ff_shares_col not in out.columns:
            raise StockMasterError(f"Column {ff_shares_col!r} not found.")
        result["ff_shares_snapshot"] = pd.to_numeric(out[ff_shares_col], errors="coerce")
        result["free_float_source"] = ff_shares_col
        if "shares_outstanding" in out.columns:
            result["ff_snapshot_shares_outstanding"] = pd.to_numeric(
                out["shares_outstanding"], errors="coerce"
            )
    elif distributed_shares_col:
        if distributed_shares_col not in out.columns:
            raise StockMasterError(f"Column {distributed_shares_col!r} not found.")
        result["ff_shares_snapshot"] = pd.to_numeric(
            out[distributed_shares_col], errors="coerce"
        )
        result["free_float_source"] = f"DART proxy:{distributed_shares_col}"
        if "shares_outstanding" in out.columns:
            result["ff_snapshot_shares_outstanding"] = pd.to_numeric(
                out["shares_outstanding"], errors="coerce"
            )
    else:
        if ff_ratio_col not in out.columns:
            raise StockMasterError(f"Column {ff_ratio_col!r} not found.")
        ratio = pd.to_numeric(out[ff_ratio_col], errors="coerce")
        if (ratio.dropna() < 0).any() or (ratio.dropna() > 100).any():
            raise StockMasterError("free-float ratio must be in 0..1 or 0..100.")
        ratio = ratio.where(ratio <= 1, ratio / 100.0)
        result["ff_ratio_snapshot"] = ratio
        result["free_float_source"] = ff_ratio_col
        if "shares_outstanding" in out.columns:
            result["ff_snapshot_shares_outstanding"] = pd.to_numeric(
                out["shares_outstanding"], errors="coerce"
            )
    return result.sort_values(["ticker", "asof_date"]).reset_index(drop=True)


def _pit_merge(market: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    snap_groups = {k: g.copy() for k, g in snapshots.groupby("ticker", sort=False)}
    snapshot_payload_cols = [c for c in snapshots.columns if c != "ticker"]
    for ticker, left in market.groupby("ticker", sort=False):
        right = snap_groups.get(ticker)
        if right is None or right.empty:
            tmp = left.copy()
            for c in snapshot_payload_cols:
                tmp[c] = np.nan
            pieces.append(tmp)
            continue
        l = left.sort_values("date")
        r = right.drop(columns=["ticker"]).sort_values("asof_date")
        pieces.append(
            pd.merge_asof(
                l,
                r,
                left_on="date",
                right_on="asof_date",
                direction="backward",
                allow_exact_matches=True,
            )
        )
    return pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def build_stock_master(
    market: pd.DataFrame,
    free_float: pd.DataFrame,
    *,
    ff_ratio_col: str | None = None,
    ff_shares_col: str | None = None,
    distributed_shares_col: str | None = None,
    require_free_float: bool = True,
) -> pd.DataFrame:
    market_n = _normalise_market(market)
    snapshots = _normalise_free_float(
        free_float,
        ff_ratio_col=ff_ratio_col,
        ff_shares_col=ff_shares_col,
        distributed_shares_col=distributed_shares_col,
    )
    out = _pit_merge(market_n, snapshots)

    if "ff_shares_snapshot" in out.columns:
        out["free_float_shares"] = pd.to_numeric(out["ff_shares_snapshot"], errors="coerce")
    elif "ff_ratio_snapshot" in out.columns:
        if "shares_outstanding" in out.columns:
            base = pd.to_numeric(out["shares_outstanding"], errors="coerce")
        else:
            base = pd.Series(np.nan, index=out.index)
        if "ff_snapshot_shares_outstanding" in out.columns:
            snap_base = pd.to_numeric(out["ff_snapshot_shares_outstanding"], errors="coerce")
            base = snap_base.fillna(base)
        if base.isna().all():
            raise StockMasterError(
                "free_float_ratio requires shares_outstanding in the snapshot or market panel."
            )
        out["free_float_shares"] = pd.to_numeric(out["ff_ratio_snapshot"], errors="coerce") * base
    else:
        raise StockMasterError("Could not construct free_float_shares.")

    if require_free_float:
        missing = out["free_float_shares"].isna() | (out["free_float_shares"] <= 0)
        if missing.any():
            examples = out.loc[missing, ["ticker", "date"]].head(10).to_dict("records")
            raise StockMasterError(
                "No valid point-in-time free float for some rows; "
                f"examples={examples}. Add an earlier snapshot or use --allow-missing-free-float."
            )

    # A historical market-price provider (e.g. KIS) may intentionally omit
    # listed-share counts because its output1 field is current, not historical.
    # In that case use the point-in-time share-status snapshot carried from DART.
    if "ff_snapshot_shares_outstanding" in out.columns:
        snap_shares = pd.to_numeric(out["ff_snapshot_shares_outstanding"], errors="coerce")
        if "shares_outstanding" in out.columns:
            market_shares = pd.to_numeric(out["shares_outstanding"], errors="coerce")
            out["shares_outstanding"] = market_shares.fillna(snap_shares)
        else:
            out["shares_outstanding"] = snap_shares

    if "shares_outstanding" in out.columns:
        denom = pd.to_numeric(out["shares_outstanding"], errors="coerce").replace(0, np.nan)
        out["free_float_ratio"] = out["free_float_shares"] / denom
    elif "ff_ratio_snapshot" in out.columns:
        out["free_float_ratio"] = out["ff_ratio_snapshot"]

    out["free_float_market_cap"] = (
        pd.to_numeric(out["close"], errors="coerce")
        * pd.to_numeric(out["free_float_shares"], errors="coerce")
    )

    # Derive market cap only when historically valid shares outstanding are
    # available. Do not backfill current KIS listed shares into past dates.
    if "shares_outstanding" in out.columns:
        derived_mcap = (
            pd.to_numeric(out["close"], errors="coerce")
            * pd.to_numeric(out["shares_outstanding"], errors="coerce")
        )
        if "market_cap" in out.columns:
            out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce").fillna(derived_mcap)
        else:
            out["market_cap"] = derived_mcap
    out = out.drop(
        columns=[
            c
            for c in ["ff_shares_snapshot", "ff_ratio_snapshot", "ff_snapshot_shares_outstanding"]
            if c in out.columns
        ]
    )
    return add_supply_features(out)


def add_supply_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {"ticker", "date", "close", "volume", "free_float_shares"}
    missing = required - set(out.columns)
    if missing:
        raise StockMasterError(f"Supply feature input missing columns: {sorted(missing)}")
    out["ticker"] = _norm_ticker(out["ticker"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    ff = pd.to_numeric(out["free_float_shares"], errors="coerce").replace(0, np.nan)
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["float_turnover_1d"] = out["volume"] / ff
    g = out.groupby("ticker", sort=False, group_keys=False)
    out["adv20_shares"] = g["volume"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    out["float_turnover_ma20"] = g["float_turnover_1d"].transform(
        lambda s: s.rolling(20, min_periods=10).mean()
    )
    out["float_days"] = ff / pd.to_numeric(out["adv20_shares"], errors="coerce").replace(0, np.nan)
    out["return_1d"] = g["close"].pct_change(fill_method=None)

    if "trading_value" in out.columns:
        out["trading_value"] = pd.to_numeric(out["trading_value"], errors="coerce")
        tv = out["trading_value"].replace(0, np.nan)
        out["amihud_1d"] = out["return_1d"].abs() / tv
        out["amihud_20d"] = out.groupby("ticker", sort=False)["amihud_1d"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
        out["adv20_value"] = out.groupby("ticker", sort=False)["trading_value"].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )
    return out
