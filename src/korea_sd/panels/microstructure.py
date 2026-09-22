from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


KEYS = ["ticker", "date"]


class PanelMergeError(ValueError):
    pass


def normalise_keys(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = df.copy()
    missing = set(KEYS) - set(out.columns)
    if missing:
        raise PanelMergeError(f"{name} is missing key columns: {sorted(missing)}")
    out["ticker"] = out["ticker"].astype(str).str.strip().str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    if out[KEYS].isna().any().any():
        raise PanelMergeError(f"{name} contains null/invalid ticker-date keys.")
    if out.duplicated(KEYS).any():
        examples = out.loc[out.duplicated(KEYS, keep=False), KEYS].head(10).to_dict("records")
        raise PanelMergeError(f"{name} contains duplicate ticker-date rows: {examples}")
    return out.sort_values(KEYS).reset_index(drop=True)


def merge_microstructure_panels(
    stock_master: pd.DataFrame,
    panels: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Strict left merge of independent panels onto Stock Master.

    Missing panel values remain NaN. Missing observations and observed zeros
    have different economic meanings and are never conflated.
    """
    merged = normalise_keys(stock_master, name="stock_master")
    original_rows = len(merged)
    for name, frame in panels.items():
        panel = normalise_keys(frame, name=name)
        non_keys = [c for c in panel.columns if c not in KEYS]
        collisions = sorted(set(non_keys) & set(merged.columns))
        if collisions:
            raise PanelMergeError(f"{name} would overwrite existing columns: {collisions}")
        merged = merged.merge(panel, on=KEYS, how="left", validate="one_to_one")
        if len(merged) != original_rows:
            raise PanelMergeError(
                f"Row count changed after merging {name}: {original_rows} -> {len(merged)}"
            )
    return merged.sort_values(KEYS).reset_index(drop=True)


def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create supply-normalized and trailing microstructure features."""
    out = normalise_keys(df, name="merged_panel")

    if "free_float_market_cap" not in out.columns and {"close", "free_float_shares"} <= set(out.columns):
        out["free_float_market_cap"] = (
            pd.to_numeric(out["close"], errors="coerce")
            * pd.to_numeric(out["free_float_shares"], errors="coerce")
        )

    def safe_div(num: str, den: str, dest: str) -> None:
        if num in out.columns and den in out.columns:
            denominator = pd.to_numeric(out[den], errors="coerce").replace(0, np.nan)
            numerator = pd.to_numeric(out[num], errors="coerce")
            out[dest] = numerator / denominator

    safe_div("lending_balance_shares", "free_float_shares", "lending_balance_float")
    safe_div("lending_net_change_shares", "free_float_shares", "lending_change_float")
    safe_div("short_volume", "free_float_shares", "short_volume_float")
    safe_div("short_value", "free_float_market_cap", "short_value_float_mcap")
    safe_div("credit_balance_shares", "free_float_shares", "credit_balance_float")
    safe_div("credit_net_change_shares", "free_float_shares", "credit_change_float")
    safe_div("program_netbuy_volume", "free_float_shares", "program_netbuy_float")
    safe_div("program_netbuy_value", "free_float_market_cap", "program_netbuy_float_mcap")

    out = out.sort_values(KEYS).reset_index(drop=True)
    grouped = out.groupby("ticker", sort=False, group_keys=False)
    for col in [
        "lending_change_float",
        "short_volume_float",
        "credit_change_float",
        "program_netbuy_float_mcap",
        "execution_strength",
    ]:
        if col not in out.columns:
            continue
        out[f"{col}_ma5"] = grouped[col].transform(
            lambda s: s.rolling(5, min_periods=3).mean()
        )
        out[f"{col}_ma20"] = grouped[col].transform(
            lambda s: s.rolling(20, min_periods=10).mean()
        )

    if {"execution_strength_ma5", "execution_strength_ma20"} <= set(out.columns):
        out["execution_strength_acceleration"] = (
            out["execution_strength_ma5"] - out["execution_strength_ma20"]
        )
    if {"program_netbuy_float_mcap_ma5", "program_netbuy_float_mcap_ma20"} <= set(out.columns):
        out["program_acceleration"] = (
            out["program_netbuy_float_mcap_ma5"]
            - out["program_netbuy_float_mcap_ma20"]
        )
    if {"credit_change_float_ma5", "credit_change_float_ma20"} <= set(out.columns):
        out["credit_acceleration"] = (
            out["credit_change_float_ma5"] - out["credit_change_float_ma20"]
        )
    if {"lending_change_float_ma5", "lending_change_float_ma20"} <= set(out.columns):
        out["lending_acceleration"] = (
            out["lending_change_float_ma5"] - out["lending_change_float_ma20"]
        )
    if {"short_volume_float_ma5", "short_volume_float_ma20"} <= set(out.columns):
        out["short_acceleration"] = (
            out["short_volume_float_ma5"] - out["short_volume_float_ma20"]
        )

    # Convenience ratios to existing market activity when available.
    safe_div("program_netbuy_value", "adv20_value", "program_netbuy_adv20")
    safe_div("short_value", "trading_value", "short_value_share_of_trading")
    return out
