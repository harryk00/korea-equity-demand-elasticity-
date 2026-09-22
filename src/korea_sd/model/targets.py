from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


class TargetBuildError(ValueError):
    pass


def _future_max(series: pd.Series, horizon: int) -> pd.Series:
    # After shift(-1), reverse-rolling maps each date t to max(t+1..t+h).
    return (
        series.shift(-1)
        .iloc[::-1]
        .rolling(horizon, min_periods=horizon)
        .max()
        .iloc[::-1]
    )


def add_forward_targets(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = (5, 10, 20, 40),
    target_horizon: int = 20,
    thresholds: Iterable[float] = (0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    required = {"ticker", "date", "close", "high"}
    missing = required - set(df.columns)
    if missing:
        raise TargetBuildError(f"Target input missing columns: {sorted(missing)}")

    horizons = tuple(sorted({int(h) for h in horizons}))
    if target_horizon not in horizons:
        horizons = tuple(sorted(set(horizons) | {int(target_horizon)}))
    if any(h < 1 for h in horizons):
        raise TargetBuildError("Horizons must be positive integers.")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    if out[["ticker", "date", "close", "high"]].isna().any().any():
        raise TargetBuildError("Target input contains invalid ticker/date/close/high values.")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    pieces: list[pd.DataFrame] = []
    for _, g in out.groupby("ticker", sort=False):
        g = g.copy()
        close = g["close"]
        high = g["high"]
        for horizon in horizons:
            future_high = _future_max(high, horizon)
            g[f"fwd_max_return_{horizon}d"] = future_high / close - 1.0
        base = g[f"fwd_max_return_{target_horizon}d"]
        for threshold in thresholds:
            pct = int(round(float(threshold) * 100))
            g[f"target_{target_horizon}d_{pct}pct"] = np.where(
                base.notna(), (base >= float(threshold)).astype(int), np.nan
            )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def add_cooldown_events(
    df: pd.DataFrame,
    *,
    target_col: str = "target_20d_30pct",
    event_col: str = "surge_event_30pct",
    cooldown: int = 20,
) -> pd.DataFrame:
    if target_col not in df.columns:
        raise TargetBuildError(f"Missing target column: {target_col}")
    if cooldown < 0:
        raise TargetBuildError("cooldown must be >= 0")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    out[event_col] = 0

    for _, idx in out.groupby("ticker", sort=False).groups.items():
        positions = list(idx)
        values = out.loc[positions, target_col].to_numpy()
        last_event_local = -(10**9)
        for local_i, value in enumerate(values):
            if pd.isna(value) or float(value) != 1.0:
                continue
            if local_i - last_event_local > cooldown:
                out.at[positions[local_i], event_col] = 1
                last_event_local = local_i
    return out
