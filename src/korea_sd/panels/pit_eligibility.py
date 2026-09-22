from __future__ import annotations

import pandas as pd


def filter_full_period_free_float_eligible(
    market: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
    distributed_shares_col: str = "distributed_shares",
) -> tuple[list[str], pd.DataFrame]:
    """Return tickers with a valid PIT free-float snapshot at the start of their market history.

    A ticker is eligible only when there is at least one snapshot satisfying:
      - snapshot.asof_date <= ticker's first market date
      - distributed_shares > 0

    This deliberately requires full-period coverage. Tickers whose first valid snapshot
    arrives after the research window begins are excluded rather than filling early rows
    with future information or NaNs.
    """
    required_market = {"ticker", "date"}
    required_snap = {"ticker", "asof_date", distributed_shares_col}
    if missing := required_market - set(market.columns):
        raise ValueError(f"market missing columns: {sorted(missing)}")
    if missing := required_snap - set(snapshots.columns):
        raise ValueError(f"snapshots missing columns: {sorted(missing)}")

    m = market[["ticker", "date"]].copy()
    m["ticker"] = m["ticker"].astype(str).str.strip().str.zfill(6)
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.normalize()

    s = snapshots[["ticker", "asof_date", distributed_shares_col]].copy()
    s["ticker"] = s["ticker"].astype(str).str.strip().str.zfill(6)
    s["asof_date"] = pd.to_datetime(s["asof_date"], errors="coerce").dt.normalize()
    s[distributed_shares_col] = pd.to_numeric(s[distributed_shares_col], errors="coerce")

    universe = (
        [str(t).zfill(6) for t in tickers]
        if tickers is not None
        else sorted(m["ticker"].dropna().unique().tolist())
    )

    first_dates = m.groupby("ticker", sort=False)["date"].min().to_dict()
    eligible: list[str] = []
    rows: list[dict[str, object]] = []

    for ticker in universe:
        first_date = first_dates.get(ticker)
        if pd.isna(first_date) or first_date is None:
            rows.append({
                "ticker": ticker,
                "status": "excluded",
                "reason": "no_market_rows",
                "first_market_date": pd.NaT,
                "earliest_valid_snapshot": pd.NaT,
                "latest_snapshot_on_or_before_start": pd.NaT,
            })
            continue

        st = s.loc[s["ticker"] == ticker].sort_values("asof_date")
        valid = st.loc[
            st["asof_date"].notna()
            & (st[distributed_shares_col] > 0)
        ]
        earliest_valid = valid["asof_date"].min() if not valid.empty else pd.NaT
        prior = valid.loc[valid["asof_date"] <= first_date]
        latest_prior = prior["asof_date"].max() if not prior.empty else pd.NaT

        if prior.empty:
            reason = "no_positive_snapshot_on_or_before_first_market_date"
            rows.append({
                "ticker": ticker,
                "status": "excluded",
                "reason": reason,
                "first_market_date": first_date,
                "earliest_valid_snapshot": earliest_valid,
                "latest_snapshot_on_or_before_start": pd.NaT,
            })
            continue

        eligible.append(ticker)
        rows.append({
            "ticker": ticker,
            "status": "eligible",
            "reason": "",
            "first_market_date": first_date,
            "earliest_valid_snapshot": earliest_valid,
            "latest_snapshot_on_or_before_start": latest_prior,
        })

    report = pd.DataFrame(rows)
    return eligible, report
