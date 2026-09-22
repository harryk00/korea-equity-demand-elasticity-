#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


CORE_FEATURES = [
    "free_float_market_cap",
    "float_days",
    "float_turnover_1d",
    "lending_balance_float",
    "lending_acceleration",
    "short_volume_float",
    "short_acceleration",
    "credit_balance_float",
    "credit_acceleration",
    "program_netbuy_float_mcap",
    "program_acceleration",
    "execution_strength",
    "execution_strength_acceleration",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Purged 2025 -> 2026 OOS ML comparison versus rule baseline."
    )
    p.add_argument(
        "--input",
        default="data/pilot50/processed/stock_master_model.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="data/pilot50/analysis/ml_oos",
    )
    p.add_argument("--target", default="target_20d_30pct")
    p.add_argument("--event", default="surge_event_30pct")
    p.add_argument("--train-year", type=int, default=2025)
    p.add_argument("--test-year", type=int, default=2026)
    p.add_argument(
        "--purge-trading-days",
        type=int,
        default=20,
        help="Remove final N train-year market dates to prevent forward-label leakage.",
    )
    p.add_argument(
        "--top-frac",
        type=float,
        default=0.05,
        help="Daily fraction selected by ML probability in OOS.",
    )
    p.add_argument(
        "--min-cross-section",
        type=int,
        default=20,
    )
    p.add_argument(
        "--features",
        nargs="*",
        default=None,
    )
    return p.parse_args()


def require_sklearn():
    try:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            brier_score_loss,
        )
        return average_precision_score, roc_auc_score, brier_score_loss
    except Exception as e:
        raise SystemExit(
            "scikit-learn is required.\n"
            "Install with:\n"
            "  python -m pip install scikit-learn lightgbm catboost\n"
            f"Original error: {e}"
        )


def add_cs_ranks(
    df: pd.DataFrame,
    features: list[str],
    min_cross_section: int,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    rank_cols = []

    for feature in features:
        col = f"{feature}__cs_pct"
        out[col] = np.nan

        for _, idx in out.groupby("date", sort=False).groups.items():
            s = pd.to_numeric(out.loc[idx, feature], errors="coerce")
            valid = s.dropna()
            if len(valid) < min_cross_section:
                continue
            out.loc[valid.index, col] = valid.rank(
                method="average", pct=True
            ).to_numpy()

        rank_cols.append(col)

    return out, rank_cols


def baseline_rule(df: pd.DataFrame, min_cross_section: int) -> pd.Series:
    result = pd.Series(0, index=df.index, dtype="Int64")

    float_rank = pd.Series(np.nan, index=df.index, dtype=float)
    turnover_rank = pd.Series(np.nan, index=df.index, dtype=float)

    for _, idx in df.groupby("date", sort=False).groups.items():
        fm = pd.to_numeric(
            df.loc[idx, "free_float_market_cap"], errors="coerce"
        ).dropna()
        tr = pd.to_numeric(
            df.loc[idx, "float_turnover_1d"], errors="coerce"
        ).dropna()

        if len(fm) >= min_cross_section:
            float_rank.loc[fm.index] = fm.rank(
                method="average", pct=True
            ).to_numpy()
        if len(tr) >= min_cross_section:
            turnover_rank.loc[tr.index] = tr.rank(
                method="average", pct=True
            ).to_numpy()

    result[
        (float_rank <= 0.20)
        & (turnover_rank > 0.80)
    ] = 1

    return result


def daily_top_fraction(
    df: pd.DataFrame,
    prob_col: str,
    top_frac: float,
) -> pd.Series:
    out = pd.Series(0, index=df.index, dtype="Int64")

    for _, idx in df.groupby("date", sort=False).groups.items():
        s = pd.to_numeric(df.loc[idx, prob_col], errors="coerce").dropna()
        if s.empty:
            continue

        n = max(1, int(np.ceil(len(s) * top_frac)))
        selected = s.sort_values(ascending=False).head(n).index
        out.loc[selected] = 1

    return out


def classification_metrics(
    y_true: pd.Series,
    prob: pd.Series,
    avg_precision,
    roc_auc,
    brier_score,
) -> dict:
    m = y_true.notna() & prob.notna()
    y = y_true.loc[m].astype(int)
    p = prob.loc[m].astype(float)

    if y.empty or y.nunique() < 2:
        return {
            "rows": int(len(y)),
            "positive_rate": float(y.mean()) if len(y) else np.nan,
            "pr_auc": np.nan,
            "roc_auc": np.nan,
            "brier": np.nan,
        }

    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "pr_auc": float(avg_precision(y, p)),
        "roc_auc": float(roc_auc(y, p)),
        "brier": float(brier_score(y, p)),
    }


def selection_metrics(
    df: pd.DataFrame,
    signal_col: str,
    target: str,
    event: str | None,
    lookback: int = 5,
) -> dict:
    valid = df.dropna(subset=[target]).copy()

    overall = float(valid[target].mean()) if len(valid) else np.nan
    selected = valid[valid[signal_col] == 1].copy()
    nonselected = valid[valid[signal_col] == 0].copy()

    selected_rate = (
        float(selected[target].mean()) if len(selected) else np.nan
    )
    nonselected_rate = (
        float(nonselected[target].mean()) if len(nonselected) else np.nan
    )

    rr_overall = (
        selected_rate / overall
        if pd.notna(selected_rate) and pd.notna(overall) and overall > 0
        else np.nan
    )

    rr_nonselected = (
        selected_rate / nonselected_rate
        if (
            pd.notna(selected_rate)
            and pd.notna(nonselected_rate)
            and nonselected_rate > 0
        )
        else np.nan
    )

    out = {
        "signal": signal_col,
        "rows": int(len(valid)),
        "overall_target_rate": overall,
        "selected_observations": int(len(selected)),
        "selection_rate": float(len(selected) / len(valid)) if len(valid) else np.nan,
        "selected_target_rate": selected_rate,
        "relative_risk_vs_overall": rr_overall,
        "relative_risk_vs_nonselected": rr_nonselected,
    }

    if event and event in df.columns:
        event_rows = df[df[event] == 1].copy()
        exact = (
            float((event_rows[signal_col] == 1).mean())
            if len(event_rows)
            else np.nan
        )

        records = []
        for ticker, g in df.sort_values("date").groupby("ticker", sort=False):
            g = g.reset_index(drop=True)
            positions = np.flatnonzero(g[event].fillna(0).to_numpy() == 1)

            for pos in positions:
                start = max(0, pos - lookback)
                caught = bool(
                    (g.iloc[start:pos + 1][signal_col] == 1).any()
                )
                records.append(caught)

        out["independent_events"] = int(len(event_rows))
        out["event_recall_exact"] = exact
        out[f"event_recall_tminus{lookback}_to_t"] = (
            float(np.mean(records)) if records else np.nan
        )

    return out


def fit_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
):
    predictions = {}
    importances = []

    fitted = 0

    # LightGBM
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=100,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.2,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        predictions["lightgbm"] = model.predict_proba(X_test)[:, 1]

        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_.astype(float),
            "model": "lightgbm",
        })
        importances.append(imp)
        fitted += 1
    except Exception as e:
        print(f"[WARN] LightGBM unavailable/failed: {e}", file=sys.stderr)

    # CatBoost
    try:
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=600,
            depth=5,
            learning_rate=0.03,
            loss_function="Logloss",
            eval_metric="PRAUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_train, y_train)
        predictions["catboost"] = model.predict_proba(X_test)[:, 1]

        imp = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.get_feature_importance().astype(float),
            "model": "catboost",
        })
        importances.append(imp)
        fitted += 1
    except Exception as e:
        print(f"[WARN] CatBoost unavailable/failed: {e}", file=sys.stderr)

    if fitted == 0:
        raise SystemExit(
            "Neither LightGBM nor CatBoost could be trained.\n"
            "Install with:\n"
            "  python -m pip install scikit-learn lightgbm catboost"
        )

    if len(predictions) >= 2:
        arr = np.column_stack(list(predictions.values()))
        predictions["ensemble"] = arr.mean(axis=1)

    importance_df = (
        pd.concat(importances, ignore_index=True)
        if importances
        else pd.DataFrame()
    )

    return predictions, importance_df


def main():
    args = parse_args()
    avg_precision, roc_auc, brier_score = require_sklearn()

    src = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    required = {
        "ticker",
        "date",
        args.target,
        "free_float_market_cap",
        "float_turnover_1d",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    requested = args.features or CORE_FEATURES
    raw_features = [f for f in requested if f in df.columns]

    if len(raw_features) < 2:
        raise SystemExit(
            f"Too few usable features. Found: {raw_features}"
        )

    # Numeric coercion only.
    for c in raw_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Cross-sectional percentile versions stabilize levels across dates/regimes.
    df, rank_features = add_cs_ranks(
        df,
        features=raw_features,
        min_cross_section=args.min_cross_section,
    )

    model_features = raw_features + rank_features

    df["baseline_rule"] = baseline_rule(
        df,
        min_cross_section=args.min_cross_section,
    )

    train_year_df = df[df["date"].dt.year == args.train_year].copy()
    test = df[df["date"].dt.year == args.test_year].copy()

    # Purge last N distinct market dates from training because target looks N days forward.
    train_dates = sorted(train_year_df["date"].dropna().unique())
    if len(train_dates) <= args.purge_trading_days:
        raise SystemExit("Not enough train dates for requested purge.")

    purge_start = pd.Timestamp(train_dates[-args.purge_trading_days])
    train = train_year_df[train_year_df["date"] < purge_start].copy()

    train = train.dropna(subset=[args.target]).copy()
    test_model = test.dropna(subset=[args.target]).copy()

    y_train = train[args.target].astype(int)
    y_test = test_model[args.target].astype(int)

    # Median imputation learned ONLY from training sample.
    medians = train[model_features].median(numeric_only=True)
    X_train = train[model_features].fillna(medians).fillna(0.0)
    X_test = test_model[model_features].fillna(medians).fillna(0.0)

    predictions, importance_df = fit_models(
        X_train,
        y_train,
        X_test,
    )

    pred_panel = test_model[
        [
            "ticker",
            "date",
            args.target,
            *( [args.event] if args.event in test_model.columns else [] ),
            "baseline_rule",
        ]
    ].copy()

    model_metric_rows = []
    selection_rows = []

    # Baseline rule selection stats.
    selection_rows.append(
        {
            "model": "baseline_rule",
            **selection_metrics(
                pred_panel,
                signal_col="baseline_rule",
                target=args.target,
                event=args.event if args.event in pred_panel.columns else None,
            ),
        }
    )

    for name, prob in predictions.items():
        prob_col = f"prob_{name}"
        signal_col = f"signal_{name}_top{int(args.top_frac * 100)}pct"

        pred_panel[prob_col] = prob
        pred_panel[signal_col] = daily_top_fraction(
            pred_panel,
            prob_col=prob_col,
            top_frac=args.top_frac,
        )

        cm = classification_metrics(
            y_true=pred_panel[args.target],
            prob=pred_panel[prob_col],
            avg_precision=avg_precision,
            roc_auc=roc_auc,
            brier_score=brier_score,
        )
        cm["model"] = name
        model_metric_rows.append(cm)

        sm = selection_metrics(
            pred_panel,
            signal_col=signal_col,
            target=args.target,
            event=args.event if args.event in pred_panel.columns else None,
        )
        sm["model"] = name
        selection_rows.append(sm)

    metrics_df = pd.DataFrame(model_metric_rows)
    selection_df = pd.DataFrame(selection_rows)

    pred_panel.to_parquet(
        out_dir / "oos_predictions.parquet",
        index=False,
    )
    metrics_df.to_csv(
        out_dir / "model_metrics.csv",
        index=False,
    )
    selection_df.to_csv(
        out_dir / "selection_comparison.csv",
        index=False,
    )

    if not importance_df.empty:
        totals = importance_df.groupby("model")["importance"].transform("sum")
        importance_df["importance_normalized"] = np.where(
            totals > 0,
            importance_df["importance"] / totals,
            np.nan,
        )
        importance_df.to_csv(
            out_dir / "feature_importance.csv",
            index=False,
        )

    pd.DataFrame({
        "feature": model_features,
        "train_median": [medians.get(c, np.nan) for c in model_features],
    }).to_csv(
        out_dir / "training_imputation.csv",
        index=False,
    )

    print("\n=== Purged Split ===")
    print(f"train year: {args.train_year}")
    print(f"train rows: {len(train):,}")
    print(f"train last date: {train['date'].max().date()}")
    print(f"purge starts: {purge_start.date()}")
    print(f"test year: {args.test_year}")
    print(f"test rows: {len(test_model):,}")
    print(f"features: {len(model_features)} ({len(raw_features)} raw + {len(rank_features)} cs-rank)")

    print("\n=== 2026 OOS Model Metrics ===")
    if metrics_df.empty:
        print("No model metrics.")
    else:
        print(
            metrics_df[
                ["model", "rows", "positive_rate", "pr_auc", "roc_auc", "brier"]
            ]
            .sort_values("pr_auc", ascending=False)
            .to_string(index=False)
        )

    print("\n=== 2026 OOS Selection Comparison ===")
    cols = [
        "model",
        "signal",
        "selected_observations",
        "selection_rate",
        "selected_target_rate",
        "relative_risk_vs_overall",
        "relative_risk_vs_nonselected",
    ]
    if "event_recall_tminus5_to_t" in selection_df.columns:
        cols += [
            "independent_events",
            "event_recall_exact",
            "event_recall_tminus5_to_t",
        ]

    print(
        selection_df[cols]
        .sort_values("selected_target_rate", ascending=False)
        .to_string(index=False)
    )

    if not importance_df.empty:
        agg = (
            importance_df.groupby("feature", as_index=False)["importance_normalized"]
            .mean()
            .sort_values("importance_normalized", ascending=False)
            .head(20)
        )
        print("\n=== Mean Normalized Feature Importance: Top 20 ===")
        print(agg.to_string(index=False))

    print("\nInterpretation rule:")
    print("- Baseline is Small Float bottom 20% AND Float Turnover top 20%.")
    print(f"- ML selects top {args.top_frac:.1%} probability names cross-sectionally each date.")
    print("- Do NOT retune on 2026 after seeing these results.")
    print("- If ML beats baseline selection quality, next step is the SAME daily MTM backtest using ML signals.")
    print("- If ML does not beat baseline, keep the simpler rule.")

    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
