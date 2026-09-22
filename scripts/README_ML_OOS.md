# Purged OOS ML vs Baseline

This patch starts the ML stage only after the rule-based signal survived:
- univariate analysis
- 2D supply-demand grid
- 2025 -> 2026 OOS validation
- executable next-open trade test
- parameter robustness
- daily mark-to-market portfolio test

## Split

Training:
- 2025 only
- final 20 distinct 2025 market dates are purged

Why purge?
`target_20d_30pct` uses future 20-day prices. Without purging, late-2025 training
labels would depend on 2026 prices, contaminating the OOS boundary.

Test:
- 2026 only
- untouched for model fitting and imputation

## Features

Default raw features:
- free_float_market_cap
- float_days
- float_turnover_1d
- lending_balance_float
- lending_acceleration
- short_volume_float
- short_acceleration
- credit_balance_float
- credit_acceleration
- program_netbuy_float_mcap
- program_acceleration
- execution_strength
- execution_strength_acceleration

The script also creates a per-date cross-sectional percentile-rank version of each
available feature. Models therefore receive raw + rank versions.

## Models

- LightGBM
- CatBoost
- simple average ensemble when both are installed

Install if needed:

```bash
python -m pip install scikit-learn lightgbm catboost
```

## OOS selection

The model is evaluated two ways:

1. Probability metrics:
   - PR-AUC
   - ROC-AUC
   - Brier score

2. Investment-candidate quality:
   - each 2026 date selects model-probability top 5%
   - compare +30% target rate and relative risk with the fixed rule baseline
   - compare independent-event recall

The fixed baseline remains:
- free-float market cap bottom 20%
- float turnover top 20%

## Run

```bash
python scripts/train_oos_ml.py \
  --input data/pilot50/processed/stock_master_model.parquet \
  --out-dir data/pilot50/analysis/ml_oos
```

## Outputs

- `oos_predictions.parquet`
- `model_metrics.csv`
- `selection_comparison.csv`
- `feature_importance.csv`
- `training_imputation.csv`

If the ML selection beats the baseline out of sample, the next step is to feed the
ML signal into the exact same daily mark-to-market portfolio engine.
