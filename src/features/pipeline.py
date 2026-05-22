"""
pipeline.py — Phase 4: Feature Engineering Pipeline
Orchestrates all transformations in the correct order:
  1. Sort by date
  2. Impute nulls
  3. Adstock
  4. Saturation
  5. Promo dummies
  6. Seasonality
  7. Save to data/processed/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from src.features.transformations import (
    apply_adstock_all_channels,
    apply_saturation_all_channels,
    create_promo_dummies,
    create_seasonality_features,
    impute_media_nulls,
)

# ── Channel Configuration ────────────────────────────────────────────────────
# Decay rates (λ) — informed by channel type
# Higher λ = longer advertising memory
DECAY_RATES = {
    'TV':               0.45,   # brand channel — longer memory
    'Digital':          0.20,   # performance channel — short memory
    'Sponsorship':      0.60,   # brand building — very long memory
    'Content_Marketing':0.50,   # content lives long
    'Online_Marketing': 0.25,   # direct response — short memory
    'Affiliates':       0.15,   # transaction-driven — near-instant
    'SEM':              0.10,   # click → buy is immediate
    'Radio':            0.35,   # mid-range memory
    'Other':            0.30,
}

# EC50 values — set to median spend per channel (sensible prior)
# These will be tuned during Bayesian modeling in Phase 5
EC50_VALUES = {
    'TV':               40_000_000,
    'Digital':          14_000_000,
    'Sponsorship':      100_000_000,
    'Content_Marketing':5_000_000,
    'Online_Marketing': 180_000_000,
    'Affiliates':       65_000_000,
    'SEM':              45_000_000,
    'Radio':            11_000_000,
    'Other':            159_000_000,
}

# Slope values (α) — start at 1.0 (will be estimated in Phase 5)
SLOPE_VALUES = {ch: 1.0 for ch in DECAY_RATES}

MEDIA_CHANNELS = list(DECAY_RATES.keys())
PROCESSED_DIR  = Path("data/processed")


def build_feature_matrix(
    df_raw: pd.DataFrame,
    df_special: pd.DataFrame,
    save: bool = True
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Input:  raw secondfile DataFrame + special sales DataFrame
    Output: model-ready feature matrix saved to data/processed/

    Steps:
      1  Sort chronologically (CRITICAL for adstock)
      2  Impute Radio/Other nulls → 0
      3  Geometric adstock per channel
      4  Hill saturation per channel
      5  Promo event dummies
      6  Seasonality features
      7  Select final modeling columns
      8  Save to data/processed/features.csv
    """
    logger.info("=== Feature Engineering Pipeline Start ===")
    df = df_raw.copy()

    # ── Step 1: Sort by date ─────────────────────────────────────────────────
    df = df.sort_values('Date').reset_index(drop=True)
    logger.info(f"Step 1 ✅ Sorted by date: {df['Date'].min()} → {df['Date'].max()}")

    # ── Step 2: Impute nulls ─────────────────────────────────────────────────
    df = impute_media_nulls(df, MEDIA_CHANNELS)
    logger.info("Step 2 ✅ Null imputation complete")

    # ── Step 3: Adstock ──────────────────────────────────────────────────────
    df = apply_adstock_all_channels(df, MEDIA_CHANNELS, DECAY_RATES)
    logger.info("Step 3 ✅ Adstock transformations complete")

    # ── Step 4: Saturation ───────────────────────────────────────────────────
    df = apply_saturation_all_channels(df, MEDIA_CHANNELS, EC50_VALUES, SLOPE_VALUES)
    logger.info("Step 4 ✅ Hill saturation transformations complete")

    # ── Step 5: Promo dummies ────────────────────────────────────────────────
    df = create_promo_dummies(df, df_special, date_col='Date')
    logger.info("Step 5 ✅ Promo dummy variables created")

    # ── Step 6: Seasonality ──────────────────────────────────────────────────
    df = create_seasonality_features(df, date_col='Date')
    logger.info("Step 6 ✅ Seasonality features created")

    # ── Step 7: Final column selection ──────────────────────────────────────
    target = 'total_gmv'
    raw_spend_cols     = MEDIA_CHANNELS
    adstock_cols       = [f"{ch}_adstock"    for ch in MEDIA_CHANNELS]
    saturated_cols     = [f"{ch}_saturated"  for ch in MEDIA_CHANNELS]
    promo_cols         = ['has_promo','promo_days','is_diwali_sale',
                          'is_christmas_sale','is_independence',
                          'is_fashion_sale','is_valentines','is_rakshabandhan']
    seasonality_cols   = ['month_sin','month_cos','quarter','is_q4','time_index']
    meta_cols          = ['Date','Year','Month','NPS','total_Discount','total_Units']

    feature_cols = (meta_cols + [target] + raw_spend_cols +
                    adstock_cols + saturated_cols +
                    promo_cols + seasonality_cols)

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_features  = df[feature_cols].copy()

    logger.info(f"Step 7 ✅ Final feature matrix: {df_features.shape}")
    logger.info(f"  Columns: {list(df_features.columns)}")

    # ── Step 8: Save ─────────────────────────────────────────────────────────
    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "features.csv"
        df_features.to_csv(out_path, index=False)
        logger.info(f"Step 8 ✅ Saved to {out_path}")

    logger.info("=== Feature Engineering Pipeline Complete ===")
    return df_features
