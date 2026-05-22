"""
transformations.py — Phase 4: Feature Engineering
Three core MMM transformations:
  1. Adstock     — advertising carryover effect (memory)
  2. Hill        — diminishing returns / saturation curve
  3. Promo dummies — event flags from SpecialSale calendar

Why these matter:
  - Without adstock: model thinks TV spend only works the week it airs
  - Without saturation: model thinks doubling spend doubles revenue forever
  - Without promos: model attributes promo-driven spikes to media channels

These are the transformations Google, Meta, and Nielsen use in production MMM.
"""

import numpy as np
import pandas as pd
from loguru import logger


# ── 1. ADSTOCK TRANSFORMATION ───────────────────────────────────────────────
# Concept: Advertising has a "memory" — a TV ad seen in October still
# influences buying behaviour in November and December.
# Geometric decay models this: each week the effect decays by factor λ.
#
# Formula: adstock[t] = spend[t] + λ * adstock[t-1]
#   λ = 0.0  → no carryover (instantaneous effect only)
#   λ = 0.5  → half the effect carries to next period
#   λ = 0.9  → strong memory (e.g. brand campaigns)
#
# TV typically: λ = 0.3–0.5
# Digital typically: λ = 0.1–0.3  (faster decay, performance-driven)
# Sponsorship: λ = 0.5–0.8        (brand building, slow decay)

def geometric_adstock(spend: np.ndarray, decay: float) -> np.ndarray:
    """
    Apply geometric adstock decay to a spend array.

    Args:
        spend: Array of media spend values (chronological order)
        decay: Decay rate λ in [0, 1]. Higher = longer memory.

    Returns:
        Array of adstocked values, same length as input.
    """
    assert 0.0 <= decay <= 1.0, f"Decay must be in [0,1], got {decay}"
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked


def apply_adstock_all_channels(
    df: pd.DataFrame,
    channels: list,
    decay_rates: dict
) -> pd.DataFrame:
    """
    Apply geometric adstock to all media channels.

    Args:
        df: DataFrame sorted by date (CRITICAL — must be chronological)
        channels: List of channel column names
        decay_rates: Dict mapping channel name → decay rate λ

    Returns:
        DataFrame with new columns: {channel}_adstock
    """
    df = df.copy()
    for ch in channels:
        if ch not in df.columns:
            logger.warning(f"Channel '{ch}' not found — skipping adstock")
            continue
        decay = decay_rates.get(ch, 0.3)
        spend = df[ch].fillna(0).values
        df[f"{ch}_adstock"] = geometric_adstock(spend, decay)
        logger.info(f"Adstock applied: {ch} (λ={decay})")
    return df


# ── 2. HILL SATURATION TRANSFORMATION ──────────────────────────────────────
# Concept: Doubling ad spend does NOT double revenue.
# The first ₹10M of TV buys a lot. The next ₹10M buys less. The next buys even less.
# This is the Law of Diminishing Returns.
#
# Hill function models this S-curve:
#   hill(x) = x^α / (EC50^α + x^α)
#
#   EC50 (half-saturation point): spend level where you get 50% of max effect
#   α (slope/shape): steepness of the S-curve
#     α < 1 → concave curve (diminishing returns from start)
#     α > 1 → S-shaped curve (build-up then saturation)
#
# Output is always in [0, 1] — think of it as "% of max possible effect achieved"

def hill_saturation(spend: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """
    Apply Hill saturation curve to adstocked spend.

    Args:
        spend: Adstocked spend array (after geometric_adstock)
        ec50:  Half-saturation point — spend level for 50% max effect
               Set to median or mean spend as a starting point
        slope: Shape parameter α. Start with 1.0.

    Returns:
        Array of saturation-transformed values in [0, 1]
    """
    assert ec50 > 0, f"EC50 must be positive, got {ec50}"
    assert slope > 0, f"Slope must be positive, got {slope}"
    spend = np.maximum(spend, 0)  # ensure non-negative
    return (spend ** slope) / (ec50 ** slope + spend ** slope)


def apply_saturation_all_channels(
    df: pd.DataFrame,
    channels: list,
    ec50_values: dict,
    slope_values: dict
) -> pd.DataFrame:
    """
    Apply Hill saturation to all adstocked channels.
    Always run AFTER apply_adstock_all_channels.

    Args:
        df: DataFrame with {channel}_adstock columns
        channels: List of channel names
        ec50_values: Dict channel → EC50 value
        slope_values: Dict channel → slope α

    Returns:
        DataFrame with new columns: {channel}_saturated
    """
    df = df.copy()
    for ch in channels:
        adstock_col = f"{ch}_adstock"
        if adstock_col not in df.columns:
            logger.warning(f"'{adstock_col}' not found — run adstock first")
            continue
        ec50  = ec50_values.get(ch, df[adstock_col].median())
        slope = slope_values.get(ch, 1.0)
        df[f"{ch}_saturated"] = hill_saturation(
            df[adstock_col].values, ec50, slope
        )
        logger.info(f"Saturation applied: {ch} (EC50={ec50:.0f}, α={slope})")
    return df


# ── 3. PROMO / EVENT DUMMY VARIABLES ───────────────────────────────────────
# Concept: Big sales events (Diwali, Christmas) cause revenue spikes
# that have NOTHING to do with media spend that month.
# If we don't control for them, the model will wrongly attribute
# promo-driven revenue to whatever channel happened to air that month.
#
# Solution: Create binary flag columns — 1 if promo occurred, 0 otherwise.

def create_promo_dummies(
    df: pd.DataFrame,
    special: pd.DataFrame,
    date_col: str = 'Date'
) -> pd.DataFrame:
    """
    Create monthly promo event dummy variables.

    Args:
        df: Main modeling DataFrame (monthly granularity)
        special: SpecialSale DataFrame with Date and Sales Name columns
        date_col: Name of date column in df

    Returns:
        DataFrame with added columns:
          - has_promo: 1 if any promo event occurred that month
          - promo_days: count of promo days in that month
          - Individual event flags for major events
    """
    df = df.copy()

    # Match special sales to months in modeling data
    special_monthly = special.copy()
    special_monthly['month_period'] = special_monthly['Date'].dt.to_period('M')
    df['month_period'] = df[date_col].dt.to_period('M')

    # Count of promo days per month
    promo_counts = special_monthly.groupby('month_period').size().reset_index()
    promo_counts.columns = ['month_period', 'promo_days']
    df = df.merge(promo_counts, on='month_period', how='left')
    df['promo_days'] = df['promo_days'].fillna(0).astype(int)
    df['has_promo'] = (df['promo_days'] > 0).astype(int)

    # Major event flags — group by event family
    major_events = {
        'is_diwali_sale':    ['Big Diwali Sale', 'Daussera sale'],
        'is_christmas_sale': ['Christmas & New Year Sale'],
        'is_independence':   ['Independence Sale', 'Republic Day'],
        'is_fashion_sale':   ['FHSD', 'BED', 'BSD-5', 'Pacman'],
        'is_valentines':     ["Valentine's Day"],
        'is_rakshabandhan':  ['Rakshabandhan Sale'],
    }

    for flag, event_names in major_events.items():
        event_months = special_monthly[
            special_monthly['Sales Name'].isin(event_names)
        ]['month_period'].unique()
        df[flag] = df['month_period'].isin(event_months).astype(int)
        logger.info(f"Promo flag: {flag} → {df[flag].sum()} months flagged")

    df = df.drop(columns=['month_period'])
    return df


# ── 4. SEASONALITY FEATURES ─────────────────────────────────────────────────
# Captures predictable patterns that repeat every year.
# Month dummies are simplest. Sine/cosine encoding is smoother.

def create_seasonality_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Create seasonality and time features.

    Returns DataFrame with:
      - month_sin, month_cos: cyclical encoding of month
      - quarter: Q1/Q2/Q3/Q4
      - is_q4: flag for Oct–Dec (peak season)
      - time_index: 0,1,2,... trend variable
    """
    df = df.copy()
    df['month_num']  = df[date_col].dt.month
    df['quarter']    = df[date_col].dt.quarter
    df['is_q4']      = (df['quarter'] == 4).astype(int)

    # Cyclical encoding — preserves that Dec(12) is close to Jan(1)
    df['month_sin']  = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos']  = np.cos(2 * np.pi * df['month_num'] / 12)

    # Linear trend — captures secular growth/decline
    df['time_index'] = range(len(df))

    logger.info("Seasonality features created: month_sin, month_cos, quarter, is_q4, time_index")
    return df


# ── 5. NULL IMPUTATION ───────────────────────────────────────────────────────
# Radio and Other have 9 nulls out of 12 rows.
# Strategy: fill with 0 — absence of spend, not missing data.
# These channels were simply not used in most months.

def impute_media_nulls(df: pd.DataFrame, channels: list) -> pd.DataFrame:
    """
    Impute null media spend values with 0.
    Rationale: null = channel not used that month, not missing data.
    """
    df = df.copy()
    for ch in channels:
        null_count = df[ch].isnull().sum()
        if null_count > 0:
            df[ch] = df[ch].fillna(0)
            logger.info(f"Imputed {null_count} nulls in '{ch}' with 0")
    return df
