"""
mmm_model.py — Phase 5: Bayesian Marketing Mix Model
Builds and samples a Bayesian MMM using PyMC.

Why Bayesian:
  - Only 12 months of data — OLS would overfit catastrophically
  - Priors enforce business constraints (ROAS > 0)
  - Posterior gives uncertainty ranges, not just point estimates
  - Credible intervals tell CMO: "SEM ROAS is ₹130M with 90% CI [16M, 284M]"
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pickle
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/processed")
REPORTS_DIR   = Path("reports/outputs")

SAT_COLS = [
    'TV_saturated', 'Digital_saturated', 'Sponsorship_saturated',
    'Content_Marketing_saturated', 'Online_Marketing_saturated',
    'Affiliates_saturated', 'SEM_saturated',
]
CHANNEL_LABELS = ['TV','Digital','Sponsorship','Content Mktg',
                  'Online Mktg','Affiliates','SEM']
PROMO_COLS     = ['has_promo', 'is_diwali_sale', 'is_christmas_sale']


def load_features() -> pd.DataFrame:
    """Load and prepare the feature matrix for modeling."""
    path = PROCESSED_DIR / "features.csv"
    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Remove confirmed outlier — August 2015 GMV = ₹419K (867x below median)
    outliers = df['total_gmv'] < 1e7
    logger.warning(f"Removing {outliers.sum()} outlier rows (GMV < ₹10M)")
    return df[~outliers].reset_index(drop=True)


def build_model(df: pd.DataFrame) -> tuple:
    """
    Build the Bayesian MMM.

    Model structure:
      GMV = baseline
            + Σ beta_media[i] * X_media[i]   (media contribution)
            + Σ beta_promo[j] * X_promo[j]   (event lift)
            + ε                               (noise)

    Priors:
      beta_media ~ HalfNormal(σ=200)  → forces positive ROAS
      beta_promo ~ Normal(μ=50, σ=100)→ promos likely positive but uncertain
      baseline   ~ HalfNormal(σ=300)  → organic revenue is positive
      sigma      ~ HalfNormal(σ=50)   → observation noise

    Returns:
        (model, y, X_media, X_promo)
    """
    y       = df['total_gmv'].values / 1e6   # scale to ₹ millions
    X_media = df[SAT_COLS].values
    X_promo = df[PROMO_COLS].values

    logger.info(f"Building model: {len(df)} observations, "
                f"{len(SAT_COLS)} media channels, {len(PROMO_COLS)} promo flags")

    with pm.Model() as model:
        # Media coefficients — positive only (business constraint)
        beta_media = pm.HalfNormal('beta_media', sigma=200, shape=len(SAT_COLS))

        # Promo lift — can be slightly negative (post-promo dip)
        beta_promo = pm.Normal('beta_promo', mu=50, sigma=100, shape=len(PROMO_COLS))

        # Organic baseline
        baseline = pm.HalfNormal('baseline', sigma=300)

        # Observation noise
        sigma = pm.HalfNormal('sigma', sigma=50)

        # Expected revenue
        mu = (baseline
              + pm.math.dot(X_media, beta_media)
              + pm.math.dot(X_promo, beta_promo))

        # Likelihood
        _ = pm.Normal('obs', mu=mu, sigma=sigma, observed=y)

    return model, y, X_media, X_promo


def sample_posterior(model, draws=1000, tune=1000, chains=2) -> az.InferenceData:
    """Sample the posterior distribution using NUTS."""
    logger.info(f"Sampling posterior: {draws} draws, {tune} tune, {chains} chains")
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            progressbar=False,
            random_seed=42,
            target_accept=0.9,
        )
    logger.info("Sampling complete")
    return trace


def compute_contributions(trace, y, X_media, X_promo) -> dict:
    """
    Compute mean channel contributions and model fit metrics.

    Returns dict with:
      - beta_mean: mean posterior coefficients
      - media_contrib: per-channel contribution per time period
      - y_pred: model predicted GMV
      - mape, r2: model accuracy metrics
      - roas_df: ROAS summary DataFrame
    """
    beta_samples  = trace.posterior['beta_media'].values.reshape(-1, len(SAT_COLS))
    promo_samples = trace.posterior['beta_promo'].values.reshape(-1, 3)
    base_samples  = trace.posterior['baseline'].values.flatten()

    beta_mean  = beta_samples.mean(axis=0)
    promo_mean = promo_samples.mean(axis=0)
    base_mean  = float(base_samples.mean())

    media_contrib = X_media * beta_mean
    promo_contrib = X_promo * promo_mean
    y_pred        = base_mean + media_contrib.sum(axis=1) + promo_contrib.sum(axis=1)

    mape = float(np.mean(np.abs((y - y_pred) / y)) * 100)
    r2   = float(1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))

    total_contrib = np.maximum(media_contrib.sum(axis=0), 0)
    pct           = total_contrib / total_contrib.sum() * 100

    roas_df = pd.DataFrame({
        'Channel':     CHANNEL_LABELS,
        'Coeff_Mean':  beta_mean.round(1),
        'CI_5pct':     np.percentile(beta_samples,  5, axis=0).round(1),
        'CI_95pct':    np.percentile(beta_samples, 95, axis=0).round(1),
        'Contrib_pct': pct.round(1),
    })

    logger.info(f"Model MAPE: {mape:.1f}%  |  R²: {r2:.3f}")
    return {
        'beta_mean': beta_mean, 'media_contrib': media_contrib,
        'y_pred': y_pred, 'mape': mape, 'r2': r2, 'roas_df': roas_df,
    }


def save_outputs(trace, results: dict):
    """Save trace and ROAS summary to disk."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "mmm_trace.pkl", 'wb') as f:
        pickle.dump(trace, f)
    logger.info("Trace saved to data/processed/mmm_trace.pkl")

    results['roas_df'].to_csv(REPORTS_DIR / "roas_summary.csv", index=False)
    logger.info("ROAS summary saved to reports/outputs/roas_summary.csv")


def run_mmm_pipeline():
    """End-to-end MMM: load → build → sample → evaluate → save."""
    logger.info("=== MMM Pipeline Start ===")
    df             = load_features()
    model, y, Xm, Xp = build_model(df)
    trace          = sample_posterior(model)
    results        = compute_contributions(trace, y, Xm, Xp)
    save_outputs(trace, results)
    logger.info("=== MMM Pipeline Complete ===")
    return trace, results
