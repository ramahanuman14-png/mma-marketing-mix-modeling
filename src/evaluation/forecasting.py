"""
forecasting.py — Phase 8: Revenue Forecasting
Generates 6-month forward-looking revenue forecasts
under 4 budget scenarios using the trained MMM.

Approach:
  1. Use MMM coefficients + Hill saturation to predict base revenue
  2. Apply historical seasonality factors per month
  3. Scale predictions to match actual historical GMV
  4. Run 4 scenarios: Status Quo, Optimal, Cut, Increase
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

REPORTS_DIR   = Path("reports/outputs")
FIGURES_DIR   = Path("reports/figures")
MEDIA_COLS    = ['TV','Digital','Sponsorship','Content_Marketing',
                 'Online_Marketing','Affiliates','SEM']
EC50          = np.array([40e6,14e6,100e6,5e6,180e6,65e6,45e6])


def hill(x: np.ndarray, ec50: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0)
    return x / (ec50 + x)


def predict_revenue(spend_vec: np.ndarray, beta: np.ndarray) -> float:
    return float(np.dot(beta, hill(spend_vec, EC50)))


def compute_seasonal_factors(df: pd.DataFrame) -> dict:
    """
    Compute monthly seasonality index from historical data.
    Index > 1 = above average month, < 1 = below average.
    """
    monthly_avg = df.groupby('Month')['total_gmv'].mean()
    overall_avg = monthly_avg.mean()
    factors     = (monthly_avg / overall_avg).to_dict()
    # Fill any missing months with 1.0
    for m in range(1, 13):
        factors.setdefault(m, 1.0)
    return factors


def build_scenarios(
    df: pd.DataFrame,
    opt_df: pd.DataFrame,
) -> dict:
    """
    Build 4 spend scenarios for forecasting.

    Returns dict of {scenario_name: spend_vector}
    """
    last3_spend  = df[MEDIA_COLS].tail(3).mean().values
    monthly_bud  = float(df['Total_Spend'].mean())
    opt_shares   = opt_df['Optimal_Spend_M'].values / opt_df['Optimal_Spend_M'].sum()
    opt_spend    = opt_shares * monthly_bud

    return {
        'Status Quo':             last3_spend,
        'Optimal Allocation':     opt_spend,
        'Budget Cut (-20%)':      last3_spend * 0.80,
        'Budget Increase (+20%)': last3_spend * 1.20,
    }


def generate_forecast(
    df: pd.DataFrame,
    beta: np.ndarray,
    scenarios: dict,
    n_months: int = 6,
    start_date: str = '2016-07-01',
) -> pd.DataFrame:
    """
    Generate monthly revenue forecast for each scenario.

    Args:
        df:         Historical feature DataFrame (clean, no outliers)
        beta:       MMM channel coefficients
        scenarios:  Dict of scenario_name → spend_vector
        n_months:   Number of months to forecast
        start_date: First forecast month

    Returns:
        DataFrame with forecasted GMV (₹M) per scenario
    """
    future_dates = pd.date_range(start_date, periods=n_months, freq='MS')
    seasonal     = compute_seasonal_factors(df)
    baseline_rev = df['total_gmv'].mean()

    # Scale factor: aligns model output to actual GMV scale
    base_pred = predict_revenue(scenarios['Status Quo'], beta)
    scale     = baseline_rev / base_pred if base_pred > 0 else 1.0
    logger.info(f"Forecast scale factor: {scale:.2f}")

    results = {}
    for name, spend in scenarios.items():
        pred       = predict_revenue(spend, beta)
        monthly    = []
        for d in future_dates:
            sf  = seasonal.get(d.month, 1.0)
            rev = pred * scale * sf / 1e6   # ₹ Millions
            monthly.append(rev)
        results[name] = monthly
        total = sum(monthly)
        logger.info(f"{name}: 6-month total = ₹{total:.0f}M")

    df_forecast = pd.DataFrame(results, index=future_dates)
    df_forecast.index.name = 'Date'
    return df_forecast


def run_forecast_pipeline():
    """Full forecasting pipeline — load → forecast → save."""
    logger.info("=== Forecast Pipeline Start ===")

    df      = pd.read_csv('data/processed/features.csv', parse_dates=['Date'])
    df      = df.sort_values('Date').reset_index(drop=True)
    df['Total_Spend'] = df[MEDIA_COLS].sum(axis=1)
    df_clean= df[df['total_gmv'] > 1e7].copy().reset_index(drop=True)

    roas_df = pd.read_csv('reports/outputs/roas_summary.csv')
    opt_df  = pd.read_csv('reports/outputs/budget_optimization.csv')
    beta    = roas_df['Coeff_Mean'].values

    scenarios   = build_scenarios(df_clean, opt_df)
    df_forecast = generate_forecast(df_clean, beta, scenarios)

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "revenue_forecast.csv"
    df_forecast.to_csv(out_path)
    logger.info(f"Forecast saved to {out_path}")

    # Print summary
    sq = df_forecast['Status Quo'].sum()
    print("\n=== 6-MONTH FORECAST SUMMARY ===")
    for col in df_forecast.columns:
        total = df_forecast[col].sum()
        diff  = total - sq
        sign  = '+' if diff >= 0 else ''
        print(f"  {col:<28} ₹{total:.0f}M  ({sign}{diff:.0f}M vs Status Quo)")

    logger.info("=== Forecast Pipeline Complete ===")
    return df_forecast
