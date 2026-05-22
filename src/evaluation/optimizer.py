"""
optimizer.py — Phase 6: Budget Optimization
Solves: given a fixed total budget, what is the optimal
channel allocation to maximize predicted GMV?

Approach: Constrained nonlinear optimization via scipy SLSQP
  - Objective: maximize Σ beta[i] * hill(spend[i])
  - Constraint: Σ spend[i] = total_budget
  - Bounds: min_pct ≤ spend[i] ≤ max_pct of total budget
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pathlib import Path
from loguru import logger

CHANNELS = ['TV','Digital','Sponsorship','Content_Marketing',
            'Online_Marketing','Affiliates','SEM']
LABELS   = ['TV','Digital','Sponsorship','Content Mktg',
            'Online Mktg','Affiliates','SEM']

# Hill curve parameters from Phase 4
EC50  = np.array([40e6, 14e6, 100e6, 5e6, 180e6, 65e6, 45e6])
SLOPE = np.ones(7)

REPORTS_DIR = Path("reports/outputs")


def hill(x: np.ndarray, ec50: np.ndarray, slope: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0)
    return (x ** slope) / (ec50 ** slope + x ** slope)


def predict_revenue(spend_vec: np.ndarray, beta: np.ndarray) -> float:
    """Predict revenue from spend using Hill saturation + beta coefficients."""
    return float(np.dot(beta, hill(spend_vec, EC50, SLOPE)))


def optimize_budget(
    beta: np.ndarray,
    total_budget: float,
    min_pct: float = 0.02,
    max_pct: float = 0.50,
    n_starts: int = 3,
) -> np.ndarray:
    """
    Find optimal spend allocation for a given total budget.

    Args:
        beta:         Channel coefficients from MMM
        total_budget: Total spend to allocate (₹)
        min_pct:      Minimum fraction per channel (default 2%)
        max_pct:      Maximum fraction per channel (default 50%)
        n_starts:     Number of random starts (helps avoid local optima)

    Returns:
        Optimal spend vector (₹)
    """
    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - total_budget}]
    bounds      = [(total_budget * min_pct, total_budget * max_pct)] * len(beta)

    starting_points = [
        np.full(len(beta), total_budget / len(beta)),
        total_budget * np.array([0.10,0.10,0.10,0.10,0.30,0.15,0.15]),
        total_budget * beta / beta.sum(),   # spend proportional to ROAS
    ]

    best_result = None
    best_rev    = -np.inf

    for x0 in starting_points:
        x0 = x0 / x0.sum() * total_budget
        result = minimize(
            lambda x: -predict_revenue(x, beta),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10},
        )
        if -result.fun > best_rev:
            best_rev    = -result.fun
            best_result = result

    logger.info(f"Optimization complete: optimal revenue = ₹{best_rev:.1f}M")
    return best_result.x


def run_scenario_analysis(
    beta: np.ndarray,
    base_budget: float,
    budget_range: tuple = (0.5, 1.5),
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Run optimization across a range of total budgets.
    Answers: "What happens to revenue if we cut/increase budget by X%?"
    """
    budgets  = np.linspace(base_budget * budget_range[0],
                           base_budget * budget_range[1], n_points)
    revenues = []
    for b in budgets:
        opt = optimize_budget(beta, b)
        revenues.append(predict_revenue(opt, beta))

    return pd.DataFrame({
        'Total_Budget_M': (budgets / 1e6).round(1),
        'Optimal_Revenue_M': np.round(revenues, 1),
    })


def build_comparison_table(
    beta: np.ndarray,
    current_spend: np.ndarray,
    optimal_spend: np.ndarray,
) -> pd.DataFrame:
    """Build current vs optimal comparison DataFrame."""
    current_rev = predict_revenue(current_spend, beta)
    optimal_rev = predict_revenue(optimal_spend, beta)

    df = pd.DataFrame({
        'Channel':         LABELS,
        'Current_Spend_M': (current_spend / 1e6).round(1),
        'Optimal_Spend_M': (optimal_spend / 1e6).round(1),
        'Change_M':        ((optimal_spend - current_spend) / 1e6).round(1),
        'Change_Pct':      ((optimal_spend - current_spend) / current_spend * 100).round(1),
    })

    logger.info(f"Current GMV:  ₹{current_rev:.1f}M")
    logger.info(f"Optimal GMV:  ₹{optimal_rev:.1f}M")
    logger.info(f"Revenue lift: +{(optimal_rev-current_rev)/current_rev*100:.1f}%")
    return df


def run_optimization_pipeline(roas_path: str, features_path: str):
    """Full optimization pipeline."""
    logger.info("=== Budget Optimization Pipeline Start ===")

    roas_df  = pd.read_csv(roas_path)
    features = pd.read_csv(features_path)
    beta     = roas_df['Coeff_Mean'].values

    current_spend = features[CHANNELS].sum().values
    total_budget  = current_spend.sum()
    logger.info(f"Total budget: ₹{total_budget/1e6:.0f}M")

    # Optimize
    optimal_spend = optimize_budget(beta, total_budget)
    comparison_df = build_comparison_table(beta, current_spend, optimal_spend)

    # Scenario analysis
    scenario_df = run_scenario_analysis(beta, total_budget)

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(REPORTS_DIR / "budget_optimization.csv", index=False)
    scenario_df.to_csv(REPORTS_DIR / "scenario_analysis.csv",    index=False)

    logger.info("Outputs saved to reports/outputs/")
    logger.info("=== Budget Optimization Pipeline Complete ===")
    return comparison_df, scenario_df
