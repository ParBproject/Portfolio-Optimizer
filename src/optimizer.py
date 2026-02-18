"""
optimizer.py
------------
Markowitz mean-variance portfolio optimisation using CVXPY.

Theory
------
The Markowitz (1952) framework selects portfolio weights w ∈ ℝⁿ to sit on
the Efficient Frontier – the set of portfolios that maximise expected return
for a given level of risk (variance).

Quadratic programme (QP) for a target return μ_t:

    min   wᵀ Σ w          (minimise portfolio variance)
    s.t.  wᵀ μ  = μ_t     (hit target return)
          1ᵀ w  = 1        (fully invested)
          w ≥ 0            (long only)
          w ≤ w_max        (optional diversification cap)

Global Minimum Variance Portfolio (GMVP) drops the return constraint.

Maximum Sharpe Ratio (MSR) is solved via the Dinkelbach / Markowitz
two-fund separation trick:
    Reformulate as: min  yᵀ Σ y
                    s.t. (μ - rf)ᵀ y = 1,  1ᵀ y = κ,  y ≥ 0
    then  w* = y / sum(y).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize


# ── helpers ───────────────────────────────────────────────────────────────────

def _arrays(mu, cov):
    return np.asarray(mu, dtype=float), np.asarray(cov, dtype=float)


def _portfolio_stats(w, mu, cov, rf=0.04):
    ret = w @ mu
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol
    return ret, vol, sharpe


# ── CVXPY-based optimisers ────────────────────────────────────────────────────

def min_variance(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    max_weight: float | None = None,
    target_return: float | None = None,
) -> dict:
    """
    Global Minimum Variance Portfolio (GMVP), or constrained to a target return.

    Parameters
    ----------
    mu            : annualised expected returns (n,)
    cov           : annualised covariance matrix (n, n)
    max_weight    : optional upper bound per asset weight (e.g. 0.40)
    target_return : if supplied, solve for minimum variance *at* this return

    Returns
    -------
    dict with keys: weights, ret, vol, sharpe, status
    """
    mu_arr, cov_arr = _arrays(mu, cov)
    n = len(mu_arr)

    w = cp.Variable(n)
    objective    = cp.Minimize(cp.quad_form(w, cov_arr))
    constraints  = [cp.sum(w) == 1, w >= 0]

    if max_weight is not None:
        constraints.append(w <= max_weight)

    if target_return is not None:
        constraints.append(mu_arr @ w >= target_return)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return {"weights": None, "status": prob.status}

    w_vals = np.clip(w.value, 0, 1)
    w_vals /= w_vals.sum()          # re-normalise after numerical clip
    ret, vol, sharpe = _portfolio_stats(w_vals, mu_arr, cov_arr)
    tickers = list(mu.index) if hasattr(mu, "index") else [f"A{i}" for i in range(n)]

    return {
        "weights": pd.Series(w_vals, index=tickers),
        "ret":     ret,
        "vol":     vol,
        "sharpe":  sharpe,
        "status":  prob.status,
    }


def max_sharpe(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
) -> dict:
    """
    Maximum Sharpe Ratio portfolio via the Dinkelbach reformulation.

    The trick: let y = w/κ where κ = (μ-rf)ᵀw (excess return scalar).
    Then maximising Sharpe is equivalent to:
        min  yᵀ Σ y
        s.t. (μ - rf)ᵀ y = 1,  1ᵀ y ≥ 0,  y ≥ 0
    and w* = y / sum(y).

    Parameters
    ----------
    mu             : annualised expected returns
    cov            : annualised covariance matrix
    risk_free_rate : annualised risk-free rate (default 4 %)
    max_weight     : optional per-asset weight cap

    Returns
    -------
    dict with keys: weights, ret, vol, sharpe, status
    """
    mu_arr, cov_arr = _arrays(mu, cov)
    n = len(mu_arr)
    excess = mu_arr - risk_free_rate

    y = cp.Variable(n)
    objective   = cp.Minimize(cp.quad_form(y, cov_arr))
    constraints = [excess @ y == 1, cp.sum(y) >= 0, y >= 0]

    if max_weight is not None:
        # max_weight constraint translates to y_i / sum(y) <= max_weight
        # Approximate: y_i <= max_weight * sum(y). Use a scalar t = sum(y).
        t = cp.Variable(1, nonneg=True)
        constraints += [cp.sum(y) == t, y <= max_weight * t]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL, warm_start=True)

    if prob.status not in ("optimal", "optimal_inaccurate") or y.value is None:
        return {"weights": None, "status": prob.status}

    w_vals = np.clip(y.value, 0, None)
    if w_vals.sum() < 1e-10:
        return {"weights": None, "status": "degenerate"}
    w_vals /= w_vals.sum()

    ret, vol, sharpe = _portfolio_stats(w_vals, mu_arr, cov_arr, risk_free_rate)
    tickers = list(mu.index) if hasattr(mu, "index") else [f"A{i}" for i in range(n)]

    return {
        "weights": pd.Series(w_vals, index=tickers),
        "ret":     ret,
        "vol":     vol,
        "sharpe":  sharpe,
        "status":  prob.status,
    }


def efficient_frontier(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    n_points: int = 60,
    max_weight: float | None = None,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Compute the Efficient Frontier by sweeping target returns.

    For each target return μ_t ∈ [μ_min, μ_max] solve the minimum-variance QP
    and record (return, volatility, sharpe, weights).

    Parameters
    ----------
    mu           : annualised expected returns
    cov          : annualised covariance matrix
    n_points     : number of frontier points
    max_weight   : optional per-asset weight cap
    risk_free_rate : for Sharpe calculation

    Returns
    -------
    pd.DataFrame  columns: [ret, vol, sharpe, <ticker_0>, …, <ticker_n>]
                  sorted by volatility ascending
    """
    mu_arr, cov_arr = _arrays(mu, cov)
    tickers = list(mu.index) if hasattr(mu, "index") else [f"A{i}" for i in range(len(mu_arr))]

    # Feasible return range – from GMVP return to maximum individual asset return
    gmvp = min_variance(mu, cov, max_weight=max_weight)
    mu_min = gmvp["ret"] if gmvp["weights"] is not None else mu_arr.min()
    mu_max = mu_arr.max() * 0.995    # slight buffer for numerical stability

    targets = np.linspace(mu_min, mu_max, n_points)
    records = []

    for target in targets:
        result = min_variance(mu, cov, max_weight=max_weight, target_return=target)
        if result["weights"] is None:
            continue
        w = result["weights"].values
        ret, vol, sharpe = _portfolio_stats(w, mu_arr, cov_arr, risk_free_rate)
        records.append([ret, vol, sharpe, *w])

    cols = ["ret", "vol", "sharpe"] + tickers
    df = pd.DataFrame(records, columns=cols).sort_values("vol").reset_index(drop=True)
    return df


# ── SciPy fallback ────────────────────────────────────────────────────────────

def min_variance_scipy(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    target_return: float | None = None,
    max_weight: float | None = None,
) -> dict:
    """
    Minimum-variance portfolio solved with SciPy SLSQP (fallback / comparison).

    Uses scipy.optimize.minimize with analytic Jacobian of the variance.
    """
    mu_arr, cov_arr = _arrays(mu, cov)
    n = len(mu_arr)
    tickers = list(mu.index) if hasattr(mu, "index") else [f"A{i}" for i in range(n)]

    def portfolio_variance(w):
        return float(w @ cov_arr @ w)

    def grad_variance(w):
        return 2 * cov_arr @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        constraints.append({"type": "ineq", "fun": lambda w: w @ mu_arr - target_return})

    bounds = [(0, max_weight or 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(
        portfolio_variance,
        w0,
        jac=grad_variance,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not res.success:
        return {"weights": None, "status": res.message}

    w_vals = np.clip(res.x, 0, 1)
    w_vals /= w_vals.sum()
    ret, vol, sharpe = _portfolio_stats(w_vals, mu_arr, cov_arr)

    return {
        "weights": pd.Series(w_vals, index=tickers),
        "ret":     ret,
        "vol":     vol,
        "sharpe":  sharpe,
        "status":  "optimal",
    }


def max_sharpe_scipy(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
) -> dict:
    """
    Maximum Sharpe ratio portfolio via SciPy SLSQP (minimise negative Sharpe).
    """
    mu_arr, cov_arr = _arrays(mu, cov)
    n = len(mu_arr)
    tickers = list(mu.index) if hasattr(mu, "index") else [f"A{i}" for i in range(n)]

    def neg_sharpe(w):
        ret = w @ mu_arr
        vol = np.sqrt(w @ cov_arr @ w)
        return -(ret - risk_free_rate) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, max_weight or 1.0)] * n
    w0 = np.ones(n) / n

    res = minimize(
        neg_sharpe, w0,
        method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if not res.success:
        return {"weights": None, "status": res.message}

    w_vals = np.clip(res.x, 0, 1);  w_vals /= w_vals.sum()
    ret, vol, sharpe = _portfolio_stats(w_vals, mu_arr, cov_arr, risk_free_rate)

    return {
        "weights": pd.Series(w_vals, index=tickers),
        "ret":     ret, "vol": vol, "sharpe": sharpe, "status": "optimal",
    }
