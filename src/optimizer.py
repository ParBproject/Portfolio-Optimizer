"""
optimizer.py
------------
Markowitz mean-variance portfolio optimisation using CVXPY.

Quadratic programme for a minimum-variance portfolio:

    min   wᵀ Σ w
    s.t.  1ᵀ w = 1
          w ≥ 0
          w ≤ w_max                 (optional)
          μᵀ w ≥ μ_target           (optional; omitted for the global minimum)

Maximum Sharpe, long only, uses the standard convex reformulation. With
excess return π = μ − rf·1:

    min   yᵀ Σ y
    s.t.  πᵀ y = 1
          y ≥ 0
          y ≤ w_max · 1ᵀ y          (optional)

and w* = y / 1ᵀ y. Failed or infeasible solves return ``weights=None``
and do not raise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import cvxpy as cp
except ImportError:  # Pyodide has SciPy, not CVXPY. The browser demo uses SLSQP.
    cp = None

_OPTIMAL = {"optimal", "optimal_inaccurate"}
_INFEASIBLE = {
    "infeasible",
    "infeasible_inaccurate",
    "unbounded",
    "unbounded_inaccurate",
}


def _failure(status: str) -> dict:
    return {
        "weights": None,
        "ret": np.nan,
        "vol": np.nan,
        "sharpe": np.nan,
        "status": status,
    }


def _inputs(mu, cov):
    """Return (mu, psd covariance, tickers, error_status)."""
    tickers = list(mu.index) if hasattr(mu, "index") else None
    mu_arr = np.asarray(mu, dtype=float).ravel()
    cov_arr = np.asarray(cov, dtype=float)
    n = mu_arr.size
    if tickers is None:
        tickers = [f"A{i}" for i in range(n)]
    if len(tickers) != n:
        raise ValueError("mu index length does not match the number of assets")
    if cov_arr.shape != (n, n):
        raise ValueError(f"cov shape {cov_arr.shape} does not match {n} assets")
    if not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(cov_arr)):
        return None, None, tickers, "non_finite_input"
    cov_arr = 0.5 * (cov_arr + cov_arr.T)
    eigvals, eigvecs = np.linalg.eigh(cov_arr)
    min_eig = float(eigvals.min()) if eigvals.size else 0.0
    if min_eig < -1e-6:
        return None, None, tickers, "covariance_not_psd"
    if min_eig < 0:
        eigvals = np.clip(eigvals, 0, None)
        cov_arr = (eigvecs * eigvals) @ eigvecs.T
    return mu_arr, cov_arr, tickers, None


def _cap_infeasible(n: int, max_weight: float | None) -> bool:
    if max_weight is None:
        return False
    if max_weight <= 0:
        return True
    return float(max_weight) * n < 1.0 - 1e-9


def _max_feasible_return(mu: np.ndarray, max_weight: float | None) -> float:
    """Highest long-only fully invested return under an optional name cap."""
    if max_weight is None or max_weight >= 1:
        return float(np.max(mu))
    remaining = 1.0
    total = 0.0
    for value in np.sort(np.asarray(mu, dtype=float))[::-1]:
        take = min(float(max_weight), remaining)
        total += take * float(value)
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


def _portfolio_stats(w, mu, cov, rf=0.04):
    w = np.asarray(w, dtype=float).ravel()
    ret = float(w @ mu)
    var = float(w @ cov @ w)
    if -1e-10 < var < 0:
        var = 0.0
    if var < 0:
        return ret, np.nan, np.nan
    vol = float(np.sqrt(var))
    if vol <= 0:
        return ret, vol, np.nan
    return ret, vol, (ret - rf) / vol


def _result(weights: pd.Series, mu, cov, rf, status: str) -> dict:
    ret, vol, sharpe = _portfolio_stats(weights.to_numpy(), mu, cov, rf)
    return {
        "weights": weights,
        "ret": ret,
        "vol": vol,
        "sharpe": sharpe,
        "status": status,
    }


def _finalize_weights(raw, tickers, max_weight, expect_simplex: bool) -> pd.Series | None:
    """Turn a solver vector into long-only weights that sum to 1.

    ``expect_simplex`` is true when the solver variable is already a
    portfolio (minimum variance). Maximum Sharpe solves for an unnormalised
    direction y, so only the normalised weights are checked against the cap.
    """
    w = np.asarray(raw, dtype=float).ravel()
    if w.size != len(tickers) or not np.all(np.isfinite(w)):
        return None
    w[w < 1e-12] = 0.0
    total = float(w.sum())
    if total <= 1e-12:
        return None
    if expect_simplex and abs(total - 1.0) > 1e-3:
        return None
    w = w / total
    if max_weight is not None and np.any(w > float(max_weight) + 1e-4):
        return None
    return pd.Series(w, index=list(tickers))


def _solve(problem: cp.Problem) -> str:
    """Solve with CLARABEL, then OSQP, then SCS. Infeasible stays infeasible."""
    last = "solver_error"
    for solver in (cp.CLARABEL, cp.OSQP, cp.SCS):
        try:
            problem.solve(solver=solver, verbose=False)
        except cp.SolverError:
            last = "solver_error"
            continue
        status = problem.status or "solver_error"
        if status in _OPTIMAL:
            return status
        if status in _INFEASIBLE:
            return status
        last = status
    return last


def min_variance(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    max_weight: float | None = None,
    target_return: float | None = None,
    risk_free_rate: float = 0.04,
) -> dict:
    """Global minimum variance, or minimum variance at a required return.

    ``risk_free_rate`` does not change the weights. It is used only for the
    reported Sharpe ratio, and it must be the same rate used everywhere else.
    """
    if cp is None:
        return min_variance_scipy(
            mu,
            cov,
            target_return=target_return,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )
    mu_arr, cov_arr, tickers, error = _inputs(mu, cov)
    if error:
        return _failure(error)
    n = len(mu_arr)
    if _cap_infeasible(n, max_weight):
        return _failure("infeasible_max_weight")
    if target_return is not None:
        ceiling = _max_feasible_return(mu_arr, max_weight)
        if target_return > ceiling + 1e-8:
            return _failure("infeasible_target_return")

    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        constraints.append(w <= max_weight)
    if target_return is not None:
        constraints.append(mu_arr @ w >= target_return)
    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cov_arr)), constraints)
    status = _solve(problem)
    if status not in _OPTIMAL or w.value is None:
        return _failure(status)
    cleaned = _finalize_weights(w.value, tickers, max_weight, expect_simplex=True)
    if cleaned is None:
        return _failure("invalid_weights")
    return _result(cleaned, mu_arr, cov_arr, risk_free_rate, status)


def _max_sharpe_qp(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
) -> dict:
    """Same maximum-Sharpe quadratic programme as the CVXPY solver, via SLSQP.

    Used when CVXPY is not installed (the in-browser demo). The programme is
    the one documented on ``max_sharpe``: minimise yᵀ Σ y subject to
    (μ − rf)ᵀ y = 1 and y ≥ 0, then normalise. It is not the direct
    negative-Sharpe objective in ``max_sharpe_scipy``.
    """
    mu_arr, cov_arr, tickers, error = _inputs(mu, cov)
    if error:
        return _failure(error)
    n = len(mu_arr)
    if _cap_infeasible(n, max_weight):
        return _failure("infeasible_max_weight")
    excess = mu_arr - float(risk_free_rate)
    if np.all(excess <= 0):
        return _failure("no_positive_excess_return")

    y0 = _feasible_sharpe_direction(excess, max_weight)
    if y0 is None:
        return _failure("no_positive_excess_return")

    def objective(y):
        return float(y @ cov_arr @ y)

    def gradient(y):
        return 2.0 * cov_arr @ y

    constraints: list[dict] = [{"type": "eq", "fun": lambda y: float(excess @ y - 1.0)}]
    if max_weight is not None:
        cap = float(max_weight)

        def cap_fun(index: int, limit: float):
            def fun(y, index=index, limit=limit):
                return limit * float(np.sum(y)) - float(y[index])

            return fun

        for i in range(n):
            constraints.append({"type": "ineq", "fun": cap_fun(i, cap)})

    res = minimize(
        objective,
        y0,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not res.success:
        return _failure(str(res.message))
    cleaned = _finalize_weights(res.x, tickers, max_weight, expect_simplex=False)
    if cleaned is None:
        return _failure("invalid_weights")
    return _result(cleaned, mu_arr, cov_arr, risk_free_rate, "optimal")


def _feasible_sharpe_direction(excess: np.ndarray, max_weight: float | None) -> np.ndarray | None:
    """A y ≥ 0 with (μ − rf)ᵀ y = 1 that already respects an optional name cap."""
    n = len(excess)
    cap = 1.0 if max_weight is None else float(max_weight)
    weights = np.zeros(n)
    remaining = 1.0
    for index in np.argsort(excess)[::-1]:
        if excess[index] <= 0 or remaining <= 1e-12:
            break
        take = min(cap, remaining)
        weights[index] = take
        remaining -= take
    if float(weights.sum()) <= 1e-12:
        return None
    scale = float(excess @ weights)
    if scale <= 1e-12:
        return None
    return weights / scale


def max_sharpe(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
) -> dict:
    """Long-only maximum Sharpe portfolio. Infeasible problems return weights=None."""
    if cp is None:
        return _max_sharpe_qp(
            mu,
            cov,
            risk_free_rate=risk_free_rate,
            max_weight=max_weight,
        )
    mu_arr, cov_arr, tickers, error = _inputs(mu, cov)
    if error:
        return _failure(error)
    n = len(mu_arr)
    if _cap_infeasible(n, max_weight):
        return _failure("infeasible_max_weight")
    excess = mu_arr - risk_free_rate
    if np.all(excess <= 0):
        return _failure("no_positive_excess_return")

    y = cp.Variable(n)
    constraints = [excess @ y == 1, y >= 0]
    if max_weight is not None:
        total = cp.Variable(nonneg=True)
        constraints += [cp.sum(y) == total, y <= max_weight * total]
    problem = cp.Problem(cp.Minimize(cp.quad_form(y, cov_arr)), constraints)
    status = _solve(problem)
    if status not in _OPTIMAL or y.value is None:
        return _failure(status)
    cleaned = _finalize_weights(y.value, tickers, max_weight, expect_simplex=False)
    if cleaned is None:
        return _failure("invalid_weights")
    return _result(cleaned, mu_arr, cov_arr, risk_free_rate, status)


def efficient_frontier(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    n_points: int = 60,
    max_weight: float | None = None,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """Minimum-variance portfolios from the global minimum up to the feasible max return.

    The top of the sweep respects ``max_weight``. Infeasible inputs produce
    an empty frame with the expected columns.
    """
    mu_arr, _cov_arr, tickers, error = _inputs(mu, cov)
    cols = ["ret", "vol", "sharpe"] + list(tickers)
    if error or _cap_infeasible(len(mu_arr), max_weight):
        return pd.DataFrame(columns=cols)

    gmvp = min_variance(mu, cov, max_weight=max_weight, risk_free_rate=risk_free_rate)
    if gmvp["weights"] is None:
        return pd.DataFrame(columns=cols)

    mu_lo = float(gmvp["ret"])
    mu_hi = _max_feasible_return(mu_arr, max_weight)
    if mu_hi < mu_lo:
        mu_hi = mu_lo
    elif mu_hi > mu_lo:
        mu_hi = mu_lo + 0.995 * (mu_hi - mu_lo)
    targets = np.linspace(mu_lo, mu_hi, n_points)
    records = []
    for target in targets:
        result = min_variance(
            mu,
            cov,
            max_weight=max_weight,
            target_return=float(target),
            risk_free_rate=risk_free_rate,
        )
        if result["weights"] is None:
            continue
        records.append(
            [result["ret"], result["vol"], result["sharpe"], *result["weights"].to_numpy()]
        )
    return pd.DataFrame(records, columns=cols).sort_values("vol").reset_index(drop=True)


def min_variance_scipy(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    target_return: float | None = None,
    max_weight: float | None = None,
    risk_free_rate: float = 0.04,
) -> dict:
    """Minimum-variance portfolio solved with SciPy SLSQP."""
    mu_arr, cov_arr, tickers, error = _inputs(mu, cov)
    if error:
        return _failure(error)
    n = len(mu_arr)
    if _cap_infeasible(n, max_weight):
        return _failure("infeasible_max_weight")
    if target_return is not None and target_return > _max_feasible_return(mu_arr, max_weight) + 1e-8:
        return _failure("infeasible_target_return")

    def portfolio_variance(w):
        return float(w @ cov_arr @ w)

    def grad_variance(w):
        return 2 * cov_arr @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if target_return is not None:
        target = float(target_return)
        constraints.append({"type": "ineq", "fun": lambda w, target=target: w @ mu_arr - target})

    upper = 1.0 if max_weight is None else float(max_weight)
    # A tight tolerance sometimes stops on a feasible point and reports a
    # line-search failure. Retry, then try a start tilted to higher-return names.
    starts = [np.ones(n) / n]
    greedy = _feasible_sharpe_direction(mu_arr - float(np.min(mu_arr)) + 1.0, max_weight)
    if greedy is not None:
        greedy = greedy / greedy.sum()
        starts.append(greedy)
    res = None
    for ftol, start in ((1e-12, starts[0]), (1e-9, starts[0]), *[(1e-9, s) for s in starts[1:]]):
        res = minimize(
            portfolio_variance,
            start,
            jac=grad_variance,
            method="SLSQP",
            bounds=[(0.0, upper)] * n,
            constraints=constraints,
            options={"ftol": ftol, "maxiter": 1000, "disp": False},
        )
        if res.success:
            break
    if res is None or not res.success:
        return _failure("solver_error" if res is None else str(res.message))
    cleaned = _finalize_weights(res.x, tickers, max_weight, expect_simplex=True)
    if cleaned is None:
        return _failure("invalid_weights")
    return _result(cleaned, mu_arr, cov_arr, risk_free_rate, "optimal")


def max_sharpe_scipy(
    mu: np.ndarray | pd.Series,
    cov: np.ndarray | pd.DataFrame,
    risk_free_rate: float = 0.04,
    max_weight: float | None = None,
) -> dict:
    """Maximum Sharpe portfolio via SciPy SLSQP (minimises negative Sharpe)."""
    mu_arr, cov_arr, tickers, error = _inputs(mu, cov)
    if error:
        return _failure(error)
    n = len(mu_arr)
    if _cap_infeasible(n, max_weight):
        return _failure("infeasible_max_weight")
    if np.all(mu_arr - risk_free_rate <= 0):
        return _failure("no_positive_excess_return")

    def neg_sharpe(w):
        var = float(w @ cov_arr @ w)
        if var <= 1e-18:
            return 1e6
        return -((w @ mu_arr) - risk_free_rate) / np.sqrt(var)

    upper = 1.0 if max_weight is None else float(max_weight)
    res = minimize(
        neg_sharpe,
        np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, upper)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )
    if not res.success:
        return _failure(str(res.message))
    cleaned = _finalize_weights(res.x, tickers, max_weight, expect_simplex=True)
    if cleaned is None:
        return _failure("invalid_weights")
    return _result(cleaned, mu_arr, cov_arr, risk_free_rate, "optimal")
