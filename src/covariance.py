"""Ledoit-Wolf shrinkage toward a scaled identity.

Ledoit and Wolf (2004), "A well-conditioned estimator for large-dimensional
covariance matrices", Journal of Multivariate Analysis. The implementation
follows scikit-learn's ``ledoit_wolf``: the covariance inside the formula is
the maximum-likelihood matrix (divide by T, not T − 1), and the target is
``(trace(S) / N) · I``.

``annualise_stats`` without shrinkage still uses the unbiased sample
covariance (divide by T − 1). These are different estimators. On the bundled
2015–2023 window the intensity is about 1%, so the published frontier stays
on the sample covariance. A 60-session window shrinks much harder.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Shrunk daily covariance and the intensity in ``[0, 1]``.

    Intensity 0 returns the maximum-likelihood covariance. Intensity 1
    returns ``(trace(S) / N) · I``. Expected returns are not shrunk.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a DataFrame")
    values = returns.to_numpy(dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("Need at least two observations to shrink a covariance.")
    if not np.all(np.isfinite(values)):
        raise ValueError("returns must be finite")

    n_samples, n_features = values.shape
    centered = values - values.mean(axis=0)
    if n_features == 1:
        variance = float(np.mean(centered[:, 0] ** 2))
        cov = pd.DataFrame(
            [[variance]],
            index=returns.columns,
            columns=returns.columns,
        )
        return cov, 0.0

    squared = centered**2
    diagonal_mean = np.sum(squared, axis=0) / n_samples
    mu = float(np.sum(diagonal_mean)) / n_features
    beta_raw = float(np.sum(squared.T @ squared))
    delta_raw = float(np.sum((centered.T @ centered) ** 2)) / n_samples**2
    beta = (1.0 / (n_features * n_samples)) * (beta_raw / n_samples - delta_raw)
    delta = delta_raw - 2.0 * mu * float(np.sum(diagonal_mean)) + n_features * mu**2
    delta /= n_features
    if delta <= 0.0 or beta <= 0.0:
        shrinkage = 0.0
    else:
        shrinkage = float(min(beta, delta) / delta)
    shrinkage = float(min(1.0, max(0.0, shrinkage)))

    sample = (centered.T @ centered) / n_samples
    shrunk = (1.0 - shrinkage) * sample
    shrunk.flat[:: n_features + 1] += shrinkage * mu
    cov = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
    return cov, shrinkage
