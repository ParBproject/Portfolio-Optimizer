import unittest

import numpy as np
import pandas as pd

from src.data_handler import shrink_covariance
from src.metrics import (
    historical_var_expected_shortfall,
    sortino_ratio,
)
from src.optimizer import min_variance


class OptimizerMetricsTests(unittest.TestCase):
    def test_covariance_shrinkage_preserves_variances(self) -> None:
        covariance = pd.DataFrame(
            [[0.04, 0.018], [0.018, 0.09]],
            index=["AAA", "BBB"],
            columns=["AAA", "BBB"],
        )
        shrunk = shrink_covariance(covariance, intensity=0.50)
        self.assertTrue(np.allclose(np.diag(shrunk), np.diag(covariance)))
        self.assertAlmostEqual(shrunk.loc["AAA", "BBB"], 0.009)

    def test_min_variance_sharpe_uses_requested_risk_free_rate(self) -> None:
        mu = pd.Series([0.10, 0.07], index=["AAA", "BBB"])
        covariance = pd.DataFrame(
            [[0.04, 0.01], [0.01, 0.02]],
            index=mu.index,
            columns=mu.index,
        )
        low_rf = min_variance(mu, covariance, risk_free_rate=0.01)
        high_rf = min_variance(mu, covariance, risk_free_rate=0.06)
        self.assertIsNotNone(low_rf["weights"])
        self.assertIsNotNone(high_rf["weights"])
        self.assertGreater(low_rf["sharpe"], high_rf["sharpe"])

    def test_expected_shortfall_is_not_better_than_var(self) -> None:
        returns = pd.Series([-0.08, -0.05, -0.02, 0.01, 0.02, 0.03])
        value_at_risk, expected_shortfall = historical_var_expected_shortfall(
            returns,
            confidence=0.80,
        )
        self.assertGreaterEqual(value_at_risk, 0.0)
        self.assertGreaterEqual(expected_shortfall, value_at_risk)

    def test_sortino_is_finite_with_downside_observations(self) -> None:
        returns = pd.Series([0.01, -0.015, 0.02, -0.005, 0.012])
        result = sortino_ratio(returns, risk_free_rate=0.0)
        self.assertTrue(np.isfinite(result))


if __name__ == "__main__":
    unittest.main()
