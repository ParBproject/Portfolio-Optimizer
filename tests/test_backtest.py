import unittest

import numpy as np
import pandas as pd

from src.backtest import backtest_fixed_allocation, walk_forward_backtest


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2022-01-03", periods=420)
        daily = rng.normal(
            loc=[0.0004, 0.00025, 0.00015],
            scale=[0.011, 0.008, 0.006],
            size=(420, 3),
        )
        prices = 100.0 * np.cumprod(1.0 + daily, axis=0)
        self.prices = pd.DataFrame(prices, index=dates, columns=["AAA", "BBB", "CCC"])

    def test_fixed_allocation_is_reproducible(self) -> None:
        result = backtest_fixed_allocation(
            self.prices,
            [0.5, 0.3, 0.2],
            rebalance_every=21,
            transaction_cost_bps=0.0,
        )
        self.assertEqual(len(result.returns), len(self.prices) - 1)
        self.assertGreater(result.wealth.iloc[-1], 0.0)
        self.assertAlmostEqual(result.total_transaction_cost, 0.0)
        self.assertTrue(np.allclose(result.weights.sum(axis=1), 1.0))

    def test_costs_reduce_terminal_wealth(self) -> None:
        free = backtest_fixed_allocation(
            self.prices,
            [0.6, 0.2, 0.2],
            rebalance_every=5,
            transaction_cost_bps=0.0,
        )
        costly = backtest_fixed_allocation(
            self.prices,
            [0.6, 0.2, 0.2],
            rebalance_every=5,
            transaction_cost_bps=25.0,
        )
        self.assertGreater(costly.total_turnover, 0.0)
        self.assertGreater(costly.total_transaction_cost, 0.0)
        self.assertLess(costly.wealth.iloc[-1], free.wealth.iloc[-1])

    def test_walk_forward_uses_only_live_window_after_lookback(self) -> None:
        result = walk_forward_backtest(
            self.prices,
            strategy="min_variance",
            lookback_days=126,
            rebalance_every=21,
            transaction_cost_bps=5.0,
        )
        self.assertGreater(len(result.returns), 0)
        self.assertGreater(result.returns.index.min(), self.prices.index[100])
        self.assertTrue(np.allclose(result.weights.sum(axis=1), 1.0, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
