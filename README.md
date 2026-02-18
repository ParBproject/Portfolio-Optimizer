# 📈 Markowitz Portfolio Optimizer

> **Advanced quantitative finance project** demonstrating convex quadratic programming for portfolio optimisation, interactive visualisation of the Efficient Frontier, and out-of-sample backtesting.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![CVXPY](https://img.shields.io/badge/Solver-CVXPY%2FclaRabel-orange)
![Plotly](https://img.shields.io/badge/Viz-Plotly%2FStreamlit-9cf)

---

## 🧠 Theory

### Markowitz Mean-Variance Framework (1952)

Given *n* risky assets with:
- **μ** ∈ ℝⁿ – annualised expected returns
- **Σ** ∈ ℝⁿˣⁿ – annualised covariance matrix

we choose portfolio weights **w** ∈ ℝⁿ to minimise risk for a given return target:

```
min   wᵀΣw            (portfolio variance)
s.t.  1ᵀw = 1         (fully invested)
      w ≥ 0            (no short-selling)
      μᵀw ≥ μ_target   (hit return target)
      w ≤ w_max        (diversification cap, optional)
```

This is a **convex Quadratic Programme (QP)** solved in polynomial time by CVXPY with the CLARABEL interior-point solver.

### Special Portfolios

| Portfolio | Description |
|-----------|-------------|
| **GMVP** | Global Minimum Variance: lowest risk regardless of return |
| **MSR**  | Max Sharpe Ratio: best risk-adjusted return, tangent to CML |
| **Efficient Frontier** | Pareto-optimal return/risk combinations |

### Maximum Sharpe via Dinkelbach Reformulation

Maximising `SR = (μᵀw - rf) / √(wᵀΣw)` is a fractional program.  
Substituting `y = w / [(μ−rf)ᵀw]` yields the equivalent convex QP:

```
min   yᵀΣy
s.t.  (μ−rf)ᵀy = 1,  1ᵀy ≥ 0,  y ≥ 0
```
then `w* = y / sum(y)`.

---

## 📁 Repository Structure

```
portfolio-optimizer/
│
├── data/
│   └── fetch_data.py              # Download & cache price data (yfinance)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: returns, distributions, correlations
│   ├── 02_optimization_basics.ipynb # Markowitz QP with CVXPY + SciPy comparison
│   ├── 03_efficient_frontier.ipynb  # Full frontier + interactive Plotly viz
│   └── 04_backtesting.ipynb       # Out-of-sample performance + drawdown
│
├── src/
│   ├── __init__.py
│   ├── data_handler.py            # Load, clean, split data; simulate portfolios
│   ├── optimizer.py               # CVXPY: min_variance, max_sharpe, frontier
│   ├── metrics.py                 # Sharpe, max drawdown, Calmar, summary table
│   └── visualization.py           # Plotly: frontier, backtest, heatmap, weights
│
├── app.py                         # 🚀 Streamlit interactive dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Setup

```bash
# 1. Clone
git clone https://github.com/yourname/portfolio-optimizer.git
cd portfolio-optimizer

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option A – Streamlit Dashboard (recommended)
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser. Configure tickers, dates,
risk-free rate, and weight constraints in the sidebar.

### Option B – Jupyter Notebooks
```bash
jupyter notebook
```
Run notebooks in order:
1. `01_data_exploration.ipynb` – understand the data
2. `02_optimization_basics.ipynb` – learn the math + CVXPY code
3. `03_efficient_frontier.ipynb` – generate the frontier interactively
4. `04_backtesting.ipynb` – evaluate out-of-sample performance

### Option C – Python API
```python
from src.data_handler  import load_data, simulate_random_portfolios
from src.optimizer     import min_variance, max_sharpe, efficient_frontier
from src.metrics       import compare_portfolios, portfolio_daily_returns, cumulative_wealth
from src.visualization import plot_efficient_frontier

# Load data
data = load_data(
    tickers   = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "SPY"],
    start     = "2015-01-01",
    end       = "2024-12-31",
    train_end = "2023-12-31",
)
mu, cov = data["mu"], data["cov"]

# Optimise
gmvp = min_variance(mu, cov)
msr  = max_sharpe(mu, cov, risk_free_rate=0.04)
ef   = efficient_frontier(mu, cov, n_points=60)
rand = simulate_random_portfolios(mu, cov, n_portfolios=5000)

# Visualise
fig = plot_efficient_frontier(ef, rand, gmvp, msr)
fig.show()
```

---

## 🎛️ Features

| Feature | Details |
|---------|---------|
| **Data** | yfinance · adjusted close · ffill gap handling · CSV cache |
| **Optimisation** | CVXPY (CLARABEL) + SciPy SLSQP fallback |
| **Constraints** | Long-only · fully-invested · optional per-asset cap |
| **Frontier** | 60-point efficient frontier sweep |
| **Visualisation** | Interactive Plotly: hover weights, Sharpe colourbar, CML |
| **Backtesting** | Fixed-weight & monthly rebalancing, vs equal-weight & SPY |
| **Metrics** | Ann. return/vol · Sharpe · Max Drawdown · Calmar ratio |
| **UI** | Streamlit dashboard with sidebar controls |

---

## 📊 Sample Outputs

### Efficient Frontier
The interactive chart shows ~5,000 random portfolios coloured by Sharpe ratio (Viridis colourscale), the optimised frontier curve, Capital Market Line, and highlighted GMVP / MSR markers with full weight breakdowns on hover.

### Backtest Metrics (illustrative, 2024 OOS)

| Metric | GMVP | Max Sharpe | Equal Weight |
|--------|------|-----------|--------------|
| Ann. Return | ~12% | ~18% | ~20% |
| Ann. Vol    | ~14% | ~19% | ~22% |
| Sharpe      | ~0.57 | ~0.74 | ~0.73 |
| Max DD      | ~9%  | ~13% | ~15% |

*(Actual values depend on the date range and market conditions.)*

---

## ⚠️ Limitations

- **Historical ≠ future**: optimisation uses past returns; no guarantee of future performance.
- **Normal returns assumed**: Markowitz assumes elliptical return distributions; real returns have fat tails and skewness.
- **No transaction costs or taxes**: live implementation must account for these.
- **Estimation risk**: small changes in μ can produce wildly different weights (the "error maximiser" problem).
- **Static rebalancing**: the simple backtest does not account for drift or cash flows.
- **Liquidity**: illiquid assets may not be tradeable at quoted prices.

---

## 🏗️ Why This Is Sophisticated

1. **Quadratic Programming**: direct implementation of the Markowitz QP using CVXPY, not a black-box library — you see every constraint.
2. **Dinkelbach Reformulation**: the max-Sharpe fractional programme is converted to a tractable convex QP analytically.
3. **Real-World Constraints**: long-only, fully-invested, and per-asset diversification caps.
4. **Out-of-Sample Validation**: backtesting with train/test split prevents look-ahead bias.
5. **Interactive Visualisation**: Plotly scatter with hover labels showing exact weights and statistics.
6. **Solver Comparison**: CVXPY (CLARABEL interior-point) vs SciPy SLSQP for pedagogical transparency.

---

## 📚 References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Sharpe, W. F. (1964). *Capital Asset Prices: A Theory of Market Equilibrium*.
- Boyd & Vandenberghe (2004). *Convex Optimization*. Cambridge University Press.
- CVXPY documentation: https://www.cvxpy.org

---

## 📝 License

MIT – free to use, fork, and modify.
