import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
start_date = "2020-01-01"
end_date = "2025-01-01"

rf = 0.03  # annual risk free rate

# download adjusted close prices
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)["Close"]

# daily returns
daily_returns = data.pct_change().dropna()

# annualized expected returns (mean daily * 252)
mu = daily_returns.mean().values * 252

# annualized covariance matrix (cov daily * 252)
cov = daily_returns.cov().values * 252

n = len(tickers)

def portfolio_return(w):
    return w @ mu

def portfolio_volatility(w):
    return np.sqrt(w @ cov @ w)

def neg_sharpe(w):
    r = portfolio_return(w)
    vol = portfolio_volatility(w)
    return -(r - rf) / vol

bounds = tuple((0, 0.4) for _ in range(n))  # max 40% per asset
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
w0 = np.ones(n) / n

# Tangency portfolio
tan_result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)
w_tan = tan_result.x

r_tan = portfolio_return(w_tan)
vol_tan = portfolio_volatility(w_tan)
sharpe_tan = (r_tan - rf) / vol_tan

print("Tickers:", tickers)
print("Tangency Portfolio Weights:", w_tan)
print("Tangency Return:", r_tan)
print("Tangency Volatility:", vol_tan)
print("Tangency Sharpe:", sharpe_tan)

# Efficient frontier
target_returns = np.linspace(mu.min(), mu.max(), 40)
frontier_returns = []
frontier_vols = []

for target in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target}
    )

    result = minimize(portfolio_volatility, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        w_opt = result.x
        frontier_returns.append(portfolio_return(w_opt))
        frontier_vols.append(portfolio_volatility(w_opt))

frontier_returns = np.array(frontier_returns)
frontier_vols = np.array(frontier_vols)

# Capital Market Line
cml_x = np.linspace(0, max(frontier_vols), 200)
cml_y = rf + sharpe_tan * cml_x

# Plot
plt.plot(frontier_vols, frontier_returns, marker="o", label="Efficient Frontier")
plt.plot(cml_x, cml_y, linestyle="--", label="Capital Market Line")

plt.scatter(vol_tan, r_tan, marker="*", s=300, label="Tangency Portfolio")

plt.xlabel("Volatility (Annualized)")
plt.ylabel("Return (Annualized)")
plt.title("Efficient Frontier (Real Stock Data)")
plt.grid(True)
plt.legend()
plt.show()
