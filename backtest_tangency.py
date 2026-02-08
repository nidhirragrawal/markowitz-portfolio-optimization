import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA"]
benchmark = "SPY"

start_date = "2020-01-01"
end_date = "2025-01-01"

# Tangency weights from your optimizer output
weights = np.array([0.4, 0.2, 0.0, 0.4])
print("\nWeights:")
for t, w in zip(tickers, weights):
    print(f"{t}: {w*100:.2f}%")

rf = 0.03  # annual risk free

# Download adjusted prices
data = yf.download(tickers + [benchmark], start=start_date, end=end_date, auto_adjust=True)["Close"]

prices = data[tickers]
spy_prices = data[benchmark]

# Daily returns
returns = prices.pct_change().dropna()
spy_returns = spy_prices.pct_change().dropna()

# Portfolio daily return
portfolio_returns = returns @ weights

# Equity curves
portfolio_equity = (1 + portfolio_returns).cumprod()
spy_equity = (1 + spy_returns).cumprod()

# Performance metrics
def annualized_return(r):
    return (1 + r).prod() ** (252 / len(r)) - 1

def annualized_volatility(r):
    return r.std() * np.sqrt(252)

def sharpe_ratio(r, rf):
    return (annualized_return(r) - rf) / annualized_volatility(r)

def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()

rolling_window = 252

rolling_mean = portfolio_returns.rolling(rolling_window).mean() * 252
rolling_vol = portfolio_returns.rolling(rolling_window).std() * np.sqrt(252)

rolling_sharpe = (rolling_mean - rf) / rolling_vol


print("=== Portfolio Performance ===")
print("Annual Return:", annualized_return(portfolio_returns))
print("Annual Volatility:", annualized_volatility(portfolio_returns))
print("Sharpe:", sharpe_ratio(portfolio_returns, rf))
print("Max Drawdown:", max_drawdown(portfolio_equity))

print("\n=== SPY Performance ===")
print("Annual Return:", annualized_return(spy_returns))
print("Annual Volatility:", annualized_volatility(spy_returns))
print("Sharpe:", sharpe_ratio(spy_returns, rf))
print("Max Drawdown:", max_drawdown(spy_equity))


plt.figure()
plt.plot(rolling_sharpe, label="Rolling Sharpe (1Y)")
plt.axhline(0, linestyle="--")
plt.title("Rolling Sharpe Ratio (Tangency Portfolio)")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.grid(True)
plt.legend()
plt.show()

# Plot equity curves
plt.plot(portfolio_equity, label="Tangency Portfolio")
plt.plot(spy_equity, label="SPY Benchmark")

plt.title("Backtest: Tangency Portfolio vs SPY")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.legend()
plt.show()


results = pd.DataFrame({
    "Portfolio_Returns": portfolio_returns,
    "SPY_Returns": spy_returns
})

results.to_csv("outputs/backtest_results.csv")
print("\nSaved backtest results to outputs/backtest_results.csv")
