import numpy as np

# expected returns of 3 assets
mu = np.array([0.12, 0.08, 0.15])

# covariance matrix
cov = np.array([
    [0.10, 0.02, 0.04],
    [0.02, 0.08, 0.01],
    [0.04, 0.01, 0.12]
])

# portfolio weights
w = np.array([0.4, 0.3, 0.3])

# portfolio expected return
portfolio_return = w @ mu

# portfolio variance
portfolio_variance = w @ cov @ w

# portfolio volatility
portfolio_volatility = np.sqrt(portfolio_variance)

print("Portfolio Return:", portfolio_return)
print("Portfolio Variance:", portfolio_variance)
print("Portfolio Volatility:", portfolio_volatility)
