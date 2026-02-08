import numpy as np
from scipy.optimize import minimize

mu = np.array([0.12, 0.08, 0.15])

cov = np.array([
    [0.10, 0.02, 0.04],
    [0.02, 0.08, 0.01],
    [0.04, 0.01, 0.12]
])

rf = 0.03

# portfolio volatility
def portfolio_volatility(w, cov):
    return np.sqrt(w @ cov @ w)

# negative Sharpe ratio (because we minimize)
def neg_sharpe(w, mu, cov, rf):
    port_return = w @ mu
    port_vol = portfolio_volatility(w, cov)
    return -(port_return - rf) / port_vol

# constraints: sum of weights = 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# bounds: no short selling (0 to 1)
bounds = tuple((0, 1) for _ in range(len(mu)))

# initial guess
w0 = np.array([1/3, 1/3, 1/3])

result = minimize(neg_sharpe, w0, args=(mu, cov, rf),
                  method="SLSQP", bounds=bounds, constraints=constraints)

w_opt = result.x

print("Optimal Weights (Max Sharpe):", w_opt)
print("Sum of weights:", np.sum(w_opt))

opt_return = w_opt @ mu
opt_vol = portfolio_volatility(w_opt, cov)
opt_sharpe = (opt_return - rf) / opt_vol

print("Return:", opt_return)
print("Volatility:", opt_vol)
print("Sharpe:", opt_sharpe)
