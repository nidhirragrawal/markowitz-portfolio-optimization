import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

mu = np.array([0.12, 0.08, 0.15])

cov = np.array([
    [0.10, 0.02, 0.04],
    [0.02, 0.08, 0.01],
    [0.04, 0.01, 0.12]
])

def portfolio_return(w):
    return w @ mu

def portfolio_volatility(w):
    return np.sqrt(w @ cov @ w)

# minimize volatility
def objective(w):
    return portfolio_volatility(w)

bounds = tuple((0, 1) for _ in range(len(mu)))
w0 = np.array([1/3, 1/3, 1/3])

target_returns = np.linspace(mu.min(), mu.max(), 30)

frontier_vols = []
frontier_returns = []

for target in target_returns:
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: portfolio_return(w) - target}
    )

    result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)

    if result.success:
        w_opt = result.x
        frontier_returns.append(portfolio_return(w_opt))
        frontier_vols.append(portfolio_volatility(w_opt))

frontier_returns = np.array(frontier_returns)
frontier_vols = np.array(frontier_vols)

rf = 0.03

def neg_sharpe(w):
    r = portfolio_return(w)
    vol = portfolio_volatility(w)
    return -(r - rf) / vol

constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(len(mu)))
w0 = np.array([1/3, 1/3, 1/3])

tangency_result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints)

w_tan = tangency_result.x
r_tan = portfolio_return(w_tan)
vol_tan = portfolio_volatility(w_tan)
sharpe_tan = (r_tan - rf) / vol_tan

print("Tangency Portfolio Weights:", w_tan)
print("Tangency Return:", r_tan)
print("Tangency Volatility:", vol_tan)
print("Tangency Sharpe:", sharpe_tan)

L = 1.5  # leverage factor

w_lev = L * w_tan
w_rf = 1 - L   # amount invested in risk-free asset (negative means borrowing)

r_lev = w_rf * rf + L * r_tan
vol_lev = L * vol_tan

print("\nLeveraged Portfolio (L=1.5):")
print("Risky Weights:", w_lev)
print("Risk-Free Weight:", w_rf)
print("Return:", r_lev)
print("Volatility:", vol_lev)


cml_x = np.linspace(0, max(frontier_vols), 100)
cml_y = rf + sharpe_tan * cml_x

plt.scatter(vol_lev, r_lev, marker="D", s=150, label="Leveraged Portfolio (L=1.5)")


plt.plot(frontier_vols, frontier_returns, marker="o", label="Efficient Frontier")

# Tangency portfolio point
plt.scatter(vol_tan, r_tan, marker="*", s=300, label="Tangency Portfolio (Max Sharpe)")

# Capital Market Line
plt.plot(cml_x, cml_y, linestyle="--", label="Capital Market Line (CML)")

plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier + Capital Market Line")
plt.grid(True)
plt.legend()
plt.show()
