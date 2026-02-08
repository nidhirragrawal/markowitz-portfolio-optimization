import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

mu = np.array([0.12, 0.08, 0.15])

cov = np.array([
    [0.10, 0.02, 0.04],
    [0.02, 0.08, 0.01],
    [0.04, 0.01, 0.12]
])

num_portfolios = 5000

returns = []
vols = []
sharpes = []
weights_list = []

rf = 0.03  # risk-free rate

for _ in range(num_portfolios):
    w = np.random.random(3)
    w = w / np.sum(w)

    port_return = w @ mu
    port_var = w @ cov @ w
    port_vol = np.sqrt(port_var)

    sharpe = (port_return - rf) / port_vol

    returns.append(port_return)
    vols.append(port_vol)
    sharpes.append(sharpe)
    weights_list.append(w)

returns = np.array(returns)
vols = np.array(vols)
sharpes = np.array(sharpes)
weights_list = np.array(weights_list)

best_idx = np.argmax(sharpes)

min_vol_idx = np.argmin(vols)

print("\nMinimum Variance Portfolio:")
print("Return:", returns[min_vol_idx])
print("Volatility:", vols[min_vol_idx])
print("Sharpe:", sharpes[min_vol_idx])
print("Weights:", weights_list[min_vol_idx])


print("Best Sharpe Portfolio:")
print("Return:", returns[best_idx])
print("Volatility:", vols[best_idx])
print("Sharpe:", sharpes[best_idx])
print("Weights:", weights_list[best_idx])

plt.scatter(vols, returns, c=sharpes, cmap="viridis")
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.title("Efficient Frontier (Random Portfolios)")

# Mark Best Sharpe
plt.scatter(vols[best_idx], returns[best_idx], marker="*", s=300, label="Max Sharpe")

# Mark Minimum Variance
plt.scatter(vols[min_vol_idx], returns[min_vol_idx], marker="X", s=200, label="Min Variance")

plt.legend()
plt.show()
