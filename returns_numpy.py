import numpy as np

prices = np.array([100, 105, 102, 110])

returns = (prices[1:] - prices[:-1]) / prices[:-1]

print("Prices:", prices)
print("Returns:", returns)
print("Mean return:", returns.mean())
print("Volatility (std):", returns.std())
