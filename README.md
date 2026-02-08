# Markowitz Portfolio Optimization (Efficient Frontier + Tangency Portfolio)

This project implements Modern Portfolio Theory (MPT) using Python.  
It constructs the Efficient Frontier, identifies the Tangency Portfolio (maximum Sharpe ratio portfolio), and backtests the optimized portfolio against the SPY benchmark.

## Features
- Efficient Frontier generation using constrained optimization (SciPy)
- Tangency Portfolio (Max Sharpe) calculation
- Capital Market Line (CML)
- Portfolio backtesting vs SPY benchmark
- Performance metrics:
  - Annualized Return
  - Annualized Volatility
  - Sharpe Ratio
  - Maximum Drawdown
- Rolling Sharpe Ratio analysis
- CSV export of returns

## Tech Stack
- Python
- NumPy, Pandas
- SciPy Optimization
- Matplotlib
- yFinance (market data)

## Methodology

### Expected Portfolio Return
\[
R_p = w^T \mu
\]

### Portfolio Variance
\[
\sigma_p^2 = w^T \Sigma w
\]

### Sharpe Ratio
\[
S = \frac{R_p - R_f}{\sigma_p}
\]

Where:
- \( w \) = portfolio weights
- \( \mu \) = expected returns vector
- \( \Sigma \) = covariance matrix
- \( R_f \) = risk-free rate

## Project Structure
