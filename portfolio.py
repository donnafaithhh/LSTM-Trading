# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# for portfolio generation
from scipy.optimize import minimize

class generate_portfolio():
    def __init__(
        self, historical_data, year, trading_days=252, risk_free_rate = 0.02
    ):
        self.historical_data = historical_data
        self.trading_days = trading_days
        self.risk_free_rate = risk_free_rate
        self.year = year
        
    def get_sharpe(self, mu, std):
        if std == 0:
            return -np.inf
        return (mu - self.risk_free_rate) / std
        
    def get_covar_matrix(self):
        returns = self.historical_data.pct_change().dropna(how='all')
        returns = returns.loc[:, returns.std() > 0]
        stocks = returns.columns.to_list()
        mean_returns = returns.mean() * self.trading_days
        covar_matrix = returns.cov() * self.trading_days
        return mean_returns.values, covar_matrix.values, stocks

    def _negative_sharpe(self, weights, mean_returns, cov_matrix):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )
        return -self.get_sharpe(port_return, port_volatility)


    def get_portfolio(self):
        mean_returns, cov_matrix, stocks = self.get_covar_matrix()
        n_assets = len(mean_returns)

        constraints = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        },)
        bounds = tuple((0, 1) for _ in range(n_assets))
        initial_weights = np.ones(n_assets) / n_assets

        # optimize weights
        result = minimize(
            self._negative_sharpe,
            initial_weights,
            args=(mean_returns, cov_matrix),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        weights = result.x

        # final portfolio
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        )
        sharpe = self.get_sharpe(port_return, port_volatility)

        results = {
            'year': self.year, 
            'expected return': port_return,
            'volatility': port_volatility,
            'sharpe ratio': sharpe,
        }
        return weights, results, stocks