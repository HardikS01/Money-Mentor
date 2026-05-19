import yfinance as yf
import pandas as pd
import numpy as np


def analyze_portfolio_risk(holdings: dict):

    price_data = []

    tickers = list(holdings.keys())

    for ticker in tickers:

        data = yf.download(ticker, period="1y")

        if data.empty:
            continue

        price_data.append(data["Close"])

    if len(price_data) == 0:

        return {
            "error": "Could not fetch price data for portfolio."
        }

    df = pd.concat(price_data, axis=1)

    df.columns = tickers[:len(df.columns)]

    returns = df.pct_change().dropna()

    weights = np.array(list(holdings.values()))
    weights = weights / weights.sum()

    cov_matrix = returns.cov() * 252

    portfolio_variance = np.dot(
        weights.T,
        np.dot(cov_matrix, weights)
    )

    volatility = np.sqrt(portfolio_variance)

    sharpe_ratio = returns.mean().mean() / volatility

    return {
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "risk_level": "High" if volatility > 0.35 else "Moderate" if volatility > 0.2 else "Low"
    }