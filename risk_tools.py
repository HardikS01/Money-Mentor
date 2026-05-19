import yfinance as yf
import numpy as np
import pandas as pd


async def analyze_portfolio_risk(holdings: dict):

    tickers = list(holdings.keys())
    weights = np.array(list(holdings.values()))

    weights = weights / weights.sum()

    data = yf.download(tickers, period="1y")["Close"]

    returns = data.pct_change().dropna()

    cov_matrix = returns.cov()

    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    # diversification score
    diversification_score = 1 - np.sum(weights**2)

    # sector allocation
    sector_allocation = {}

    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector", "Unknown")
            sector_allocation[sector] = sector_allocation.get(sector, 0) + holdings[ticker]
        except:
            pass

    total = sum(sector_allocation.values())

    for k in sector_allocation:
        sector_allocation[k] = sector_allocation[k] / total

    # risk level
    if portfolio_volatility < 0.15:
        risk_level = "Low"
    elif portfolio_volatility < 0.30:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return {
        "portfolio_volatility": float(portfolio_volatility),
        "diversification_score": float(diversification_score),
        "sector_concentration": sector_allocation,
        "risk_level": risk_level
    }