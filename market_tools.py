import yfinance as yf
import pandas as pd
import numpy as np
import asyncio


def calculate_rsi(prices, period=14):

    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


async def analyze_market_trend(ticker: str):

    try:
        ticker = ticker.strip().upper()

        if not any(ticker.endswith(sfx) for sfx in [
            ".NS", ".BO", ".NY", ".AX", ".TO", ".L", ".F"
        ]):
            ticker += ".NS"

        data = yf.download(ticker, period="1y", progress=False)

        if data.empty:
            return {"error": f"No data found for {ticker}"}

        # ensure prices is a Series
        prices = data["Close"].squeeze()

        ma50 = prices.rolling(window=50).mean().iloc[-1]
        ma200 = prices.rolling(window=200).mean().iloc[-1]

        rsi = calculate_rsi(prices).iloc[-1]

        latest_price = prices.iloc[-1]

        momentum = (latest_price - prices.iloc[-30]) / prices.iloc[-30]

        # Trend detection
        if float(ma50) > float(ma200):
            trend = "Bullish"
        elif float(ma50) < float(ma200):
            trend = "Bearish"
        else:
            trend = "Neutral"

        # Momentum classification
        if momentum > 0.1:
            momentum_strength = "Strong"
        elif momentum > 0:
            momentum_strength = "Moderate"
        else:
            momentum_strength = "Weak"

        # RSI signal
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        return {
            "ticker": ticker,
            "latest_price": float(latest_price),
            "trend": trend,
            "momentum_strength": momentum_strength,
            "momentum_value": float(momentum),
            "rsi": float(rsi),
            "rsi_signal": rsi_signal,
            "moving_average_50": float(ma50),
            "moving_average_200": float(ma200)
        }

    except Exception as e:

        return {
            "error": str(e),
            "ticker": ticker
        }


