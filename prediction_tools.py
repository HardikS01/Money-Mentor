import yfinance as yf
import pandas as pd
from pandas.tseries.offsets import BDay
from functools import lru_cache


# ─────────────────────────────────────────────────────────────────
# Cache historical data
# ─────────────────────────────────────────────────────────────────
@lru_cache(maxsize=32)
def _get_history(ticker_clean: str) -> pd.Series:
    ticker_obj = yf.Ticker(ticker_clean)
    hist = ticker_obj.history(period="3y")

    if hist.empty:
        return pd.Series(dtype=float)

    series = hist["Close"].dropna()
    series.index = series.index.tz_localize(None)
    return series


def _fit_model_fast(series: pd.Series, days: int):

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA

    trimmed = series.iloc[-500:]

    if days <= 30:

        model = ExponentialSmoothing(
            trimmed,
            trend="add",
            damped_trend=True,
            initialization_method="estimated",
        ).fit(optimized=True)

        return model.forecast(days)

    elif days <= 120:

        model = ARIMA(trimmed, order=(5, 1, 0)).fit()
        return model.forecast(steps=days)

    else:

        model = ARIMA(trimmed, order=(2, 1, 2)).fit()
        return model.forecast(steps=days)


async def predict_stock_profit(ticker: str, purchase_date: str, future_date: str):

    try:

        ticker_clean = ticker.strip().upper()

        if not any(ticker_clean.endswith(sfx) for sfx in [
            ".NS", ".BO", ".NY", ".AX", ".TO", ".L", ".F"
        ]):
            ticker_clean += ".NS"

        series = _get_history(ticker_clean)

        if series.empty:
            return {"error": f"No historical data found for {ticker_clean}"}

        series = series.asfreq("B").ffill()

        today = series.index[-1]

        purchase_dt = pd.to_datetime(purchase_date).tz_localize(None)
        future_dt = pd.to_datetime(future_date).tz_localize(None)

        if future_dt <= today:
            return {"error": "Future date must be later than the last market date."}

        series_dates = set(series.index.strftime("%Y-%m-%d"))

        while purchase_dt.strftime("%Y-%m-%d") not in series_dates:
            purchase_dt += BDay(1)

        purchase_price = float(series.loc[purchase_dt.strftime("%Y-%m-%d")])
        latest_price = float(series.iloc[-1])

        forecast_index = pd.bdate_range(today + BDay(1), future_dt)
        days = len(forecast_index)

        if days <= 0:
            return {"error": "Invalid forecast window — future date must be after today"}

        forecast = _fit_model_fast(series, days)

        forecast_price = float(forecast.iloc[-1])

        pnl = forecast_price - purchase_price
        pct = (pnl / purchase_price) * 100

        return {
            "ticker": ticker_clean,
            "purchase_date": str(purchase_dt.date()),
            "purchase_price": round(purchase_price, 2),
            "latest_price": round(latest_price, 2),
            "predicted_future_price": round(forecast_price, 2),
            "profit_loss_purchase_to_future": round(pnl, 2),
            "pct_change": round(pct, 2),
            "days_forecasted": days,
        }

    except Exception as e:
        return {"error": str(e)}