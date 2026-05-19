import yfinance as yf
import json
from datetime import datetime
from typing import List,Dict,Any


async def get_current_stock_price(ticker:str):

    ticker_map={
        "TCS":"TCS.NS",
        "INFY":"INFY.NS",
        "RELIANCE":"RELIANCE.NS"
    }

    ticker = ticker_map.get(ticker.upper(),ticker)

    stock = yf.Ticker(ticker)
    info = stock.info

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    return json.dumps({
        "ticker":ticker,
        "price":price,
        "company":info.get("longName"),
        "currency":info.get("currency"),
        "timestamp":datetime.now().isoformat()
    },indent=2)



async def screen_stocks(criteria:dict)->List[Dict[str,Any]]:

    tickers = criteria["tickers"]
    max_pe = criteria.get("max_pe")

    results=[]

    for ticker in tickers:

        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get("trailingPE")

        if max_pe and (pe is None or pe > max_pe):
            continue

        results.append({
            "ticker":ticker,
            "company":info.get("longName"),
            "sector":info.get("sector"),
            "price":info.get("currentPrice"),
            "pe_ratio":pe
        })

    return results