import json
import os
import re
from typing import Dict, Optional

from pydantic import BaseModel
from groq import Groq


class ToolDecision(BaseModel):
    tool: str
    parameters: Optional[Dict] = {}


TOOLS = [
    {
        "name": "get_current_stock_price",
        "description": "Get current stock price of a company",
        "parameters": {"ticker": "Stock ticker like AAPL, TSLA, INFY.NS"}
    },
    {
        "name": "analyze_market_trend",
        "description": "Analyze stock trend using indicators",
        "parameters": {"ticker": "Stock ticker"}
    },
    {
        "name": "predict_stock_profit",
        "description": "Predict future stock price",
        "parameters": {
            "ticker": "Stock ticker",
            "purchase_date": "YYYY-MM-DD",
            "future_date": "YYYY-MM-DD"
        }
    },
    {
        "name": "analyze_portfolio_risk",
        "description": "Analyze risk of a portfolio",
        "parameters": {"holdings": "dictionary of ticker:quantity"}
    },
    {
        "name": "analyze_portfolio_risk_tool",
        "description": "Analyze volatility and diversification risk of a stock portfolio",
        "parameters": {"holdings": "dictionary of ticker and quantity"}
    },
]


class FinancialAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # -----------------------------
    # Tool Decision
    # -----------------------------
    def decide_tool(self, query: str):

        prompt = f"""
You are a financial AI assistant.

User question:
{query}

Available tools:
{json.dumps(TOOLS, indent=2)}

Rules:
- If financial data is required select a tool
- Extract parameters from the query
- If no tool is needed return tool="none"

Return STRICT JSON only.

Example:

{{
 "tool":"get_current_stock_price",
 "parameters":{{"ticker":"AAPL"}}
}}

or

{{
 "tool":"none",
 "parameters":{{}}
}}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.choices[0].message.content

        decision = self._safe_parse(text)

        if decision.tool != "none":
            decision.parameters = self._ensure_parameters(query, decision.parameters)

        return decision

    # -----------------------------
    # Safe JSON parsing
    # -----------------------------
    def _safe_parse(self, text):

        match = re.search(r"\{.*\}", text, re.DOTALL)

        if not match:
            return ToolDecision(tool="none", parameters={})

        try:
            return ToolDecision.model_validate_json(match.group())
        except Exception:
            return ToolDecision(tool="none", parameters={})

    # -----------------------------
    # Parameter completion
    # -----------------------------
    def _ensure_parameters(self, query, params):

        if "ticker" not in params:
            ticker = self.resolve_ticker(query)

            if ticker:
                params["ticker"] = ticker

        return params

    # -----------------------------
    # Ticker extraction
    # -----------------------------
    def resolve_ticker(self, query):

        prompt = f"""
Extract the stock ticker symbol from this query.

Query:
{query}

Examples:
Apple → AAPL
Tesla → TSLA
Microsoft → MSFT
Amazon → AMZN
Google → GOOGL
Infosys → INFY.NS
Reliance → RELIANCE.NS

Return ONLY ticker symbol.
If none return NONE.
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        ticker = response.choices[0].message.content.strip()

        if ticker == "NONE":
            return None

        return ticker

    # -----------------------------
    # Normal AI response
    # -----------------------------
    def general_answer(self, query):

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful financial assistant."},
                {"role": "user", "content": query}
            ]
        )

        return response.choices[0].message.content