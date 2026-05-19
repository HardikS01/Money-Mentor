import os
from groq import Groq


class InvestmentPlanner:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def plan(self, query):

        prompt = f"""
You are a financial planning AI.

User question:
{query}

Available tools:
1. get_current_stock_price
2. analyze_market_trend
3. predict_stock_profit
4. analyze_portfolio_risk_tool

Determine which tools should be executed to answer the question.

Return JSON only.

Example:

{{
 "tools": [
  "get_current_stock_price",
  "analyze_market_trend",
  "predict_stock_profit"
 ]
}}
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content