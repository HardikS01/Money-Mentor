import os
from groq import Groq


class InvestmentAdvisor:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate_advice(self, query, tool_outputs):

        prompt = f"""
You are a professional financial advisor.

User question:
{query}

Tool outputs:
{tool_outputs}

Based on these outputs provide:

1. Investment recommendation (BUY / HOLD / SELL)
2. Reasoning
3. Risk level
4. Market outlook
5. Portfolio impact

Be concise but professional.
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content