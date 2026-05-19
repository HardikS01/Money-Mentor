from mcp.server.fastmcp import FastMCP

from tools.stock_tools import get_current_stock_price, screen_stocks
from tools.prediction_tools import predict_stock_profit
from tools.savings_tools import forecast_savings
from tools.risk_tools import analyze_portfolio_risk
from tools.market_tools import analyze_market_trend
from tools.portfolio_tools import analyze_portfolio_risk

mcp = FastMCP("finbot-mcp")

mcp.tool()(get_current_stock_price)
mcp.tool()(screen_stocks)
mcp.tool()(predict_stock_profit)
mcp.tool()(forecast_savings)
mcp.tool()(analyze_portfolio_risk)
mcp.tool()(analyze_market_trend)

@mcp.tool()
async def analyze_portfolio_risk_tool(holdings: dict):

    return analyze_portfolio_risk(holdings)

if __name__ == "__main__":
    print("Starting FinBot MCP Server")
    mcp.run(transport="stdio")