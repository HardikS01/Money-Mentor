import streamlit as st
import asyncio
import json
import math
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from datetime import date, datetime
from typing import Any, Dict, List, Union
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.session import ClientSession
import platform
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
load_dotenv()
from ai.financial_agent import FinancialAgent
from ai.investment_planner import InvestmentPlanner
from ai.investment_advisor import InvestmentAdvisor

planner = InvestmentPlanner()
advisor = InvestmentAdvisor()

agent = FinancialAgent()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)
if GEMINI_ENABLED:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = "gemini-2.0-flash"
    except Exception:
        GEMINI_ENABLED = False

if platform.system().lower().startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ─────────────────────────────────────────────
# Page Config & Global CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FinBot",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* ── Streamlit overrides ── */
.stApp { background: #0a0e1a; }
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0f1422;
    border-right: 1px solid rgba(99,179,237,0.12);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #63b3ed;
}

/* ── Header ── */
.finbot-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0 0.5rem;
    border-bottom: 1px solid rgba(99,179,237,0.15);
    margin-bottom: 1.5rem;
}
.finbot-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed 0%, #4fd1c5 50%, #68d391 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
}
.finbot-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #4a5568;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Section headings ── */
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
}
h1 { font-size: 1.8rem !important; color: #f7fafc !important; }
h2 { font-size: 1.3rem !important; color: #e2e8f0 !important; }
h3 { font-size: 1.05rem !important; color: #cbd5e0 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1a2234 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 14px;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(99,179,237,0.4);
    box-shadow: 0 0 20px rgba(99,179,237,0.08);
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #718096 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.4rem !important;
    color: #63b3ed !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em;
    border-radius: 10px !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    background: linear-gradient(135deg, #1a2234, #162032) !important;
    color: #63b3ed !important;
    transition: all 0.2s !important;
    padding: 0.45rem 1.2rem !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e3a5f, #1a2e4a) !important;
    border-color: #63b3ed !important;
    box-shadow: 0 0 18px rgba(99,179,237,0.22) !important;
    color: #bee3f8 !important;
}
.stButton > button[disabled] {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}

/* ── Inputs / Selects ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #63b3ed !important;
    box-shadow: 0 0 0 2px rgba(99,179,237,0.15) !important;
}

/* ── DataFrames ── */
.dataframe-container, [data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(99,179,237,0.15) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(99,179,237,0.1) !important;
    margin: 1.5rem 0 !important;
}

/* ── Status messages ── */
.stSuccess {
    background: rgba(72,187,120,0.1) !important;
    border-left: 3px solid #48bb78 !important;
    border-radius: 0 10px 10px 0 !important;
}
.stWarning {
    background: rgba(237,137,54,0.1) !important;
    border-left: 3px solid #ed8936 !important;
    border-radius: 0 10px 10px 0 !important;
}
.stError {
    background: rgba(245,101,101,0.1) !important;
    border-left: 3px solid #f56565 !important;
    border-radius: 0 10px 10px 0 !important;
}
.stInfo {
    background: rgba(99,179,237,0.07) !important;
    border-left: 3px solid #63b3ed !important;
    border-radius: 0 10px 10px 0 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #63b3ed !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1422;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,179,237,0.12);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: #718096 !important;
    border-radius: 8px !important;
    padding: 0.4rem 1rem !important;
    letter-spacing: 0.04em;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3a5c, #1a2e4a) !important;
    color: #63b3ed !important;
}

/* ── Sidebar tool cards ── */
.tool-card {
    background: linear-gradient(135deg, #111827, #141e30);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
}
.tool-card:hover {
    border-color: rgba(99,179,237,0.35);
    background: linear-gradient(135deg, #162032, #1a2e4a);
}
.tool-card-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    color: #63b3ed;
}
.tool-card-desc {
    font-size: 0.72rem;
    color: #718096;
    margin-top: 2px;
}

/* ── Chat area ── */
.chat-bubble {
    background: linear-gradient(135deg, #111827, #141e30);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.chat-bubble.user { border-color: rgba(99,179,237,0.25); }
.chat-bubble.ai   { border-color: rgba(79,209,197,0.2); }

/* ── Section label ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,179,237,0.08);
}

/* ── Connection badge ── */
.conn-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
}
.conn-badge.connected {
    background: rgba(72,187,120,0.12);
    border: 1px solid rgba(72,187,120,0.3);
    color: #68d391;
}
.conn-badge.disconnected {
    background: rgba(245,101,101,0.1);
    border: 1px solid rgba(245,101,101,0.25);
    color: #fc8181;
}
.pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.pulse.green { background: #68d391; box-shadow: 0 0 6px #68d391; }
.pulse.red   { background: #fc8181; box-shadow: 0 0 6px #fc8181; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }

/* ── Form ── */
[data-testid="stForm"] {
    background: #0f1422;
    border: 1px solid rgba(99,179,237,0.1);
    border-radius: 14px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#cbd5e0", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#63b3ed", "#4fd1c5", "#68d391", "#f6ad55", "#fc8181", "#b794f4", "#fbb6ce"],
    xaxis=dict(gridcolor="rgba(99,179,237,0.07)", zerolinecolor="rgba(99,179,237,0.12)"),
    yaxis=dict(gridcolor="rgba(99,179,237,0.07)", zerolinecolor="rgba(99,179,237,0.12)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,179,237,0.15)"),
)

def apply_plotly_theme(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
@st.cache_resource
def get_event_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

loop = get_event_loop()

for k, v in [
    ("connected", False), ("user_portfolio", None), ("tools", []),
    ("filtered_tickers", None), ("filtered_details", None),
    ("mcp_client", None), ("mcp_session", None),
    ("read_stream", None), ("write_stream", None),
    ("artifact_paths", []), ("last_payload", None),
    ("chat_history", []), ("selected_tool", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

ARTIFACT_DIR = os.path.join(".", "data", "session_exports")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# Helpers (all original functions preserved)
# ─────────────────────────────────────────────
def _save_artifact(name: str, payload: Any):
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ARTIFACT_DIR, f"{ts}_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        st.session_state.artifact_paths.append(path)
        st.session_state.artifact_paths = st.session_state.artifact_paths[-10:]
        return path
    except Exception:
        return None

def human_currency(n: Union[int, float, None], currency: str = "USD") -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "-"
    sign = "-" if float(n) < 0 else ""
    n = abs(float(n))
    unit = ""
    for unit in ["", "K", "M", "B", "T"]:
        if n < 1000.0:
            break
        n /= 1000.0
    symbols = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£"}
    sym = symbols.get(currency.upper(), "")
    return f"{sign}{sym}{n:,.2f}{unit}"

def infer_currency_from_ticker(ticker: str | None) -> str:
    if not ticker:
        return "USD"
    tick = str(ticker).upper()
    if tick.endswith(".NS") or tick.endswith(".BO"):
        return "INR"
    return "USD"

def tidy_float(x: Any) -> Any:
    try:
        f = float(x)
        if abs(f) >= 100:
            return f"{f:,.0f}"
        return f"{f:,.2f}"
    except Exception:
        return x


# ─────────────────────────────────────────────
# Smart Ticker Resolver
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def resolve_ticker_smart(raw: str):
    # Try ticker as-is, then with common exchange suffixes.
    # Returns (resolved_ticker, current_price) or (raw, None) if nothing works.
    raw = raw.strip().upper()
    if "." in raw:
        base = raw.split(".")[0]
        candidates = [raw, base, base+".NS", base+".BO", base+".L", base+".AX"]
    else:
        candidates = [
            raw,           # US stocks: AAPL, MSFT etc.
            raw + ".NS",   # NSE India
            raw + ".BO",   # BSE India
            raw + ".L",    # London
            raw + ".AX",   # ASX Australia
            raw + ".TO",   # Toronto
            raw + ".F",    # Frankfurt
        ]
    for candidate in candidates:
        try:
            t = yf.Ticker(candidate)
            info = t.info or {}
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            if price is None:
                hist = t.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
            if price is not None and float(price) > 0:
                return candidate, float(price)
        except Exception:
            continue
    return raw, None


def resolve_holdings_tickers(raw_holdings: dict) -> dict:
    # Given {TICKER: qty}, resolve each to best known symbol.
    resolved = {}
    bad = []
    for raw_ticker, qty in raw_holdings.items():
        good_ticker, price = resolve_ticker_smart(raw_ticker)
        if price is None:
            bad.append(raw_ticker)
        resolved[good_ticker] = qty
    if bad:
        st.warning(f"Could not resolve tickers (no price data): {', '.join(bad)}")
    return resolved


def render_result_payload(payload: Any, context: Dict[str, Any] | None = None):
    st.session_state.last_payload = payload

    # ── Error response ──
    if isinstance(payload, dict) and list(payload.keys()) == ["error"]:
        err_msg = payload["error"]
        st.markdown(f'''
        <div style="background:rgba(245,101,101,0.08);border:1px solid rgba(245,101,101,0.3);
                    border-radius:12px;padding:1rem 1.2rem;margin-top:0.5rem">
            <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#fc8181;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">
                ⚠️ Tool Error
            </div>
            <div style="font-size:0.9rem;color:#fed7d7;line-height:1.5">{err_msg}</div>
        </div>
        ''', unsafe_allow_html=True)
        return

    # ── String response ──
    if isinstance(payload, str):
        # Check if it looks like an error message
        if payload.lower().startswith("error"):
            st.markdown(f'''
            <div style="background:rgba(245,101,101,0.08);border:1px solid rgba(245,101,101,0.3);
                        border-radius:12px;padding:1rem 1.2rem;margin-top:0.5rem">
                <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#fc8181;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">
                    ⚠️ Tool Error
                </div>
                <div style="font-size:0.9rem;color:#fed7d7;line-height:1.5;font-family:DM Mono,monospace">{payload}</div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.code(payload)
        return

    if isinstance(payload, dict) and "Derived_Values" in payload and "Predicted_Savings" in payload:
        _save_artifact("forecast_savings", payload)
        derived = payload["Derived_Values"]
        savings = payload["Predicted_Savings"]

        st.markdown('<div class="section-label">💰 Budget Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Income", human_currency(derived.get("Income"), "INR"))
        c2.metric("Desired Savings %", f"{derived.get('Desired_Savings_Percentage', 0)}%")
        c3.metric("Desired Savings", human_currency(derived.get("Desired_Savings"), "INR"))
        c4.metric("Disposable Income", human_currency(derived.get("Disposable_Income"), "INR"))

        rows = []
        for k, v in savings.items():
            label = k.split("_")[-1].replace("_", " ")
            rows.append({"Category": label, "Potential_Saving": float(v or 0)})
        df = pd.DataFrame(rows).sort_values("Potential_Saving", ascending=False)
        df["Potential_Saving_Display"] = df["Potential_Saving"].apply(lambda x: human_currency(x, "INR"))

        st.markdown('<div class="section-label" style="margin-top:1.5rem">📊 Potential Monthly Savings by Category</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Bar Chart", "Pie Chart", "Table"])

        with tab1:
            fig = go.Figure(go.Bar(
                x=df["Category"], y=df["Potential_Saving"],
                marker=dict(
                    color=df["Potential_Saving"],
                    colorscale=[[0, "#1a3a5c"], [0.5, "#63b3ed"], [1, "#4fd1c5"]],
                    line=dict(color="rgba(99,179,237,0.3)", width=1),
                ),
                text=[human_currency(v, "INR") for v in df["Potential_Saving"]],
                textposition="outside",
                textfont=dict(color="#cbd5e0", size=10),
            ))
            fig.update_layout(title="Potential Savings per Category", xaxis_title="", yaxis_title="Amount (₹)", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if not df.empty:
                fig2 = go.Figure(go.Pie(
                    labels=df["Category"],
                    values=df["Potential_Saving"],
                    hole=0.45,
                    textinfo="label+percent",
                    textfont=dict(color="#cbd5e0"),
                    marker=dict(
                        colors=["#63b3ed","#4fd1c5","#68d391","#f6ad55","#fc8181","#b794f4","#fbb6ce","#90cdf4"],
                        line=dict(color="#0a0e1a", width=2),
                    ),
                ))
                fig2.update_layout(title="Savings Distribution", **PLOTLY_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.dataframe(df[["Category", "Potential_Saving_Display"]], use_container_width=True, hide_index=True)

        quart = payload.get("Quartiles") or {}
        if quart:
            qt_rows = []
            for cat, qvals in quart.items():
                qt_rows.append({
                    "Category": cat,
                    "Q1": qvals.get("Q1"),
                    "Q2 (Median)": qvals.get("Q2"),
                    "Q3": qvals.get("Q3"),
                })
            st.markdown('<div class="section-label" style="margin-top:1.5rem">📐 Benchmark Quartiles</div>', unsafe_allow_html=True)

            qt_df = pd.DataFrame(qt_rows)
            fig3 = go.Figure()
            for col, color in [("Q1","#68d391"), ("Q2 (Median)","#63b3ed"), ("Q3","#fc8181")]:
                fig3.add_trace(go.Bar(name=col, x=qt_df["Category"], y=qt_df[col], marker_color=color))
            fig3.update_layout(barmode="group", title="Spending Quartiles vs Benchmark", **PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)
            st.dataframe(pd.DataFrame(qt_rows), use_container_width=True, hide_index=True)

        fb = payload.get("Quartile_Feedback") or {}
        st.markdown('<div class="section-label" style="margin-top:1.5rem">🧭 Expense Feedback</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        slightly = fb.get("Slightly_High_Q2_Q3", [])
        high = fb.get("Highly_Overspending_Above_Q3", [])
        with c1:
            st.markdown("**⚠️ Slightly High (Q2–Q3)**")
            if slightly:
                for item in slightly:
                    st.markdown(f'<span style="background:rgba(237,137,54,0.15);border:1px solid rgba(237,137,54,0.3);border-radius:6px;padding:3px 10px;font-size:0.82rem;color:#f6ad55;margin-right:6px">{item}</span>', unsafe_allow_html=True)
            else:
                st.success("All good 👍")
        with c2:
            st.markdown("**🚨 Highly Overspending (Above Q3)**")
            if high:
                for item in high:
                    st.markdown(f'<span style="background:rgba(245,101,101,0.15);border:1px solid rgba(245,101,101,0.3);border-radius:6px;padding:3px 10px;font-size:0.82rem;color:#fc8181;margin-right:6px">{item}</span>', unsafe_allow_html=True)
            else:
                st.success("All good 👍")
        return

    # ── Portfolio Summary ──
    if isinstance(payload, dict) and "analytics" in payload and "holdings" in payload:
        _save_artifact("portfolio_summary", payload)
        base = payload.get("base_currency", "INR")

        st.markdown('<div class="section-label">🏦 Portfolio Overview</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        analytics = payload.get("analytics", {})
        k1.metric("Total Value", human_currency(payload.get("portfolio_value_base"), base))
        k2.metric("# Holdings", analytics.get("num_holdings", 0))
        top_p = analytics.get("top_performers") or []
        worst_p = analytics.get("worst_performers") or []
        if top_p:
            k3.metric("Best Performer", top_p[0].get("ticker", "—"),
                      f"{top_p[0].get('return_pct_since_purchase', 0):.2f}%")
        if worst_p:
            k4.metric("Worst Performer", worst_p[0].get("ticker", "—"),
                      f"{worst_p[0].get('return_pct_since_purchase', 0):.2f}%")

        alloc = analytics.get("sector_allocation", {})
        tab1, tab2, tab3, tab4 = st.tabs(["Sector Allocation", "Performance", "Holdings Table", "Raw Data"])

        with tab1:
            if alloc:
                sectors = list(alloc.keys())
                weights = [v * 100 for v in alloc.values()]
                cola, colb = st.columns([1, 1])
                with cola:
                    fig = go.Figure(go.Pie(
                        labels=sectors, values=weights, hole=0.5,
                        textinfo="label+percent",
                        textfont=dict(color="#cbd5e0", size=11),
                        marker=dict(
                            colors=["#63b3ed","#4fd1c5","#68d391","#f6ad55","#fc8181","#b794f4","#fbb6ce"],
                            line=dict(color="#0a0e1a", width=2),
                        ),
                    ))
                    fig.update_layout(title="Sector Weights", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                with colb:
                    fig2 = go.Figure(go.Bar(
                        x=sectors, y=weights,
                        marker=dict(
                            color=weights,
                            colorscale=[[0,"#1a3a5c"],[1,"#4fd1c5"]],
                            line=dict(color="rgba(99,179,237,0.3)", width=1),
                        ),
                        text=[f"{w:.1f}%" for w in weights],
                        textposition="outside",
                        textfont=dict(color="#cbd5e0", size=10),
                    ))
                    fig2.update_layout(title="Sector Bar Chart", yaxis_title="Weight %", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No sector allocation data available.")

        with tab2:
            if top_p or worst_p:
                all_performers = (top_p or []) + (worst_p or [])
                perf_df = pd.DataFrame(all_performers)
                if not perf_df.empty and "return_pct_since_purchase" in perf_df.columns:
                    perf_df = perf_df.sort_values("return_pct_since_purchase", ascending=False)
                    colors = ["#68d391" if v >= 0 else "#fc8181" for v in perf_df["return_pct_since_purchase"]]
                    fig = go.Figure(go.Bar(
                        x=perf_df["ticker"],
                        y=perf_df["return_pct_since_purchase"],
                        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
                        text=[f"{v:.2f}%" for v in perf_df["return_pct_since_purchase"]],
                        textposition="outside",
                        textfont=dict(color="#cbd5e0"),
                    ))
                    fig.update_layout(title="Return % Since Purchase", yaxis_title="Return %", **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

                ctop, cworst = st.columns(2)
                if top_p:
                    ctop.markdown("**🟢 Top Performers**")
                    top_df = pd.DataFrame(top_p)
                    if "return_pct_since_purchase" in top_df.columns:
                        top_df.rename(columns={"return_pct_since_purchase": "Return %"}, inplace=True)
                    ctop.dataframe(top_df, use_container_width=True, hide_index=True)
                if worst_p:
                    cworst.markdown("**🔴 Worst Performers**")
                    worst_df = pd.DataFrame(worst_p)
                    if "return_pct_since_purchase" in worst_df.columns:
                        worst_df.rename(columns={"return_pct_since_purchase": "Return %"}, inplace=True)
                    cworst.dataframe(worst_df, use_container_width=True, hide_index=True)
            else:
                st.info("No performance data available.")

        with tab3:
            table = pd.DataFrame(payload["holdings"])
            cols_order = ["ticker","quantity","price","currency","value_native","value_base",
                          "weight","purchase_price","return_pct_since_purchase","sector","pe_ratio","dividend_yield"]
            table = table[[c for c in cols_order if c in table.columns]]
            table.rename(columns={
                "value_native": f"Value (native)",
                "value_base":   f"Value ({base})",
                "weight": "Weight",
                "price": "Live Price",
                "purchase_price": "Purchase Price",
                "return_pct_since_purchase": "Return %"
            }, inplace=True)
            if "Weight" in table.columns:
                table["Weight"] = table["Weight"].apply(lambda x: f"{float(x)*100:.2f}%")
            st.dataframe(table, use_container_width=True, hide_index=True)

        with tab4:
            st.json(payload)
        return

    # ── Price Tool ──
    if isinstance(payload, dict) and "price" in payload:
        cur = infer_currency_from_ticker(payload.get("ticker"))
        st.markdown('<div class="section-label">💹 Live Price</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", payload.get("ticker", "—"))
        c2.metric("Company", payload.get("company_name", "—"))
        c3.metric("Price", human_currency(payload.get("price"), payload.get("currency", cur)))
        c4.metric("Market", payload.get("market", "—"))
        return

    # ── Predict Stock Profit ──
    if isinstance(payload, dict) and {"purchase_price", "latest_price", "predicted_future_price"} <= payload.keys():
        tick = payload.get("ticker", "—")
        cur = infer_currency_from_ticker(tick)
        purchase = payload.get("purchase_price")
        latest = payload.get("latest_price")
        future = payload.get("predicted_future_price")
        pnl = payload.get("profit_loss_purchase_to_future")

        st.markdown('<div class="section-label">📈 Prediction Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", tick)
        c2.metric("Buy Price", human_currency(purchase, cur))
        c3.metric("Latest Price", human_currency(latest, cur))
        c4.metric("Forecast Price", human_currency(future, cur))

        try:
            pct = ((future - purchase) / purchase) * 100.0
        except Exception:
            pct = None

        colA, colB = st.columns([1, 2])
        with colA:
            delta_str = f"{pct:.2f}%" if pct is not None else None
            colA.metric("Expected P/L", human_currency(pnl, cur), delta_str)
        with colB:
            labels = ["Buy", "Latest", "Forecast"]
            values = [purchase, latest, future]
            colors_bar = ["#63b3ed", "#4fd1c5", "#68d391" if (future or 0) >= (purchase or 0) else "#fc8181"]
            fig = go.Figure(go.Bar(
                x=labels, y=values,
                marker=dict(color=colors_bar, line=dict(color="rgba(255,255,255,0.1)", width=1)),
                text=[human_currency(v, cur) for v in values],
                textposition="outside",
                textfont=dict(color="#cbd5e0"),
            ))
            fig.update_layout(title=f"{tick} Price Comparison", yaxis_title=f"Price ({cur})", **PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        return

    # ── Screened Stocks List ──
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        df = pd.DataFrame(payload)
        st.markdown('<div class="section-label">🔍 Screened Results</div>', unsafe_allow_html=True)

        num_cols = [c for c in df.columns if df[c].dtype in ["float64","int64"]]
        tab1, tab2 = st.tabs(["Table", "Chart"])
        with tab1:
            st.dataframe(df, use_container_width=True, hide_index=True)
        with tab2:
            if "pe_ratio" in df.columns and "ticker" in df.columns:
                fig = go.Figure(go.Bar(
                    x=df["ticker"], y=df["pe_ratio"],
                    marker=dict(
                        color=df["pe_ratio"],
                        colorscale=[[0,"#68d391"],[0.5,"#63b3ed"],[1,"#fc8181"]],
                        line=dict(color="rgba(255,255,255,0.1)", width=1),
                    ),
                    text=[f"{v:.1f}" for v in df["pe_ratio"]],
                    textposition="outside",
                    textfont=dict(color="#cbd5e0"),
                ))
                fig.update_layout(title="P/E Ratios", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)
            elif num_cols and "ticker" in df.columns:
                selected_col = num_cols[0]
                fig = px.bar(df, x="ticker", y=selected_col, title=selected_col.replace("_", " ").title())
                apply_plotly_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chartable numeric columns found.")
        return

    # ── Generic dict with known risk/trend keys ──
    if isinstance(payload, dict):
        # Check if it has meaningful nested structure (risk/trend tool outputs)
        has_nested = any(isinstance(v, dict) for v in payload.values())
        if has_nested:
            for k, v in payload.items():
                label = k.replace("_", " ").title()
                if isinstance(v, dict):
                    st.markdown(f'<div class="section-label" style="margin-top:0.8rem">{label}</div>', unsafe_allow_html=True)
                    inner_df = pd.DataFrame([[ik, iv] for ik, iv in v.items()], columns=["Metric", "Value"])
                    st.dataframe(inner_df, use_container_width=True, hide_index=True)
                elif isinstance(v, list):
                    st.markdown(f'<div class="section-label" style="margin-top:0.8rem">{label}</div>', unsafe_allow_html=True)
                    if v and isinstance(v[0], dict):
                        st.dataframe(pd.DataFrame(v), use_container_width=True, hide_index=True)
                    else:
                        st.write(", ".join(str(i) for i in v))
                else:
                    st.metric(label, str(v))
        else:
            df = pd.DataFrame([[k, v] for k, v in payload.items()], columns=["Key", "Value"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        return

    if isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), use_container_width=True, hide_index=True)
        return

    st.code(json.dumps(payload, indent=2))

def render_result(raw: Any):
    if isinstance(raw, list):
        for chunk in raw:
            try:
                data = json.loads(chunk.text)
            except Exception:
                data = getattr(chunk, "text", chunk)
            render_result_payload(data)
    else:
        try:
            data = json.loads(raw)
        except Exception:
            data = raw
        render_result_payload(data)

# ─────────────────────────────────────────────
# Connection helpers
# ─────────────────────────────────────────────
async def _connect_async():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SERVER_PATH = os.path.join(BASE_DIR, "server.py")
    server = StdioServerParameters(command=sys.executable, args=[SERVER_PATH], cwd=BASE_DIR)
    client = stdio_client(server)
    read_stream, write_stream = await client.__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    init = await session.initialize()
    tools = await session.list_tools()
    st.session_state.mcp_client = client
    st.session_state.read_stream = read_stream
    st.session_state.write_stream = write_stream
    st.session_state.mcp_session = session
    st.session_state.tools = [{"name": t.name, "description": t.description} for t in tools.tools]
    st.session_state.connected = True
    st.success(f"Connected to {init.serverInfo.name}")

def connect_to_server():
    try:
        loop.run_until_complete(_connect_async())
    except Exception as e:
        st.error(f"Connection failed: {e}")
        st.exception(e)

async def _disconnect_async():
    if st.session_state.mcp_session is not None:
        try:
            await st.session_state.mcp_session.__aexit__(None, None, None)
        except Exception:
            pass
    if st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
        except Exception:
            pass
    st.session_state.mcp_client = None
    st.session_state.mcp_session = None
    st.session_state.read_stream = None
    st.session_state.write_stream = None

def _require_session() -> ClientSession | None:
    if not st.session_state.connected or st.session_state.mcp_session is None:
        st.warning("Please connect to the server first.")
        return None
    return st.session_state.mcp_session

def execute_tool(tool_name: str, args: dict):
    session = _require_session()
    if session is None:
        return

    async def run(sess: ClientSession):
        try:
            with st.spinner(f"⚡ Running {tool_name}..."):
                result = await sess.call_tool(tool_name, args)
            st.success("✅ Execution complete")
            st.markdown('<div class="section-label" style="margin-top:1rem">📊 Result</div>', unsafe_allow_html=True)
            render_result(result.content)
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.exception(e)

    loop.run_until_complete(run(session))

def call_tool_return_json(tool_name: str, args: dict):
    session = _require_session()
    if session is None:
        return None

    async def run(sess: ClientSession):
        result = await sess.call_tool(tool_name, args)
        rows = []
        for item in result.content:
            try:
                rows.append(json.loads(item.text))
            except Exception:
                rows.append(item.text)
        if len(rows) == 1:
            return rows[0]
        return rows

    return loop.run_until_complete(run(session))

@st.cache_data(ttl=60 * 60 * 24)
def get_yf_universe():
    try:
        return sorted(set(yf.tickers_sp500() + yf.tickers_dow()))
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "BRK-B", "MRF.NS", "TCS.NS", "INFY.NS"]

def _read_file(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()[:100_000]
    except Exception:
        return ""

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                    background:linear-gradient(135deg,#63b3ed,#4fd1c5);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;letter-spacing:-1px">FinBot</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4a5568;
                    letter-spacing:0.15em;text-transform:uppercase;margin-top:3px">
            AI Financial Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Connection status
    if st.session_state.connected:
        st.markdown('<div class="conn-badge connected"><span class="pulse green"></span>Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="conn-badge disconnected"><span class="pulse red"></span>Disconnected</div>', unsafe_allow_html=True)

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        st.button("🔌 Connect", disabled=st.session_state.connected, on_click=connect_to_server, use_container_width=True)
    with col2:
        def _disconnect_click():
            try:
                loop.run_until_complete(_disconnect_async())
            finally:
                st.session_state.connected = False
                st.session_state.tools = []
                st.session_state.filtered_tickers = None
                st.session_state.filtered_details = None
                st.rerun()
        st.button("⏏ Disconnect", disabled=not st.session_state.connected, on_click=_disconnect_click, use_container_width=True)

    st.divider()

    if st.session_state.connected and st.session_state.tools:
        st.markdown('<div class="section-label">🛠 Available Tools</div>', unsafe_allow_html=True)
        _sidebar_tool_icons = {
            "get_current_stock_price": "💹",
            "portfolio_summary": "🏦",
            "analyze_portfolio_risk": "⚠️",
            "analyze_market_trend": "📉",
            "screen_stocks": "🔍",
            "predict_stock_profit": "🔮",
            "forecast_savings": "💰",
        }
        for _t in st.session_state.tools:
            _icon = _sidebar_tool_icons.get(_t["name"], "⚙️")
            _name_pretty = _t["name"].replace("_", " ").title()
            _is_active = st.session_state.selected_tool == _t["name"]
            _active_style = "border-color:rgba(99,179,237,0.5);background:linear-gradient(135deg,#162032,#1a2e4a);" if _is_active else ""
            st.markdown(f"""
            <div class="tool-card" style="{_active_style}">
                <div class="tool-card-name">{_icon} {_name_pretty}</div>
                <div class="tool-card-desc">{(_t.get("description") or "")[:60]}...</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Open {_name_pretty}", key=f"sidebar_btn_{_t['name']}", use_container_width=True):
                st.session_state.selected_tool = _t["name"]
                st.rerun()
        st.markdown("""
        <style>
        [data-testid="stSidebar"] .stButton > button {
            margin-top: -52px !important;
            height: 52px !important;
            opacity: 0 !important;
            position: relative !important;
            z-index: 10 !important;
            width: 100% !important;
            border-radius: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#4a5568;font-size:0.8rem;text-align:center;padding:1rem 0">Connect to load tools</div>', unsafe_allow_html=True)

    st.divider()

    # Recent artifacts
    if st.session_state.artifact_paths:
        st.markdown('<div class="section-label">📁 Recent Exports</div>', unsafe_allow_html=True)
        for p in reversed(st.session_state.artifact_paths[-5:]):
            fname = os.path.basename(p)
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#718096;padding:3px 0">📄 {fname}</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="finbot-header">
    <div>
        <div class="finbot-logo">💹 FinBot</div>
        <div class="finbot-tagline">AI-powered financial intelligence dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main content — two tabs
# ─────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs(["🛠 Tools", "💬 AI Chat"])

# ══════════════════════════════════════════════
# TAB 1: TOOLS
# ══════════════════════════════════════════════
with main_tab1:
    if not (st.session_state.connected and st.session_state.tools):
        st.markdown("""
        <div style="text-align:center;padding:4rem 0">
            <div style="font-size:3rem">🔌</div>
            <div style="font-family:Syne,sans-serif;font-size:1.2rem;color:#718096;margin-top:0.8rem">
                Connect to the server using the sidebar to load tools
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tool_names = [t["name"] for t in st.session_state.tools]
        tool_icons = {
            "get_current_stock_price": "💹",
            "portfolio_summary": "🏦",
            "analyze_portfolio_risk": "⚠️",
            "analyze_market_trend": "📉",
            "screen_stocks": "🔍",
            "predict_stock_profit": "🔮",
            "forecast_savings": "💰",
        }

        label_map = {n: f"{tool_icons.get(n,'⚙️')} {n.replace('_',' ').title()}" for n in tool_names}
        # Sync selectbox index from sidebar click
        _default_idx = 0
        if st.session_state.selected_tool in tool_names:
            _default_idx = tool_names.index(st.session_state.selected_tool)
        selected_tool = st.selectbox(
            "Select Tool",
            tool_names,
            index=_default_idx,
            format_func=lambda n: label_map[n],
        )
        # Keep session state in sync when selectbox changes
        st.session_state.selected_tool = selected_tool

        st.divider()

        # ─── get_current_stock_price ───
        if selected_tool == "get_current_stock_price":
            st.markdown("### 💹 Live Stock Price")
            ticker = st.text_input("Enter Ticker", "AAPL", placeholder="e.g. AAPL, TCS, RELIANCE, MSFT")
            st.caption("Exchange suffix optional — auto-detected if omitted (e.g. TCS → TCS.NS)")
            def _run_price():
                resolved, _ = resolve_ticker_smart(ticker)
                execute_tool(selected_tool, {"ticker": resolved})
            st.button("▶️ Get Price", on_click=_run_price)

        # ─── portfolio_summary ───
        elif selected_tool == "portfolio_summary":
            st.markdown("### 🏦 Portfolio Summary")
            base_ccy = st.selectbox("Base currency", ["INR", "USD", "EUR", "GBP"], index=0)
            st.caption("Optional: Upload CSV with columns: Ticker, Quantity, PurchasePrice")
            up = st.file_uploader("Upload holdings CSV", type=["csv"], accept_multiple_files=False)
            st.markdown("#### ✏️ Add / Modify Holdings")
            universe = get_yf_universe()
            with st.form("portfolio_form"):
                init_df = pd.DataFrame([{"Ticker": "AAPL", "Quantity": 10.0, "PurchasePrice": ""}])
                edited = st.data_editor(
                    init_df,
                    num_rows="dynamic",
                    column_config={
                        "Ticker": st.column_config.SelectboxColumn(options=universe + ["MRF.NS", "TCS.NS", "INFY.NS"]),
                        "Quantity": st.column_config.NumberColumn(min_value=0.0, step=1.0, format="%.2f"),
                        "PurchasePrice": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
                    },
                    use_container_width=True
                )
                st.form_submit_button("Apply")

            if up is not None:
                try:
                    df_csv = pd.read_csv(up)
                    df_csv.columns = [c.strip().lower() for c in df_csv.columns]
                    map_rows = []
                    for _, r in df_csv.iterrows():
                        map_rows.append({
                            "Ticker": str(r.get("ticker") or r.get("symbol") or "").strip(),
                            "Quantity": float(r.get("quantity") or r.get("qty") or 0.0),
                            "PurchasePrice": float(r.get("purchaseprice") or r.get("ppx") or 0.0) if (r.get("purchaseprice") or r.get("ppx")) else ""
                        })
                    edited = pd.DataFrame(map_rows)
                except Exception as e:
                    st.error(f"CSV parse failed: {e}")

            if not edited.empty:
                preview_rows = []
                for _, r in edited.iterrows():
                    tk = str(r["Ticker"]).strip()
                    if not tk:
                        continue
                    qty = float(r.get("Quantity") or 0.0)
                    ppx = r.get("PurchasePrice")
                    try:
                        info = yf.Ticker(tk).info or {}
                        price = info.get("currentPrice")
                        if price is None:
                            hist = yf.Ticker(tk).history(period="1d")
                            price = float(hist["Close"].iloc[-1]) if not hist.empty else None
                    except Exception:
                        price = None
                    val = (price * qty) if (price is not None) else None
                    ret = None
                    if (ppx not in [None, ""]) and (price is not None):
                        try:
                            ret = ((float(price) - float(ppx)) / float(ppx)) * 100.0
                        except Exception:
                            ret = None
                    preview_rows.append({
                        "Ticker": tk,
                        "Quantity": qty,
                        "PurchasePrice": ppx if ppx not in [None, ""] else None,
                        "LivePrice": round(float(price), 2) if price is not None else None,
                        "RowValue": round(float(val), 2) if val is not None else None,
                        "Return%": round(float(ret), 2) if ret is not None else None
                    })
                st.markdown("#### 👁 Live Preview")
                prev_df = pd.DataFrame(preview_rows)
                st.dataframe(prev_df, use_container_width=True, hide_index=True)

                # Mini sparkline
                if prev_df["Return%"].notna().any():
                    ret_data = prev_df.dropna(subset=["Return%"])
                    fig = go.Figure(go.Bar(
                        x=ret_data["Ticker"], y=ret_data["Return%"],
                        marker=dict(color=["#68d391" if v >= 0 else "#fc8181" for v in ret_data["Return%"]]),
                        text=[f"{v:.2f}%" for v in ret_data["Return%"]],
                        textposition="outside",
                        textfont=dict(color="#cbd5e0", size=10),
                    ))
                    fig.update_layout(title="Live Return % Overview", height=280, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)

            if st.button("▶️ Run Portfolio Summary"):
                rows = []
                for _, r in edited.iterrows():
                    tk = str(r["Ticker"]).strip()
                    if not tk:
                        continue
                    qty = float(r.get("Quantity") or 0.0)
                    ppx = r.get("PurchasePrice")
                    rows.append((tk, qty, ppx))
                if not rows:
                    st.warning("Please add at least one holding.")
                else:
                    raw_holdings = {tk: qty for tk, qty, _ in rows}
                    with st.spinner("Resolving tickers..."):
                        holdings = resolve_holdings_tickers(raw_holdings)
                    ppx = {}
                    for tk_raw, qty, pp in rows:
                        # Find the resolved ticker for this raw ticker
                        resolved_tk, _ = resolve_ticker_smart(tk_raw)
                        if pp not in [None, ""]:
                            try:
                                ppx[resolved_tk] = float(pp)
                            except Exception:
                                pass
                    execute_tool(selected_tool, {"holdings": holdings, "purchase_prices": ppx, "base_currency": base_ccy})

        # ─── analyze_portfolio_risk ───
        elif selected_tool == "analyze_portfolio_risk":
            st.markdown("### ⚠️ Portfolio Risk Analyzer")
            holdings_text = st.text_area("Holdings JSON", value='{"AAPL": 10, "MSFT": 5, "GOOGL": 3}', height=120)
            st.caption("Tickers without exchange suffix (e.g. TCS, RELIANCE) are auto-resolved to NSE/BSE/NYSE.")
            if st.button("▶️ Analyze Risk"):
                try:
                    raw_holdings = json.loads(holdings_text)
                    with st.spinner("Resolving tickers..."):
                        holdings = resolve_holdings_tickers(raw_holdings)
                    execute_tool(selected_tool, {"holdings": holdings})
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        # ─── analyze_market_trend ───
        elif selected_tool == "analyze_market_trend":
            st.markdown("### 📉 Market Trend Analyzer")
            ticker = st.text_input("Enter Stock Ticker", value="TCS", placeholder="e.g. TCS, AAPL, RELIANCE")
            st.caption("Exchange suffix optional — auto-resolved if omitted")
            if st.button("▶️ Analyze Trend"):
                with st.spinner("Resolving ticker..."):
                    resolved, price = resolve_ticker_smart(ticker)
                if price is None:
                    st.error(f"Could not find price data for '{ticker}'. Try adding exchange suffix e.g. TCS.NS")
                else:
                    if resolved != ticker:
                        st.info(f"Resolved **{ticker}** → **{resolved}**")
                    execute_tool(selected_tool, {"ticker": resolved})

        # ─── screen_stocks ───
        elif selected_tool == "screen_stocks":
            st.markdown("### 🔍 Stock Screener")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                max_pe = st.number_input("Max P/E", value=40.0, step=1.0, min_value=0.0)
            with col_f2:
                mcap_options = {
                    "100M": 100_000_000, "500M": 500_000_000, "1B": 1_000_000_000,
                    "5B": 5_000_000_000, "10B": 10_000_000_000, "50B": 50_000_000_000,
                    "100B": 100_000_000_000,
                }
                mcap_label = st.selectbox("Min Market Cap", list(mcap_options.keys()), index=2)
                min_mcap = mcap_options[mcap_label]

            st.caption("Screener uses only S&P 500 + DOW 30. Manual tickers are not filtered.")

            if st.button("🔍 Find Eligible Stocks"):
                universe = get_yf_universe()
                with st.spinner(f"Filtering {len(universe)} tickers..."):
                    result = call_tool_return_json(
                        "screen_stocks",
                        {"criteria": {"tickers": universe, "max_pe": max_pe, "min_market_cap": min_mcap}},
                    )
                    eligible = [row.get("ticker") for row in (result or []) if isinstance(row, dict) and not row.get("error")]
                    st.session_state.filtered_details = result
                    st.session_state.filtered_tickers = sorted(set([t for t in eligible if t]))

            if st.session_state.filtered_tickers is not None:
                count = len(st.session_state.filtered_tickers)
                st.success(f"✅ {count} matching stocks found.")
                selected_from_list = st.multiselect("Matching Stocks", st.session_state.filtered_tickers, default=[])
            else:
                selected_from_list = []

            manual = st.text_input("Or add custom tickers (comma-separated)", "")
            if st.button("▶️ Run Screener"):
                final_tickers = list(selected_from_list)
                final_tickers += [t.strip() for t in manual.split(",") if t.strip()]
                final_tickers = sorted(set(final_tickers))
                if not final_tickers:
                    st.warning("Please select at least one ticker.")
                else:
                    if manual.strip():
                        execute_tool("screen_stocks", {"criteria": {"tickers": final_tickers}})
                    else:
                        execute_tool("screen_stocks", {"criteria": {"tickers": final_tickers, "max_pe": max_pe, "min_market_cap": min_mcap}})

        # ─── predict_stock_profit ───
        elif selected_tool == "predict_stock_profit":
            st.markdown("### 🔮 Stock Profit Predictor")
            col_t, col_d1, col_d2 = st.columns(3)
            with col_t:
                ticker = st.text_input("Ticker", "MRF.NS")
            with col_d1:
                purchase_date = st.date_input("Purchase Date", date(2025, 11, 3))
            with col_d2:
                future_date = st.date_input("Future Date", date(2026, 1, 25))
            if st.button("▶️ Predict"):
                pd_str = purchase_date.strftime("%Y-%m-%d") if hasattr(purchase_date, "strftime") else str(purchase_date)
                fd_str = future_date.strftime("%Y-%m-%d") if hasattr(future_date, "strftime") else str(future_date)
                with st.spinner("Resolving ticker..."):
                    resolved_ticker, price = resolve_ticker_smart(ticker)
                if price is None:
                    st.error(f"Could not find price data for '{ticker}'. Try adding exchange suffix e.g. MRF.NS")
                else:
                    if resolved_ticker != ticker:
                        st.info(f"Resolved **{ticker}** → **{resolved_ticker}**")
                    execute_tool(selected_tool, {"ticker": resolved_ticker, "purchase_date": pd_str, "future_date": fd_str})

        # ─── forecast_savings ───
        elif selected_tool == "forecast_savings":
            st.markdown("### 💰 Savings Forecaster")
            col_inc, col_pct = st.columns([2, 1])
            with col_inc:
                income = st.number_input("Monthly Income (₹)", min_value=0.0, value=90000.0, step=1000.0)
            with col_pct:
                desired_pct = st.slider("Desired Savings %", min_value=1, max_value=80, value=20, step=1)

            st.markdown("#### 📋 Expense Breakdown")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                rent = st.number_input("Rent", min_value=0.0, value=15000.0, step=500.0)
                loan = st.number_input("Loan Repayment", min_value=0.0, value=8000.0, step=500.0)
                insurance = st.number_input("Insurance", min_value=0.0, value=3000.0, step=500.0)
                groceries = st.number_input("Groceries", min_value=0.0, value=9000.0, step=500.0)
            with exp_col2:
                transport = st.number_input("Transport", min_value=0.0, value=3000.0, step=500.0)
                eating_out = st.number_input("Eating Out", min_value=0.0, value=2000.0, step=500.0)
                entertainment = st.number_input("Entertainment", min_value=0.0, value=2500.0, step=500.0)
                utilities = st.number_input("Utilities", min_value=0.0, value=4000.0, step=500.0)
            with exp_col3:
                healthcare = st.number_input("Healthcare", min_value=0.0, value=2000.0, step=500.0)
                education = st.number_input("Education", min_value=0.0, value=0.0, step=500.0)
                misc = st.number_input("Miscellaneous", min_value=0.0, value=3000.0, step=500.0)

            total_exp = rent + loan + insurance + groceries + transport + eating_out + entertainment + utilities + healthcare + education + misc
            surplus = income - total_exp
            surplus_pct = (surplus / income * 100) if income > 0 else 0

            st.markdown("#### 📊 Budget Preview")
            prev_c1, prev_c2, prev_c3 = st.columns(3)
            prev_c1.metric("Total Expenses", human_currency(total_exp, "INR"))
            prev_c2.metric("Surplus", human_currency(surplus, "INR"), f"{surplus_pct:.1f}%")
            prev_c3.metric("Target Savings", human_currency(income * desired_pct / 100, "INR"), f"{desired_pct}%")

            exp_breakdown = {
                "Rent": rent, "Loan": loan, "Insurance": insurance, "Groceries": groceries,
                "Transport": transport, "Eating Out": eating_out, "Entertainment": entertainment,
                "Utilities": utilities, "Healthcare": healthcare, "Education": education, "Misc": misc,
            }
            exp_df = pd.DataFrame({"Category": list(exp_breakdown.keys()), "Amount": list(exp_breakdown.values())})
            exp_df = exp_df[exp_df["Amount"] > 0].sort_values("Amount", ascending=False)
            if not exp_df.empty:
                fig = go.Figure(go.Pie(
                    labels=exp_df["Category"], values=exp_df["Amount"], hole=0.4,
                    textinfo="label+percent",
                    textfont=dict(color="#cbd5e0"),
                    marker=dict(
                        colors=["#63b3ed","#4fd1c5","#68d391","#f6ad55","#fc8181","#b794f4","#fbb6ce","#90cdf4","#fbd38d","#9ae6b4","#bee3f8"],
                        line=dict(color="#0a0e1a", width=2),
                    ),
                ))
                fig.update_layout(title="Current Expense Distribution", height=320, **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("▶️ Forecast Savings"):
                budget_dict = {
                    "Income": income, "Rent": rent, "Loan_Repayment": loan, "Insurance": insurance,
                    "Groceries": groceries, "Transport": transport, "Eating_Out": eating_out,
                    "Entertainment": entertainment, "Utilities": utilities, "Healthcare": healthcare,
                    "Education": education, "Miscellaneous": misc,
                }
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
                with open(temp_file.name, "w") as f:
                    json.dump(budget_dict, f, indent=2)
                execute_tool(selected_tool, {"json_path": temp_file.name, "desired_savings_percentage": desired_pct})

# ══════════════════════════════════════════════
# TAB 2: AI CHAT
# ══════════════════════════════════════════════
with main_tab2:
    st.markdown("### 💬 Talk to Your Money")

    recent_context = list(reversed(st.session_state.artifact_paths))[:3]

    # Portfolio input
    with st.expander("📊 Set Your Portfolio", expanded=False):
        portfolio_text = st.text_input(
            "Portfolio (example: AAPL:10, TSLA:5, MSFT:8)",
            placeholder="AAPL:10, TSLA:5, MSFT:8",
        )
        if st.button("💾 Save Portfolio"):
            if portfolio_text:
                raw_holdings = {}
                items = portfolio_text.split(",")
                for item in items:
                    try:
                        ticker_s, qty_s = item.split(":")
                        raw_holdings[ticker_s.strip().upper()] = int(qty_s.strip())
                    except Exception:
                        pass
                with st.spinner("Resolving tickers..."):
                    holdings = resolve_holdings_tickers(raw_holdings)
                st.session_state.user_portfolio = holdings
                resolved_display = ", ".join(f"{t}×{q}" for t, q in holdings.items())
                st.success(f"Portfolio saved: {resolved_display}")

    # Chat history display
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-10:]:
            role_icon = "🧑" if msg["role"] == "user" else "🤖"
            bubble_class = "user" if msg["role"] == "user" else "ai"
            st.markdown(f"""
            <div class="chat-bubble {bubble_class}">
                <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5568;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.1em">
                    {role_icon} {msg["role"].title()}
                </div>
                <div style="font-size:0.9rem;color:#e2e8f0;line-height:1.6">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Input area
    prompt = st.text_area(
        "Ask anything about your savings or portfolio:",
        height=100,
        placeholder="e.g. Where am I overspending? What's my sector risk? Should I buy AAPL?",
        key="chat_prompt",
    )

    if st.button("▶️ Ask FinBot", type="primary"):
        if not prompt.strip():
            st.warning("Please type a question.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            decision = agent.decide_tool(prompt)

            if decision.tool == "none":
                with st.spinner("Thinking…"):
                    answer = agent.general_answer(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.markdown(f"""
                <div class="chat-bubble ai">
                    <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5568;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.1em">
                        🤖 FinBot
                    </div>
                    <div style="font-size:0.9rem;color:#e2e8f0;line-height:1.6">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

            elif "buy" in prompt.lower() or "sell" in prompt.lower():
                ticker = agent.resolve_ticker(prompt)
                with st.spinner("Planning investment…"):
                    plan = planner.plan(prompt)
                st.markdown(f"""
                <div class="chat-bubble ai" style="border-color:rgba(104,211,145,0.25)">
                    <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5568;margin-bottom:6px">
                        📋 PLANNER DECISION
                    </div>
                    <div style="font-size:0.9rem;color:#e2e8f0;line-height:1.6">{plan}</div>
                </div>
                """, unsafe_allow_html=True)

                tool_outputs = {}
                with st.spinner(f"Fetching live data for {ticker}…"):
                    price = call_tool_return_json("get_current_stock_price", {"ticker": ticker})
                    tool_outputs["price"] = price
                    trend = call_tool_return_json("analyze_market_trend", {"ticker": ticker})
                    tool_outputs["trend"] = trend
                    prediction = call_tool_return_json("predict_stock_profit", {
                        "ticker": ticker, "purchase_date": "2024-01-01", "future_date": "2026-01-01"
                    })
                    tool_outputs["prediction"] = prediction

                if price:
                    st.markdown('<div class="section-label" style="margin-top:1rem">💹 Live Data</div>', unsafe_allow_html=True)
                    render_result_payload(price)

                with st.spinner("Generating investment advice…"):
                    advice = advisor.generate_advice(prompt, tool_outputs)
                st.session_state.chat_history.append({"role": "assistant", "content": advice})
                st.markdown(f"""
                <div class="chat-bubble ai" style="border-color:rgba(99,179,237,0.3);margin-top:1rem">
                    <div style="font-family:DM Mono,monospace;font-size:0.65rem;color:#4a5568;margin-bottom:6px">
                        🤖 AI INVESTMENT ADVICE
                    </div>
                    <div style="font-size:0.9rem;color:#e2e8f0;line-height:1.6">{advice}</div>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── Inject saved portfolio if tool needs holdings and none provided ──
                params = dict(decision.parameters) if decision.parameters else {}
                if decision.tool in ("portfolio_summary", "analyze_portfolio_risk") and "holdings" not in params:
                    if st.session_state.user_portfolio:
                        params["holdings"] = st.session_state.user_portfolio
                        if decision.tool == "portfolio_summary":
                            params.setdefault("base_currency", "INR")
                    else:
                        st.warning("No portfolio saved. Please set your portfolio in the expander above.")
                        st.stop()

                with st.spinner(f"Running {decision.tool}…"):
                    result = call_tool_return_json(decision.tool, params)
                st.markdown('<div class="section-label" style="margin-top:1rem">📊 Result</div>', unsafe_allow_html=True)
                render_result_payload(result)
