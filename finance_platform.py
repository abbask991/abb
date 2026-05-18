import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FinanceOps — Automation Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { font-size: 12px !important; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-green { background: rgba(34,197,94,0.15); color: #22c55e; }
    .badge-red { background: rgba(239,68,68,0.15); color: #ef4444; }
    .badge-yellow { background: rgba(245,158,11,0.15); color: #f59e0b; }
    .badge-blue { background: rgba(59,130,246,0.15); color: #3b82f6; }
    .badge-purple { background: rgba(168,85,247,0.15); color: #a855f7; }
    .alert-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .alert-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
    .alert-desc { font-size: 12px; color: #94a3b8; }
    .section-divider { margin: 8px 0 16px 0; border-top: 1px solid #334155; }
    div[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Mock Data
# ──────────────────────────────────────────────

transactions_data = [
    {"ID": "TXN-001", "Type": "Deposit", "Amount": 25000, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Alpha", "Client": "Acme Corp", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-18 08:23", "Description": "Client deposit via wire"},
    {"ID": "TXN-002", "Type": "Withdrawal", "Amount": 12500, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Beta", "Client": "Globe Ltd", "PSP": "Adyen", "Bank": "HSBC", "Timestamp": "2026-05-18 09:15", "Description": "Client withdrawal request"},
    {"ID": "TXN-003", "Type": "Deposit", "Amount": 8700, "Currency": "EUR", "Status": "Pending", "Source System": "PSP-Alpha", "Client": "NovaTech", "PSP": "Stripe", "Bank": "Deutsche", "Timestamp": "2026-05-18 09:42", "Description": "Pending deposit confirmation"},
    {"ID": "TXN-004", "Type": "Transfer", "Amount": 50000, "Currency": "USD", "Status": "Completed", "Source System": "Internal", "Client": "Internal", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 10:01", "Description": "Treasury transfer"},
    {"ID": "TXN-005", "Type": "Fee", "Amount": 125, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Alpha", "Client": "Acme Corp", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-18 08:23", "Description": "Processing fee"},
    {"ID": "TXN-006", "Type": "Commission", "Amount": 375, "Currency": "USD", "Status": "Completed", "Source System": "IB-System", "Client": "IB-Partner-01", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 10:30", "Description": "IB commission payout"},
    {"ID": "TXN-007", "Type": "Deposit", "Amount": 150000, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Gamma", "Client": "MegaFund", "PSP": "Worldpay", "Bank": "Barclays", "Timestamp": "2026-05-17 14:22", "Description": "Large client deposit"},
    {"ID": "TXN-008", "Type": "Withdrawal", "Amount": 45000, "Currency": "GBP", "Status": "Failed", "Source System": "PSP-Beta", "Client": "BritCo", "PSP": "Adyen", "Bank": "Barclays", "Timestamp": "2026-05-17 16:45", "Description": "Failed withdrawal - insufficient PSP balance"},
    {"ID": "TXN-009", "Type": "Deposit", "Amount": 32000, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Alpha", "Client": "SolarInc", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-17 11:10", "Description": "Client deposit"},
    {"ID": "TXN-010", "Type": "Withdrawal", "Amount": 18200, "Currency": "EUR", "Status": "Pending", "Source System": "PSP-Beta", "Client": "EuroTrade", "PSP": "Adyen", "Bank": "Deutsche", "Timestamp": "2026-05-18 07:50", "Description": "Pending withdrawal"},
    {"ID": "TXN-011", "Type": "Fee", "Amount": 89, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Gamma", "Client": "MegaFund", "PSP": "Worldpay", "Bank": "Barclays", "Timestamp": "2026-05-17 14:22", "Description": "Processing fee"},
    {"ID": "TXN-012", "Type": "Deposit", "Amount": 5600, "Currency": "USD", "Status": "Reversed", "Source System": "PSP-Alpha", "Client": "QuickPay", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-16 09:30", "Description": "Reversed deposit - chargeback"},
    {"ID": "TXN-013", "Type": "Transfer", "Amount": 200000, "Currency": "USD", "Status": "Completed", "Source System": "Internal", "Client": "Internal", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-16 15:00", "Description": "Liquidity rebalancing"},
    {"ID": "TXN-014", "Type": "Commission", "Amount": 1250, "Currency": "USD", "Status": "Completed", "Source System": "IB-System", "Client": "IB-Partner-02", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-16 16:00", "Description": "Monthly IB commission"},
    {"ID": "TXN-015", "Type": "Deposit", "Amount": 72000, "Currency": "USD", "Status": "Completed", "Source System": "PSP-Alpha", "Client": "TradeCo", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-15 10:20", "Description": "Client deposit"},
]

reconciliation_data = [
    {"Case ID": "REC-001", "Transaction": "TXN-001", "Status": "Matched", "Match Score": 100, "Amount": 25000, "Expected": 25000, "Source → Target": "PSP-Alpha → Bank-Chase", "Case Status": "Closed", "Notes": "Auto-matched"},
    {"Case ID": "REC-002", "Transaction": "TXN-003", "Status": "Partial", "Match Score": 72, "Amount": 8700, "Expected": 9000, "Source → Target": "PSP-Alpha → Bank-Deutsche", "Case Status": "Investigating", "Notes": "Amount mismatch — FX differential suspected"},
    {"Case ID": "REC-003", "Transaction": "TXN-008", "Status": "Exception", "Match Score": 0, "Amount": 45000, "Expected": 45000, "Source → Target": "PSP-Beta → Bank-Barclays", "Case Status": "Open", "Notes": "Failed withdrawal — no bank record found"},
    {"Case ID": "REC-004", "Transaction": "TXN-007", "Status": "Matched", "Match Score": 100, "Amount": 150000, "Expected": 150000, "Source → Target": "PSP-Gamma → Bank-Barclays", "Case Status": "Closed", "Notes": "Auto-matched"},
    {"Case ID": "REC-005", "Transaction": "TXN-012", "Status": "Exception", "Match Score": 50, "Amount": 5600, "Expected": 5600, "Source → Target": "PSP-Alpha → Bank-Chase", "Case Status": "Investigating", "Notes": "Chargeback reversal — investigating"},
    {"Case ID": "REC-006", "Transaction": "TXN-009", "Status": "Matched", "Match Score": 100, "Amount": 32000, "Expected": 32000, "Source → Target": "PSP-Alpha → Bank-Chase", "Case Status": "Closed", "Notes": "Auto-matched"},
    {"Case ID": "REC-007", "Transaction": "TXN-010", "Status": "Partial", "Match Score": 65, "Amount": 18200, "Expected": 18500, "Source → Target": "PSP-Beta → Bank-Deutsche", "Case Status": "Open", "Notes": "Pending — amount discrepancy"},
    {"Case ID": "REC-008", "Transaction": "TXN-002", "Status": "Matched", "Match Score": 100, "Amount": 12500, "Expected": 12500, "Source → Target": "PSP-Beta → Bank-HSBC", "Case Status": "Closed", "Notes": "Auto-matched"},
]

ledger_data = [
    {"ID": "LED-001", "Type": "Debit", "Account": "Company Cash", "Category": "Company Cash", "Amount": 25000, "Currency": "USD", "Reference": "TXN-001", "Timestamp": "2026-05-18 08:23", "Description": "Cash received from deposit"},
    {"ID": "LED-002", "Type": "Credit", "Account": "Client Liability - Acme Corp", "Category": "Client Liability", "Amount": 25000, "Currency": "USD", "Reference": "TXN-001", "Timestamp": "2026-05-18 08:23", "Description": "Client balance credited"},
    {"ID": "LED-003", "Type": "Credit", "Account": "Company Cash", "Category": "Company Cash", "Amount": 12500, "Currency": "USD", "Reference": "TXN-002", "Timestamp": "2026-05-18 09:15", "Description": "Cash disbursed for withdrawal"},
    {"ID": "LED-004", "Type": "Debit", "Account": "Client Liability - Globe Ltd", "Category": "Client Liability", "Amount": 12500, "Currency": "USD", "Reference": "TXN-002", "Timestamp": "2026-05-18 09:15", "Description": "Client balance debited"},
    {"ID": "LED-005", "Type": "Debit", "Account": "PSP Receivable - Stripe", "Category": "PSP Accounts", "Amount": 8700, "Currency": "EUR", "Reference": "TXN-003", "Timestamp": "2026-05-18 09:42", "Description": "Pending PSP settlement"},
    {"ID": "LED-006", "Type": "Debit", "Account": "Company Cash - Operating", "Category": "Company Cash", "Amount": 50000, "Currency": "USD", "Reference": "TXN-004", "Timestamp": "2026-05-18 10:01", "Description": "Treasury transfer in"},
    {"ID": "LED-007", "Type": "Credit", "Account": "Company Cash - Reserve", "Category": "Company Cash", "Amount": 50000, "Currency": "USD", "Reference": "TXN-004", "Timestamp": "2026-05-18 10:01", "Description": "Treasury transfer out"},
    {"ID": "LED-008", "Type": "Debit", "Account": "Fee Income", "Category": "Fee Accounts", "Amount": 125, "Currency": "USD", "Reference": "TXN-005", "Timestamp": "2026-05-18 08:23", "Description": "Processing fee collected"},
    {"ID": "LED-009", "Type": "Credit", "Account": "Commission Payable - IB-01", "Category": "Commission Accounts", "Amount": 375, "Currency": "USD", "Reference": "TXN-006", "Timestamp": "2026-05-18 10:30", "Description": "IB commission accrued"},
    {"ID": "LED-010", "Type": "Debit", "Account": "Company Cash", "Category": "Company Cash", "Amount": 150000, "Currency": "USD", "Reference": "TXN-007", "Timestamp": "2026-05-17 14:22", "Description": "Large deposit received"},
    {"ID": "LED-011", "Type": "Credit", "Account": "Client Liability - MegaFund", "Category": "Client Liability", "Amount": 150000, "Currency": "USD", "Reference": "TXN-007", "Timestamp": "2026-05-17 14:22", "Description": "Client balance credited"},
    {"ID": "LED-012", "Type": "Debit", "Account": "Company Cash", "Category": "Company Cash", "Amount": 32000, "Currency": "USD", "Reference": "TXN-009", "Timestamp": "2026-05-17 11:10", "Description": "Deposit received"},
    {"ID": "LED-013", "Type": "Credit", "Account": "Client Liability - SolarInc", "Category": "Client Liability", "Amount": 32000, "Currency": "USD", "Reference": "TXN-009", "Timestamp": "2026-05-17 11:10", "Description": "Client balance credited"},
    {"ID": "LED-014", "Type": "Credit", "Account": "Client Liability - EuroTrade", "Category": "Client Liability", "Amount": 18200, "Currency": "EUR", "Reference": "TXN-010", "Timestamp": "2026-05-18 07:50", "Description": "Pending withdrawal reserved"},
    {"ID": "LED-015", "Type": "Debit", "Account": "Fee Income", "Category": "Fee Accounts", "Amount": 89, "Currency": "USD", "Reference": "TXN-011", "Timestamp": "2026-05-17 14:22", "Description": "Processing fee collected"},
]

alerts_data = [
    {"ID": "ALT-001", "Title": "Cash Imbalance Detected", "Category": "Financial", "Severity": "Critical", "Status": "Open", "Description": "Company cash account shows $12,400 discrepancy between expected and actual balance.", "Linked Txns": "TXN-004, TXN-013", "Created": "2026-05-18 10:15"},
    {"ID": "ALT-002", "Title": "Withdrawal Spike Alert", "Category": "Financial", "Severity": "High", "Status": "Investigating", "Description": "Withdrawal volume increased 340% compared to 7-day average.", "Linked Txns": "TXN-002, TXN-008, TXN-010", "Created": "2026-05-18 09:30"},
    {"ID": "ALT-003", "Title": "PSP Settlement Delay - Adyen", "Category": "Operational", "Severity": "Medium", "Status": "Open", "Description": "Adyen settlement batch delayed by 4 hours. Estimated funds: €63,200.", "Linked Txns": "TXN-002, TXN-010", "Created": "2026-05-18 08:00"},
    {"ID": "ALT-004", "Title": "Reconciliation Exception Rate High", "Category": "Operational", "Severity": "High", "Status": "Investigating", "Description": "Exception rate at 18% for today, above 5% threshold.", "Linked Txns": "TXN-008, TXN-012", "Created": "2026-05-18 10:00"},
    {"ID": "ALT-005", "Title": "Chargeback Reversal", "Category": "Financial", "Severity": "Medium", "Status": "Resolved", "Description": "Chargeback on TXN-012 processed. Client QuickPay debited $5,600.", "Linked Txns": "TXN-012", "Created": "2026-05-16 10:30"},
    {"ID": "ALT-006", "Title": "Low Liquidity Buffer Warning", "Category": "Financial", "Severity": "Critical", "Status": "Open", "Description": "Available liquidity buffer dropped below 15% threshold. Current: 11.2%.", "Linked Txns": "", "Created": "2026-05-18 07:00"},
]

bank_balances = [
    {"Bank": "Chase", "Currency": "USD", "Balance": 2450000},
    {"Bank": "HSBC", "Currency": "USD", "Balance": 890000},
    {"Bank": "Deutsche Bank", "Currency": "EUR", "Balance": 1250000},
    {"Bank": "Barclays", "Currency": "GBP", "Balance": 675000},
    {"Bank": "Barclays", "Currency": "USD", "Balance": 320000},
]

psp_balances = [
    {"PSP": "Stripe", "Currency": "USD", "Balance": 185000, "Pending In": 32000, "Pending Out": 8500},
    {"PSP": "Stripe", "Currency": "EUR", "Balance": 45000, "Pending In": 8700, "Pending Out": 0},
    {"PSP": "Adyen", "Currency": "USD", "Balance": 92000, "Pending In": 0, "Pending Out": 12500},
    {"PSP": "Adyen", "Currency": "EUR", "Balance": 63200, "Pending In": 0, "Pending Out": 18200},
    {"PSP": "Worldpay", "Currency": "USD", "Balance": 210000, "Pending In": 0, "Pending Out": 0},
    {"PSP": "Worldpay", "Currency": "GBP", "Balance": 55000, "Pending In": 0, "Pending Out": 45000},
]

cash_flow_data = [
    {"Date": "May 12", "Deposits": 185000, "Withdrawals": 72000, "Net Flow": 113000},
    {"Date": "May 13", "Deposits": 210000, "Withdrawals": 95000, "Net Flow": 115000},
    {"Date": "May 14", "Deposits": 145000, "Withdrawals": 120000, "Net Flow": 25000},
    {"Date": "May 15", "Deposits": 290000, "Withdrawals": 88000, "Net Flow": 202000},
    {"Date": "May 16", "Deposits": 178000, "Withdrawals": 156000, "Net Flow": 22000},
    {"Date": "May 17", "Deposits": 320000, "Withdrawals": 105000, "Net Flow": 215000},
    {"Date": "May 18", "Deposits": 265700, "Withdrawals": 75700, "Net Flow": 190000},
]

liquidity_trend = [
    {"Date": "May 12", "Available": 4200000, "Buffer %": 18.2},
    {"Date": "May 13", "Available": 4315000, "Buffer %": 17.8},
    {"Date": "May 14", "Available": 4050000, "Buffer %": 15.5},
    {"Date": "May 15", "Available": 4400000, "Buffer %": 16.1},
    {"Date": "May 16", "Available": 4150000, "Buffer %": 14.2},
    {"Date": "May 17", "Available": 4580000, "Buffer %": 13.8},
    {"Date": "May 18", "Available": 4585000, "Buffer %": 11.2},
]

profitability_data = [
    {"Client": "Acme Corp", "Revenue": 45000, "Costs": 12000, "Profit": 33000},
    {"Client": "Globe Ltd", "Revenue": 28000, "Costs": 9500, "Profit": 18500},
    {"Client": "MegaFund", "Revenue": 82000, "Costs": 18000, "Profit": 64000},
    {"Client": "NovaTech", "Revenue": 15000, "Costs": 6200, "Profit": 8800},
    {"Client": "SolarInc", "Revenue": 32000, "Costs": 8800, "Profit": 23200},
    {"Client": "TradeCo", "Revenue": 51000, "Costs": 14500, "Profit": 36500},
]

ib_profitability_data = [
    {"Partner": "IB-Partner-01", "Clients": 42, "Volume": 1250000, "Commission": 18750, "Net Revenue": 62500},
    {"Partner": "IB-Partner-02", "Clients": 28, "Volume": 890000, "Commission": 13350, "Net Revenue": 44500},
    {"Partner": "IB-Partner-03", "Clients": 65, "Volume": 2100000, "Commission": 31500, "Net Revenue": 105000},
    {"Partner": "IB-Partner-04", "Clients": 15, "Volume": 420000, "Commission": 6300, "Net Revenue": 21000},
]

monthly_kpis = [
    {"Month": "Jan", "Net Flow": 1200000, "Op Costs": 85000, "Exceptions": 12},
    {"Month": "Feb", "Net Flow": 980000, "Op Costs": 78000, "Exceptions": 8},
    {"Month": "Mar", "Net Flow": 1450000, "Op Costs": 92000, "Exceptions": 15},
    {"Month": "Apr", "Net Flow": 1100000, "Op Costs": 88000, "Exceptions": 10},
    {"Month": "May", "Net Flow": 1680000, "Op Costs": 95000, "Exceptions": 18},
]

campaign_data = [
    {"Campaign": "Welcome Bonus", "Spend": 45000, "Deposits": 180000, "ROI %": 300},
    {"Campaign": "Loyalty Program", "Spend": 22000, "Deposits": 95000, "ROI %": 332},
    {"Campaign": "Referral Bonus", "Spend": 15000, "Deposits": 72000, "ROI %": 380},
    {"Campaign": "VIP Cashback", "Spend": 35000, "Deposits": 110000, "ROI %": 214},
]

# ──────────────────────────────────────────────
# DataFrames
# ──────────────────────────────────────────────
df_txn = pd.DataFrame(transactions_data)
df_rec = pd.DataFrame(reconciliation_data)
df_ledger = pd.DataFrame(ledger_data)
df_alerts = pd.DataFrame(alerts_data)
df_bank = pd.DataFrame(bank_balances)
df_psp = pd.DataFrame(psp_balances)
df_cash = pd.DataFrame(cash_flow_data)
df_liq = pd.DataFrame(liquidity_trend)
df_profit = pd.DataFrame(profitability_data)
df_ib = pd.DataFrame(ib_profitability_data)
df_kpi = pd.DataFrame(monthly_kpis)
df_camp = pd.DataFrame(campaign_data)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def fmt(n):
    return f"${n:,.0f}"

def severity_color(sev):
    colors = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵"}
    return colors.get(sev, "⚪")

def status_badge(status):
    colors = {
        "Completed": "badge-green", "Matched": "badge-green", "Closed": "badge-green", "Resolved": "badge-green",
        "Pending": "badge-yellow", "Partial": "badge-yellow", "Investigating": "badge-yellow", "Open": "badge-yellow",
        "Failed": "badge-red", "Exception": "badge-red", "Critical": "badge-red",
        "Reversed": "badge-purple",
    }
    css = colors.get(status, "badge-blue")
    return f'<span class="badge {css}">{status}</span>'

def chart_colors():
    return {
        "green": "#22c55e", "red": "#ef4444", "blue": "#3b82f6",
        "purple": "#6366f1", "yellow": "#f59e0b", "cyan": "#06b6d4",
    }

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💰 **FinanceOps**")
    st.caption("AUTOMATION PLATFORM")
    st.markdown("---")

    open_alert_count = len(df_alerts[df_alerts["Status"] == "Open"])

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "💸 Transactions",
            "🔄 Reconciliation",
            "📒 Ledger",
            "💧 Liquidity",
            f"⚠️ Alerts & Exceptions ({open_alert_count})",
            "📈 Reports & Analytics",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(f"🟢 **System Online**")
    st.caption(datetime.now().strftime("%a, %b %d, %Y"))


# ══════════════════════════════════════════════
# 1. DASHBOARD
# ══════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.caption("Finance Automation Platform / Overview")

    total_deposits = df_txn[(df_txn["Type"] == "Deposit") & (df_txn["Status"] == "Completed")]["Amount"].sum()
    total_withdrawals = df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Completed")]["Amount"].sum()
    net_flow = total_deposits - total_withdrawals
    total_cash = df_bank["Balance"].sum()
    active_alerts = len(df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])])
    critical_count = len(df_alerts[(df_alerts["Severity"] == "Critical") & (df_alerts["Status"] != "Resolved")])
    buffer_pct = df_liq.iloc[-1]["Buffer %"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash Position", fmt(total_cash), "+3.2%")
    c2.metric("Net Flow Today", fmt(net_flow), "+12.5%")
    c3.metric("Liquidity Buffer", f"{buffer_pct}%", "-2.6%", delta_color="inverse")
    c4.metric("Active Alerts", active_alerts, f"{critical_count} critical", delta_color="inverse")

    st.markdown("")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Cash Flow (7-Day)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Deposits"], mode="lines", name="Deposits", fill="tozeroy", line=dict(color="#22c55e", width=2), fillcolor="rgba(34,197,94,0.15)"))
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Withdrawals"], mode="lines", name="Withdrawals", fill="tozeroy", line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Transaction Mix")
        type_counts = df_txn["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_pie = px.pie(type_counts, values="Count", names="Type", hole=0.55, color_discrete_sequence=["#6366f1", "#22c55e", "#ef4444", "#f59e0b", "#a855f7"])
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=True, legend=dict(font=dict(size=11)))
        fig_pie.update_traces(textinfo="none")
        st.plotly_chart(fig_pie, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Liquidity Trend")
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["Available"], mode="lines", name="Available Cash", fill="tozeroy", line=dict(color="#6366f1", width=2), fillcolor="rgba(99,102,241,0.15)"))
        fig_liq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_liq, use_container_width=True)

    with col_b:
        st.subheader("Net Flow (7-Day)")
        fig_net = px.bar(df_cash, x="Date", y="Net Flow", color_discrete_sequence=["#6366f1"])
        fig_net.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_net, use_container_width=True)

    st.subheader(f"Active Alerts ({active_alerts})")
    for _, alert in df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])].iterrows():
        icon = severity_color(alert["Severity"])
        st.markdown(f"""
        <div class="alert-card">
            <div class="alert-title">{icon} {alert["Title"]}</div>
            <div class="alert-desc">{alert["Description"]}</div>
            <div style="margin-top:6px">
                {status_badge(alert["Severity"])} {status_badge(alert["Category"])} {status_badge(alert["Status"])}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 2. TRANSACTIONS
# ══════════════════════════════════════════════
elif page == "💸 Transactions":
    st.title("💸 Transactions")
    st.caption("All Transactions — Deposits, Withdrawals, Transfers, Fees & Commissions")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        search = st.text_input("🔍 Search", placeholder="ID, client, description...")
    with col2:
        type_filter = st.selectbox("Type", ["All"] + sorted(df_txn["Type"].unique().tolist()))
    with col3:
        status_filter = st.selectbox("Status", ["All"] + sorted(df_txn["Status"].unique().tolist()))
    with col4:
        psp_filter = st.selectbox("PSP", ["All"] + sorted(df_txn["PSP"].unique().tolist()))
    with col5:
        bank_filter = st.selectbox("Bank", ["All"] + sorted(df_txn["Bank"].unique().tolist()))

    filtered = df_txn.copy()
    if search:
        mask = filtered.apply(lambda r: search.lower() in r["ID"].lower() or search.lower() in r["Client"].lower() or search.lower() in r["Description"].lower(), axis=1)
        filtered = filtered[mask]
    if type_filter != "All":
        filtered = filtered[filtered["Type"] == type_filter]
    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]
    if psp_filter != "All":
        filtered = filtered[filtered["PSP"] == psp_filter]
    if bank_filter != "All":
        filtered = filtered[filtered["Bank"] == bank_filter]

    st.caption(f"Showing {len(filtered)} of {len(df_txn)} transactions")
    st.dataframe(
        filtered[["ID", "Type", "Client", "Amount", "Currency", "PSP", "Bank", "Status", "Timestamp"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Amount": st.column_config.NumberColumn(format="$%d"),
        },
    )

    st.markdown("---")
    st.subheader("Transaction Details")
    selected_txn = st.selectbox("Select transaction to view details", filtered["ID"].tolist())
    if selected_txn:
        tx = df_txn[df_txn["ID"] == selected_txn].iloc[0]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**ID:** {tx['ID']}")
            st.markdown(f"**Type:** {tx['Type']}")
            st.markdown(f"**Amount:** {fmt(tx['Amount'])} {tx['Currency']}")
            st.markdown(f"**Status:** {tx['Status']}")
            st.markdown(f"**Client:** {tx['Client']}")
        with c2:
            st.markdown(f"**Source System:** {tx['Source System']}")
            st.markdown(f"**PSP:** {tx['PSP']}")
            st.markdown(f"**Bank:** {tx['Bank']}")
            st.markdown(f"**Timestamp:** {tx['Timestamp']}")
            st.markdown(f"**Description:** {tx['Description']}")


# ══════════════════════════════════════════════
# 3. RECONCILIATION
# ══════════════════════════════════════════════
elif page == "🔄 Reconciliation":
    st.title("🔄 Reconciliation")
    st.caption("Matched Transactions, Partial Matches & Exceptions")

    matched = len(df_rec[df_rec["Status"] == "Matched"])
    partial = len(df_rec[df_rec["Status"] == "Partial"])
    exceptions = len(df_rec[df_rec["Status"] == "Exception"])

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Matched", matched, f"{matched/len(df_rec)*100:.0f}%")
    c2.metric("⚠️ Partial", partial, f"{partial/len(df_rec)*100:.0f}%")
    c3.metric("❌ Exceptions", exceptions, f"{exceptions/len(df_rec)*100:.0f}%", delta_color="inverse")

    tab1, tab2, tab3, tab4 = st.tabs(["All Cases", "Matched", "Partial", "Exceptions"])

    def show_rec_table(data):
        st.dataframe(
            data[["Case ID", "Transaction", "Status", "Match Score", "Amount", "Expected", "Source → Target", "Case Status", "Notes"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Match Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"),
                "Amount": st.column_config.NumberColumn(format="$%d"),
                "Expected": st.column_config.NumberColumn(format="$%d"),
            },
        )

    with tab1:
        show_rec_table(df_rec)
    with tab2:
        show_rec_table(df_rec[df_rec["Status"] == "Matched"])
    with tab3:
        show_rec_table(df_rec[df_rec["Status"] == "Partial"])
    with tab4:
        show_rec_table(df_rec[df_rec["Status"] == "Exception"])

    st.markdown("---")
    st.subheader("Reconciliation Cases — Actions")
    open_cases = df_rec[df_rec["Case Status"].isin(["Open", "Investigating"])]
    if len(open_cases) > 0:
        selected_case = st.selectbox("Select case", open_cases["Case ID"].tolist())
        case = df_rec[df_rec["Case ID"] == selected_case].iloc[0]
        st.info(f"**{case['Case ID']}** — {case['Notes']} (Match Score: {case['Match Score']}%)")
        ac1, ac2, ac3 = st.columns(3)
        ac1.button("🔍 Investigate", key="inv", use_container_width=True)
        ac2.button("✅ Approve", key="app", use_container_width=True)
        ac3.button("❌ Reject", key="rej", use_container_width=True)
    else:
        st.success("All cases are resolved!")

    st.markdown("---")
    st.subheader("Reconciliation Log")
    st.dataframe(
        df_rec[["Case ID", "Transaction", "Case Status", "Notes"]].sort_values("Case ID", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════
# 4. LEDGER
# ══════════════════════════════════════════════
elif page == "📒 Ledger":
    st.title("📒 Ledger")
    st.caption("Ledger Entries — Debit & Credit, Accounts, Audit Trail")

    total_debits = df_ledger[df_ledger["Type"] == "Debit"]["Amount"].sum()
    total_credits = df_ledger[df_ledger["Type"] == "Credit"]["Amount"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Debits", fmt(total_debits))
    c2.metric("Total Credits", fmt(total_credits))
    c3.metric("Entries", len(df_ledger))

    tab1, tab2, tab3 = st.tabs(["All Entries", "Debit Entries", "Credit Entries"])

    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        cat_filter = st.selectbox("Account Category", ["All"] + sorted(df_ledger["Category"].unique().tolist()))

    filtered_ledger = df_ledger.copy()
    if cat_filter != "All":
        filtered_ledger = filtered_ledger[filtered_ledger["Category"] == cat_filter]

    def show_ledger(data):
        st.dataframe(
            data[["ID", "Type", "Account", "Amount", "Currency", "Reference", "Timestamp", "Description"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Amount": st.column_config.NumberColumn(format="$%d")},
        )

    with tab1:
        show_ledger(filtered_ledger)
    with tab2:
        show_ledger(filtered_ledger[filtered_ledger["Type"] == "Debit"])
    with tab3:
        show_ledger(filtered_ledger[filtered_ledger["Type"] == "Credit"])

    st.markdown("---")
    col_acct, col_audit = st.columns(2)

    with col_acct:
        st.subheader("Account Summary")
        acct_summary = df_ledger.groupby("Account").agg(
            Debits=("Amount", lambda x: x[df_ledger.loc[x.index, "Type"] == "Debit"].sum()),
            Credits=("Amount", lambda x: x[df_ledger.loc[x.index, "Type"] == "Credit"].sum()),
            Entries=("ID", "count"),
        ).reset_index()
        acct_summary["Net"] = acct_summary["Debits"] - acct_summary["Credits"]
        st.dataframe(
            acct_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Debits": st.column_config.NumberColumn(format="$%d"),
                "Credits": st.column_config.NumberColumn(format="$%d"),
                "Net": st.column_config.NumberColumn(format="$%d"),
            },
        )

    with col_audit:
        st.subheader("Audit Trail")
        audit = df_ledger.sort_values("Timestamp", ascending=False).head(8)
        st.dataframe(
            audit[["Timestamp", "ID", "Type", "Description"]],
            use_container_width=True,
            hide_index=True,
        )


# ══════════════════════════════════════════════
# 5. LIQUIDITY
# ══════════════════════════════════════════════
elif page == "💧 Liquidity":
    st.title("💧 Liquidity")
    st.caption("Bank Balances, PSP Balances, Liquidity Calculation")

    total_bank = df_bank["Balance"].sum()
    total_psp = df_psp["Balance"].sum()
    pending_withdrawals = df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Pending")]["Amount"].sum() + df_psp["Pending Out"].sum()
    bonus_exposure = 125000
    commission_liabilities = 48500
    available_cash = total_bank + total_psp
    net_liquidity = available_cash - pending_withdrawals - bonus_exposure - commission_liabilities
    buffer = (net_liquidity / available_cash) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available Cash", fmt(available_cash))
    c2.metric("Net Liquidity", fmt(net_liquidity))
    c3.metric("Liquidity Buffer", f"{buffer:.1f}%", delta_color="inverse" if buffer < 15 else "normal")
    c4.metric("Pending Withdrawals", fmt(pending_withdrawals))

    col_bank, col_psp = st.columns(2)

    with col_bank:
        st.subheader("🏦 Bank Balances")
        st.dataframe(
            df_bank,
            use_container_width=True,
            hide_index=True,
            column_config={"Balance": st.column_config.NumberColumn(format="$%d")},
        )

    with col_psp:
        st.subheader("💳 PSP Balances")
        st.dataframe(
            df_psp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Balance": st.column_config.NumberColumn(format="$%d"),
                "Pending In": st.column_config.NumberColumn(format="$%d"),
                "Pending Out": st.column_config.NumberColumn(format="$%d"),
            },
        )

    col_calc, col_trend = st.columns(2)

    with col_calc:
        st.subheader("Liquidity Calculation")
        calc_data = pd.DataFrame({
            "Item": ["Available Cash", "Pending Withdrawals", "Bonus Exposure", "Commission Liabilities", "Net Liquidity"],
            "Amount": [available_cash, pending_withdrawals, bonus_exposure, commission_liabilities, net_liquidity],
        })
        fig_calc = px.bar(calc_data, y="Item", x="Amount", orientation="h", color="Item", color_discrete_sequence=["#22c55e", "#ef4444", "#f59e0b", "#a855f7", "#6366f1"])
        fig_calc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_calc, use_container_width=True)

    with col_trend:
        st.subheader("Liquidity Buffer Trend")
        fig_buf = go.Figure()
        fig_buf.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["Buffer %"], mode="lines+markers", name="Buffer %", line=dict(color="#f59e0b", width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.15)"))
        fig_buf.add_hline(y=15, line_dash="dash", line_color="#ef4444", annotation_text="15% Threshold")
        fig_buf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=300, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_buf, use_container_width=True)

    st.subheader("Liability Breakdown")
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric("📉 Pending Withdrawals", fmt(pending_withdrawals))
    lc2.metric("🎁 Bonus Exposure", fmt(bonus_exposure))
    lc3.metric("👥 Commission Liabilities", fmt(commission_liabilities))


# ══════════════════════════════════════════════
# 6. ALERTS & EXCEPTIONS
# ══════════════════════════════════════════════
elif "Alerts" in page:
    st.title("⚠️ Alerts & Exceptions")
    st.caption("Financial & Operational Alerts, Case Management")

    critical = len(df_alerts[(df_alerts["Severity"] == "Critical") & (df_alerts["Status"] != "Resolved")])
    open_count = len(df_alerts[df_alerts["Status"] == "Open"])
    investigating = len(df_alerts[df_alerts["Status"] == "Investigating"])
    resolved = len(df_alerts[df_alerts["Status"] == "Resolved"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Critical", critical)
    c2.metric("🟡 Open", open_count)
    c3.metric("🔵 Investigating", investigating)
    c4.metric("🟢 Resolved", resolved)

    tab_all, tab_fin, tab_ops = st.tabs(["All Alerts", "Financial Alerts", "Operational Alerts"])

    col_sev, col_stat = st.columns(2)
    with col_sev:
        sev_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
    with col_stat:
        stat_filter = st.selectbox("Status", ["All", "Open", "Investigating", "Resolved", "Closed"], key="alert_status")

    def filter_alerts(data):
        d = data.copy()
        if sev_filter != "All":
            d = d[d["Severity"] == sev_filter]
        if stat_filter != "All":
            d = d[d["Status"] == stat_filter]
        return d

    def show_alerts(data):
        if len(data) == 0:
            st.success("No alerts match your filters")
            return
        for _, alert in data.iterrows():
            icon = severity_color(alert["Severity"])
            st.markdown(f"""
            <div class="alert-card">
                <div class="alert-title">{icon} {alert["Title"]}</div>
                <div class="alert-desc">{alert["Description"]}</div>
                <div style="margin-top:6px">
                    {status_badge(alert["Severity"])} {status_badge(alert["Category"])} {status_badge(alert["Status"])}
                    <span style="font-size:11px; color:#64748b; margin-left:8px">{alert["Created"]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab_all:
        show_alerts(filter_alerts(df_alerts))
    with tab_fin:
        show_alerts(filter_alerts(df_alerts[df_alerts["Category"] == "Financial"]))
    with tab_ops:
        show_alerts(filter_alerts(df_alerts[df_alerts["Category"] == "Operational"]))

    st.markdown("---")
    st.subheader("Case Management")
    active_alerts = df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])]
    if len(active_alerts) > 0:
        selected_alert = st.selectbox("Select alert", active_alerts["ID"].tolist())
        al = df_alerts[df_alerts["ID"] == selected_alert].iloc[0]
        st.info(f"**{al['Title']}** — {al['Description']}")
        st.markdown(f"**Severity:** {al['Severity']} | **Category:** {al['Category']} | **Linked Txns:** {al['Linked Txns']}")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.button("📋 Open Case", use_container_width=True)
        bc2.button("🔍 Investigate", use_container_width=True)
        bc3.button("✅ Resolve", use_container_width=True)
        bc4.button("❌ Dismiss", use_container_width=True)


# ══════════════════════════════════════════════
# 7. REPORTS & ANALYTICS
# ══════════════════════════════════════════════
elif page == "📈 Reports & Analytics":
    st.title("📈 Reports & Analytics")
    st.caption("Profitability Analysis, Financial KPIs, Generated Reports")

    tab_profit, tab_kpis, tab_reports = st.tabs(["Profitability Analysis", "Financial KPIs", "Generated Reports"])

    with tab_profit:
        total_revenue = df_profit["Revenue"].sum()
        total_costs = df_profit["Costs"].sum()
        total_profit = df_profit["Profit"].sum()
        margin = (total_profit / total_revenue) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Revenue", fmt(total_revenue), "+8.2%")
        c2.metric("Operating Costs", fmt(total_costs), "-3.1%", delta_color="inverse")
        c3.metric("Net Profit", fmt(total_profit), "+12.4%")
        c4.metric("Profit Margin", f"{margin:.1f}%", "+1.8pp")

        profit_tab1, profit_tab2, profit_tab3 = st.tabs(["Client Profitability", "IB Profitability", "Campaign Profitability"])

        with profit_tab1:
            col_chart, col_table = st.columns(2)
            with col_chart:
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Revenue"], name="Revenue", marker_color="#22c55e"))
                fig_cp.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Costs"], name="Costs", marker_color="#ef4444"))
                fig_cp.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Profit"], name="Profit", marker_color="#6366f1"))
                fig_cp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0, r=0, t=10, b=0), barmode="group")
                st.plotly_chart(fig_cp, use_container_width=True)
            with col_table:
                profit_display = df_profit.copy()
                profit_display["Margin"] = (profit_display["Profit"] / profit_display["Revenue"] * 100).round(1).astype(str) + "%"
                st.dataframe(
                    profit_display.sort_values("Profit", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Revenue": st.column_config.NumberColumn(format="$%d"),
                        "Costs": st.column_config.NumberColumn(format="$%d"),
                        "Profit": st.column_config.NumberColumn(format="$%d"),
                    },
                )

        with profit_tab2:
            col_chart2, col_table2 = st.columns(2)
            with col_chart2:
                fig_ib = go.Figure()
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"], y=df_ib["Net Revenue"], name="Net Revenue", marker_color="#22c55e"))
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"], y=df_ib["Commission"], name="Commission", marker_color="#a855f7"))
                fig_ib.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0, r=0, t=10, b=0), barmode="group")
                st.plotly_chart(fig_ib, use_container_width=True)
            with col_table2:
                st.dataframe(
                    df_ib.sort_values("Net Revenue", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Volume": st.column_config.NumberColumn(format="$%d"),
                        "Commission": st.column_config.NumberColumn(format="$%d"),
                        "Net Revenue": st.column_config.NumberColumn(format="$%d"),
                    },
                )

        with profit_tab3:
            col_chart3, col_table3 = st.columns(2)
            with col_chart3:
                fig_camp = go.Figure()
                fig_camp.add_trace(go.Bar(x=df_camp["Campaign"], y=df_camp["Spend"], name="Spend", marker_color="#ef4444"))
                fig_camp.add_trace(go.Bar(x=df_camp["Campaign"], y=df_camp["Deposits"], name="Deposits Generated", marker_color="#22c55e"))
                fig_camp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0, r=0, t=10, b=0), barmode="group")
                st.plotly_chart(fig_camp, use_container_width=True)
            with col_table3:
                st.dataframe(
                    df_camp.sort_values("ROI %", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Spend": st.column_config.NumberColumn(format="$%d"),
                        "Deposits": st.column_config.NumberColumn(format="$%d"),
                        "ROI %": st.column_config.ProgressColumn(min_value=0, max_value=400, format="%d%%"),
                    },
                )

    with tab_kpis:
        col_nf, col_oc = st.columns(2)

        with col_nf:
            st.subheader("Net Flow Trend")
            fig_nf = px.line(df_kpi, x="Month", y="Net Flow", markers=True, color_discrete_sequence=["#6366f1"])
            fig_nf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_nf, use_container_width=True)

        with col_oc:
            st.subheader("Operating Costs")
            fig_oc = px.bar(df_kpi, x="Month", y="Op Costs", color_discrete_sequence=["#ef4444"])
            fig_oc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig_oc, use_container_width=True)

        col_ex, col_cf = st.columns(2)

        with col_ex:
            st.subheader("Exception Trends")
            fig_ex = px.line(df_kpi, x="Month", y="Exceptions", markers=True, color_discrete_sequence=["#f59e0b"])
            fig_ex.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_ex, use_container_width=True)

        with col_cf:
            st.subheader("Daily Cash Flow (This Week)")
            fig_dcf = go.Figure()
            fig_dcf.add_trace(go.Bar(x=df_cash["Date"], y=df_cash["Deposits"], name="Deposits", marker_color="#22c55e"))
            fig_dcf.add_trace(go.Bar(x=df_cash["Date"], y=df_cash["Withdrawals"], name="Withdrawals", marker_color="#ef4444"))
            fig_dcf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0), barmode="group")
            st.plotly_chart(fig_dcf, use_container_width=True)

        st.subheader("Monthly KPI Summary")
        kpi_display = df_kpi.copy()
        kpi_display["Cost Ratio"] = (kpi_display["Op Costs"] / kpi_display["Net Flow"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            kpi_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Net Flow": st.column_config.NumberColumn(format="$%d"),
                "Op Costs": st.column_config.NumberColumn(format="$%d"),
            },
        )

    with tab_reports:
        st.subheader("Report Templates")

        reports = [
            {"name": "📅 Daily Finance Report", "desc": "End-of-day financial summary with all transaction activity and reconciliation status."},
            {"name": "📋 Weekly Control Report", "desc": "Weekly compliance and control metrics with exception analysis."},
            {"name": "📊 Monthly Financial Report", "desc": "Comprehensive monthly financial performance report with P&L analysis."},
        ]

        for r in reports:
            with st.container():
                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    st.markdown(f"**{r['name']}**")
                    st.caption(r["desc"])
                with rc2:
                    st.button("📥 Generate", key=f"gen_{r['name']}", use_container_width=True)
                st.markdown("---")

        st.subheader("Recent Reports")
        recent_reports = pd.DataFrame([
            {"Report": "Daily Finance Report", "Period": "May 17, 2026", "Generated": "May 18, 2026 01:00", "Status": "Ready"},
            {"Report": "Weekly Control Report", "Period": "May 12–18, 2026", "Generated": "May 18, 2026 06:00", "Status": "Ready"},
            {"Report": "Monthly Financial Report", "Period": "April 2026", "Generated": "May 01, 2026 08:00", "Status": "Ready"},
            {"Report": "Daily Finance Report", "Period": "May 18, 2026", "Generated": "—", "Status": "Scheduled 01:00"},
        ])
        st.dataframe(recent_reports, use_container_width=True, hide_index=True)
