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
    .arch-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 6px;
    }
    .arch-box h4 {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #e2e8f0;
    }
    .arch-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
        font-size: 13px;
        color: #94a3b8;
    }
    .arch-item .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .arch-arrow {
        text-align: center;
        font-size: 22px;
        color: #6366f1;
        margin: 4px 0;
        line-height: 1;
    }
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
    .status-online { background: #22c55e; }
    .status-warning { background: #f59e0b; }
    .status-offline { background: #ef4444; }
    .pipeline-metric {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
    }
    .pipeline-metric .value {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
    }
    .pipeline-metric .label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Mock Data
# ──────────────────────────────────────────────

transactions_data = [
    {"ID": "TXN-001", "Type": "Deposit", "Amount": 25000, "Currency": "USD", "Status": "Completed", "Source": "CRM", "Source System": "PSP-Alpha", "Client": "Acme Corp", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-18 08:23", "Description": "Client deposit via wire"},
    {"ID": "TXN-002", "Type": "Withdrawal", "Amount": 12500, "Currency": "USD", "Status": "Completed", "Source": "CRM", "Source System": "PSP-Beta", "Client": "Globe Ltd", "PSP": "Adyen", "Bank": "HSBC", "Timestamp": "2026-05-18 09:15", "Description": "Client withdrawal request"},
    {"ID": "TXN-003", "Type": "Deposit", "Amount": 8700, "Currency": "EUR", "Status": "Pending", "Source": "PSP Payment", "Source System": "PSP-Alpha", "Client": "NovaTech", "PSP": "Stripe", "Bank": "Deutsche", "Timestamp": "2026-05-18 09:42", "Description": "PSP payment pending confirmation"},
    {"ID": "TXN-004", "Type": "Transfer", "Amount": 50000, "Currency": "USD", "Status": "Completed", "Source": "Bank Transfer", "Source System": "Internal", "Client": "Internal", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 10:01", "Description": "Treasury bank transfer"},
    {"ID": "TXN-005", "Type": "Fee", "Amount": 125, "Currency": "USD", "Status": "Completed", "Source": "PSP Payment", "Source System": "PSP-Alpha", "Client": "Acme Corp", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-18 08:23", "Description": "Processing fee"},
    {"ID": "TXN-006", "Type": "Commission", "Amount": 375, "Currency": "USD", "Status": "Completed", "Source": "Commission", "Source System": "IB-System", "Client": "IB-Partner-01", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 10:30", "Description": "IB commission payout"},
    {"ID": "TXN-007", "Type": "Deposit", "Amount": 150000, "Currency": "USD", "Status": "Completed", "Source": "CRM", "Source System": "PSP-Gamma", "Client": "MegaFund", "PSP": "Worldpay", "Bank": "Barclays", "Timestamp": "2026-05-17 14:22", "Description": "Large client CRM deposit"},
    {"ID": "TXN-008", "Type": "Withdrawal", "Amount": 45000, "Currency": "GBP", "Status": "Failed", "Source": "CRM", "Source System": "PSP-Beta", "Client": "BritCo", "PSP": "Adyen", "Bank": "Barclays", "Timestamp": "2026-05-17 16:45", "Description": "Failed withdrawal - insufficient PSP balance"},
    {"ID": "TXN-009", "Type": "Deposit", "Amount": 32000, "Currency": "USD", "Status": "Completed", "Source": "PSP Payment", "Source System": "PSP-Alpha", "Client": "SolarInc", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-17 11:10", "Description": "PSP payment deposit"},
    {"ID": "TXN-010", "Type": "Withdrawal", "Amount": 18200, "Currency": "EUR", "Status": "Pending", "Source": "CRM", "Source System": "PSP-Beta", "Client": "EuroTrade", "PSP": "Adyen", "Bank": "Deutsche", "Timestamp": "2026-05-18 07:50", "Description": "Pending CRM withdrawal"},
    {"ID": "TXN-011", "Type": "Fee", "Amount": 89, "Currency": "USD", "Status": "Completed", "Source": "PSP Payment", "Source System": "PSP-Gamma", "Client": "MegaFund", "PSP": "Worldpay", "Bank": "Barclays", "Timestamp": "2026-05-17 14:22", "Description": "Processing fee"},
    {"ID": "TXN-012", "Type": "Deposit", "Amount": 5600, "Currency": "USD", "Status": "Reversed", "Source": "CRM", "Source System": "PSP-Alpha", "Client": "QuickPay", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-16 09:30", "Description": "Reversed CRM deposit - chargeback"},
    {"ID": "TXN-013", "Type": "Transfer", "Amount": 200000, "Currency": "USD", "Status": "Completed", "Source": "Bank Transfer", "Source System": "Internal", "Client": "Internal", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-16 15:00", "Description": "Liquidity rebalancing bank transfer"},
    {"ID": "TXN-014", "Type": "Commission", "Amount": 1250, "Currency": "USD", "Status": "Completed", "Source": "Commission", "Source System": "IB-System", "Client": "IB-Partner-02", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-16 16:00", "Description": "Monthly IB commission"},
    {"ID": "TXN-015", "Type": "Deposit", "Amount": 72000, "Currency": "USD", "Status": "Completed", "Source": "CRM", "Source System": "PSP-Alpha", "Client": "TradeCo", "PSP": "Stripe", "Bank": "Chase", "Timestamp": "2026-05-15 10:20", "Description": "Client CRM deposit"},
    {"ID": "TXN-016", "Type": "Deposit", "Amount": 15000, "Currency": "USD", "Status": "Completed", "Source": "Bonus", "Source System": "CRM", "Client": "Acme Corp", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 11:00", "Description": "Welcome bonus credit"},
    {"ID": "TXN-017", "Type": "Deposit", "Amount": 5000, "Currency": "USD", "Status": "Completed", "Source": "Bonus", "Source System": "CRM", "Client": "MegaFund", "PSP": "N/A", "Bank": "Barclays", "Timestamp": "2026-05-17 09:00", "Description": "Loyalty bonus credit"},
    {"ID": "TXN-018", "Type": "Deposit", "Amount": 42000, "Currency": "EUR", "Status": "Completed", "Source": "PSP Payment", "Source System": "PSP-Beta", "Client": "EuroTrade", "PSP": "Adyen", "Bank": "Deutsche", "Timestamp": "2026-05-18 06:30", "Description": "PSP payment settlement"},
    {"ID": "TXN-019", "Type": "Transfer", "Amount": 85000, "Currency": "USD", "Status": "Completed", "Source": "Bank Transfer", "Source System": "Internal", "Client": "Internal", "PSP": "N/A", "Bank": "HSBC", "Timestamp": "2026-05-18 08:00", "Description": "Inter-bank transfer"},
    {"ID": "TXN-020", "Type": "Commission", "Amount": 2100, "Currency": "USD", "Status": "Pending", "Source": "Commission", "Source System": "IB-System", "Client": "IB-Partner-03", "PSP": "N/A", "Bank": "Chase", "Timestamp": "2026-05-18 11:15", "Description": "Weekly IB commission accrual"},
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
            "🏗️ System Architecture",
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

    # ── Computed values ──
    opening_balance = 4250000
    cash_in_today = df_txn[(df_txn["Type"].isin(["Deposit"])) & (df_txn["Timestamp"].str.startswith("2026-05-18")) & (df_txn["Status"].isin(["Completed", "Pending"]))]["Amount"].sum()
    cash_out_today = df_txn[(df_txn["Type"].isin(["Withdrawal", "Commission", "Fee"])) & (df_txn["Timestamp"].str.startswith("2026-05-18")) & (df_txn["Status"].isin(["Completed", "Pending"]))]["Amount"].sum()
    net_flow = cash_in_today - cash_out_today

    total_bank = df_bank["Balance"].sum()
    total_psp = df_psp["Balance"].sum()
    available_cash = total_bank + total_psp
    pending_withdrawals = df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Pending")]["Amount"].sum() + df_psp["Pending Out"].sum()
    psp_total_balance = df_psp["Balance"].sum()

    unresolved_exceptions = len(df_rec[df_rec["Status"] == "Exception"])
    settlement_delays = len(df_alerts[(df_alerts["Title"].str.contains("Delay|Settlement", case=False)) & (df_alerts["Status"] != "Resolved")])
    liquidity_warnings = len(df_alerts[(df_alerts["Title"].str.contains("Liquidity|Buffer", case=False)) & (df_alerts["Status"] != "Resolved")])

    daily_volume = len(df_txn[df_txn["Timestamp"].str.startswith("2026-05-18")])
    total_rec = len(df_rec)
    matched_rec = len(df_rec[df_rec["Status"] == "Matched"])
    recon_rate = (matched_rec / total_rec * 100) if total_rec > 0 else 0
    unmatched_txns = len(df_rec[df_rec["Status"].isin(["Partial", "Exception"])])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINANCIAL OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("💵 Financial Overview")
    fo1, fo2, fo3, fo4 = st.columns(4)
    fo1.metric("Opening Balance", fmt(opening_balance))
    fo2.metric("Cash In Today", fmt(cash_in_today), f"+{cash_in_today/opening_balance*100:.1f}%")
    fo3.metric("Cash Out Today", fmt(cash_out_today), f"-{cash_out_today/opening_balance*100:.1f}%", delta_color="inverse")
    fo4.metric("Net Flow", fmt(net_flow), f"{'+'if net_flow>=0 else ''}{net_flow/opening_balance*100:.1f}%")

    st.markdown("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LIQUIDITY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("💧 Liquidity")
    lq1, lq2, lq3 = st.columns(3)
    lq1.metric("Available Cash", fmt(available_cash))
    lq2.metric("Pending Withdrawals", fmt(pending_withdrawals), delta_color="inverse")
    lq3.metric("PSP Balances", fmt(psp_total_balance))

    col_liq_chart, col_psp_chart = st.columns(2)

    with col_liq_chart:
        fig_liq = go.Figure()
        fig_liq.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["Available"], mode="lines+markers", name="Available Cash", fill="tozeroy", line=dict(color="#6366f1", width=2), fillcolor="rgba(99,102,241,0.12)"))
        fig_liq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_liq, use_container_width=True)

    with col_psp_chart:
        psp_summary = df_psp.groupby("PSP")["Balance"].sum().reset_index()
        fig_psp = px.bar(psp_summary, x="PSP", y="Balance", color="PSP", color_discrete_sequence=["#6366f1", "#22c55e", "#f59e0b"])
        fig_psp.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_psp, use_container_width=True)

    st.markdown("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ALERTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("🚨 Alerts")
    al1, al2, al3 = st.columns(3)
    al1.metric("Unresolved Exceptions", unresolved_exceptions, delta_color="inverse")
    al2.metric("Settlement Delays", settlement_delays, delta_color="inverse")
    al3.metric("Liquidity Warnings", liquidity_warnings, delta_color="inverse")

    active_alerts_df = df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])]
    for _, alert in active_alerts_df.iterrows():
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

    st.markdown("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # KPIs
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("📈 KPIs")
    kp1, kp2, kp3 = st.columns(3)
    kp1.metric("Daily Volume", f"{daily_volume} txns", f"Today's transactions")
    kp2.metric("Reconciliation Rate", f"{recon_rate:.1f}%", f"{matched_rec}/{total_rec} matched")
    kp3.metric("Unmatched Transactions", unmatched_txns, delta_color="inverse")

    col_flow, col_mix = st.columns([2, 1])

    with col_flow:
        st.markdown("**Cash Flow (7-Day)**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Deposits"], mode="lines", name="Deposits", fill="tozeroy", line=dict(color="#22c55e", width=2), fillcolor="rgba(34,197,94,0.15)"))
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Withdrawals"], mode="lines", name="Withdrawals", fill="tozeroy", line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,0.15)"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col_mix:
        st.markdown("**Transaction Mix**")
        type_counts = df_txn["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig_pie = px.pie(type_counts, values="Count", names="Type", hole=0.55, color_discrete_sequence=["#6366f1", "#22c55e", "#ef4444", "#f59e0b", "#a855f7"])
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=True, legend=dict(font=dict(size=11)))
        fig_pie.update_traces(textinfo="none")
        st.plotly_chart(fig_pie, use_container_width=True)

    col_net, col_buf = st.columns(2)

    with col_net:
        st.markdown("**Net Flow (7-Day)**")
        fig_net = px.bar(df_cash, x="Date", y="Net Flow", color_discrete_sequence=["#6366f1"])
        fig_net.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_net, use_container_width=True)

    with col_buf:
        st.markdown("**Liquidity Buffer Trend**")
        fig_buf = go.Figure()
        fig_buf.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["Buffer %"], mode="lines+markers", line=dict(color="#f59e0b", width=2.5), fill="tozeroy", fillcolor="rgba(245,158,11,0.12)"))
        fig_buf.add_hline(y=15, line_dash="dash", line_color="#ef4444", annotation_text="15% Threshold")
        fig_buf.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=240, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_buf, use_container_width=True)


# ══════════════════════════════════════════════
# SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════
elif page == "🏗️ System Architecture":
    st.title("🏗️ System Architecture")
    st.caption("End-to-end data flow pipeline — from connected systems to end users")

    # ── Pipeline health metrics ──
    st.subheader("Pipeline Health")
    ph1, ph2, ph3, ph4, ph5, ph6 = st.columns(6)
    ph1.markdown('<div class="pipeline-metric"><div class="value">6</div><div class="label">Connected Systems</div></div>', unsafe_allow_html=True)
    ph2.markdown('<div class="pipeline-metric"><div class="value">4</div><div class="label">API Connectors</div></div>', unsafe_allow_html=True)
    ph3.markdown(f'<div class="pipeline-metric"><div class="value">{len(df_txn)}</div><div class="label">Txns Processed</div></div>', unsafe_allow_html=True)
    matched_count = len(df_rec[df_rec["Status"] == "Matched"])
    ph4.markdown(f'<div class="pipeline-metric"><div class="value">{matched_count}/{len(df_rec)}</div><div class="label">Reconciled</div></div>', unsafe_allow_html=True)
    ph5.markdown(f'<div class="pipeline-metric"><div class="value">{len(df_ledger)}</div><div class="label">Ledger Entries</div></div>', unsafe_allow_html=True)
    active_al = len(df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])])
    ph6.markdown(f'<div class="pipeline-metric"><div class="value">{active_al}</div><div class="label">Active Alerts</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. CONNECTED SYSTEMS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("1️⃣ Connected Systems")
    st.caption("External platforms feeding data into the pipeline")

    systems = [
        {"name": "CRM", "desc": "Client deposits, withdrawals, account data", "status": "online", "last_sync": "2 min ago", "records": "8,420"},
        {"name": "PSP Gateways", "desc": "Stripe, Adyen, Worldpay payment processing", "status": "online", "last_sync": "30 sec ago", "records": "12,105"},
        {"name": "Bank Accounts", "desc": "Chase, HSBC, Deutsche Bank, Barclays", "status": "online", "last_sync": "5 min ago", "records": "3,218"},
        {"name": "Trading Platform", "desc": "Client trading activity and P&L data", "status": "online", "last_sync": "1 min ago", "records": "45,890"},
        {"name": "Bonus Engine", "desc": "Welcome, loyalty, referral bonus credits", "status": "warning", "last_sync": "18 min ago", "records": "1,240"},
        {"name": "Commission Engine", "desc": "IB commission calculations and payouts", "status": "online", "last_sync": "10 min ago", "records": "892"},
    ]

    sys_cols = st.columns(3)
    for i, sys in enumerate(systems):
        col = sys_cols[i % 3]
        dot_class = f"status-{sys['status']}"
        status_label = "Online" if sys["status"] == "online" else "Delayed" if sys["status"] == "warning" else "Offline"
        with col:
            st.markdown(f"""
            <div class="arch-box">
                <h4><span class="status-dot {dot_class}"></span> {sys["name"]}</h4>
                <div class="arch-item">{sys["desc"]}</div>
                <div style="display:flex; justify-content:space-between; margin-top:8px; font-size:11px; color:#64748b;">
                    <span>Last sync: {sys["last_sync"]}</span>
                    <span>{sys["records"]} records</span>
                </div>
                <div style="margin-top:6px">{status_badge(status_label.replace("Delayed", "Investigating").replace("Online", "Completed").replace("Offline", "Failed"))}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. DATA INTEGRATION LAYER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("2️⃣ Data Integration Layer")
    st.caption("Ingestion, validation, and routing of incoming data")

    int_cols = st.columns(4)

    integration_items = [
        {"title": "API Connectors", "icon": "🔌", "details": ["REST APIs (CRM, PSP)", "WebSocket (Trading)", "SFTP (Bank files)", "OAuth 2.0 Auth"], "status": "4/4 Active", "color": "#22c55e"},
        {"title": "File Imports", "icon": "📁", "details": ["Bank statements (MT940)", "PSP settlement CSV", "Commission reports", "Bonus batch files"], "status": "Last: 06:00", "color": "#3b82f6"},
        {"title": "Scheduled Data Pull", "icon": "⏰", "details": ["Real-time: PSP, CRM", "Every 5 min: Trading", "Every 15 min: Banks", "Hourly: Commissions"], "status": "On schedule", "color": "#22c55e"},
        {"title": "Data Validation", "icon": "✅", "details": ["Schema validation", "Amount range checks", "Duplicate detection", "Referential integrity"], "status": "99.8% pass rate", "color": "#22c55e"},
    ]

    for idx, item in enumerate(integration_items):
        with int_cols[idx]:
            items_html = "".join([f'<div class="arch-item"><div class="dot" style="background:{item["color"]}"></div>{d}</div>' for d in item["details"]])
            st.markdown(f"""
            <div class="arch-box">
                <h4>{item["icon"]} {item["title"]}</h4>
                {items_html}
                <div style="margin-top:8px; font-size:11px; color:{item['color']}">{item["status"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. DATA NORMALIZATION LAYER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("3️⃣ Data Normalization Layer")
    st.caption("Standardizing data formats across all sources")

    norm_cols = st.columns(4)

    norm_items = [
        {"title": "Currency Standardization", "icon": "💱", "details": ["USD base currency", "Real-time FX rates", "Multi-currency support", "EUR, GBP, USD, CHF"], "metric": "3 currencies"},
        {"title": "Time Standardization", "icon": "🕐", "details": ["UTC normalization", "Timezone conversion", "Settlement date calc", "T+0 / T+1 handling"], "metric": "UTC aligned"},
        {"title": "ID Mapping", "icon": "🔗", "details": ["Cross-system ID link", "Client ID unification", "Transaction ref mapping", "PSP → Internal ID"], "metric": "100% mapped"},
        {"title": "Transaction Type Mapping", "icon": "🏷️", "details": ["Deposit classification", "Withdrawal mapping", "Fee categorization", "Commission tagging"], "metric": "6 types"},
    ]

    for idx, item in enumerate(norm_items):
        with norm_cols[idx]:
            items_html = "".join([f'<div class="arch-item"><div class="dot" style="background:#6366f1"></div>{d}</div>' for d in item["details"]])
            st.markdown(f"""
            <div class="arch-box">
                <h4>{item["icon"]} {item["title"]}</h4>
                {items_html}
                <div style="margin-top:8px; font-size:11px; color:#6366f1">{item["metric"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. RECONCILIATION ENGINE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("4️⃣ Reconciliation Engine")
    st.caption("Automated matching and exception detection")

    re_matched = len(df_rec[df_rec["Status"] == "Matched"])
    re_partial = len(df_rec[df_rec["Status"] == "Partial"])
    re_exception = len(df_rec[df_rec["Status"] == "Exception"])
    re_rate = (re_matched / len(df_rec) * 100) if len(df_rec) > 0 else 0

    re1, re2, re3, re4 = st.columns(4)
    re1.metric("ID Match", f"{re_matched} matched", f"{re_rate:.0f}% rate")
    re2.metric("Amount Match", f"{re_partial} partial", "FX variance check")
    re3.metric("Time Window Match", "±2 hours", "Settlement tolerance")
    re4.metric("Exception Detection", f"{re_exception} found", delta_color="inverse")

    recon_col1, recon_col2 = st.columns(2)
    with recon_col1:
        rec_summary = pd.DataFrame({"Status": ["Matched", "Partial", "Exception"], "Count": [re_matched, re_partial, re_exception]})
        fig_rec = px.pie(rec_summary, values="Count", names="Status", hole=0.55, color_discrete_map={"Matched": "#22c55e", "Partial": "#f59e0b", "Exception": "#ef4444"})
        fig_rec.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=250, margin=dict(l=0, r=0, t=10, b=0))
        fig_rec.update_traces(textinfo="value+percent")
        st.plotly_chart(fig_rec, use_container_width=True)
    with recon_col2:
        st.dataframe(
            df_rec[["Case ID", "Transaction", "Status", "Match Score", "Case Status"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Match Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")},
        )

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. CENTRAL FINANCIAL LEDGER
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("5️⃣ Central Financial Ledger")
    st.caption("Double-entry bookkeeping across all account categories")

    ledger_categories = [
        {"name": "Client Funds", "icon": "👤", "debits": df_ledger[(df_ledger["Category"] == "Client Liability") & (df_ledger["Type"] == "Debit")]["Amount"].sum(), "credits": df_ledger[(df_ledger["Category"] == "Client Liability") & (df_ledger["Type"] == "Credit")]["Amount"].sum()},
        {"name": "Company Cash", "icon": "🏢", "debits": df_ledger[(df_ledger["Category"] == "Company Cash") & (df_ledger["Type"] == "Debit")]["Amount"].sum(), "credits": df_ledger[(df_ledger["Category"] == "Company Cash") & (df_ledger["Type"] == "Credit")]["Amount"].sum()},
        {"name": "Fees", "icon": "💳", "debits": df_ledger[(df_ledger["Category"] == "Fee Accounts") & (df_ledger["Type"] == "Debit")]["Amount"].sum(), "credits": df_ledger[(df_ledger["Category"] == "Fee Accounts") & (df_ledger["Type"] == "Credit")]["Amount"].sum()},
        {"name": "Commissions", "icon": "🤝", "debits": df_ledger[(df_ledger["Category"] == "Commission Accounts") & (df_ledger["Type"] == "Debit")]["Amount"].sum(), "credits": df_ledger[(df_ledger["Category"] == "Commission Accounts") & (df_ledger["Type"] == "Credit")]["Amount"].sum()},
        {"name": "Adjustments", "icon": "🔧", "debits": 0, "credits": 0},
    ]

    led_cols = st.columns(5)
    for idx, cat in enumerate(ledger_categories):
        net = cat["debits"] - cat["credits"]
        with led_cols[idx]:
            st.markdown(f"""
            <div class="arch-box" style="text-align:center">
                <h4>{cat["icon"]} {cat["name"]}</h4>
                <div style="font-size:11px; color:#64748b; margin-bottom:4px">DR: {fmt(cat["debits"])}</div>
                <div style="font-size:11px; color:#64748b; margin-bottom:4px">CR: {fmt(cat["credits"])}</div>
                <div style="font-size:16px; font-weight:700; color:{'#22c55e' if net >= 0 else '#ef4444'}">{fmt(abs(net))}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. LIQUIDITY ENGINE + ALERT ENGINE (side by side)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    eng_left, eng_right = st.columns(2)

    with eng_left:
        st.subheader("6a️⃣ Liquidity Engine")
        total_bank = df_bank["Balance"].sum()
        total_psp = df_psp["Balance"].sum()
        avail = total_bank + total_psp
        pend_wd = df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Pending")]["Amount"].sum() + df_psp["Pending Out"].sum()
        psp_hold = df_psp["Pending Out"].sum()
        net_liq = avail - pend_wd - 125000 - 48500

        liq_items = [
            {"label": "Available Cash", "value": fmt(avail), "color": "#22c55e"},
            {"label": "Pending Withdrawals", "value": fmt(pend_wd), "color": "#ef4444"},
            {"label": "PSP Holdbacks", "value": fmt(psp_hold), "color": "#f59e0b"},
            {"label": "Net Liquidity", "value": fmt(net_liq), "color": "#6366f1"},
        ]

        for item in liq_items:
            st.markdown(f"""
            <div class="arch-item" style="justify-content:space-between; padding:8px 12px; background:#0f172a; border-radius:6px; margin-bottom:4px;">
                <span>{item["label"]}</span>
                <span style="font-weight:700; color:{item["color"]}">{item["value"]}</span>
            </div>
            """, unsafe_allow_html=True)

    with eng_right:
        st.subheader("6b️⃣ Alert & Exception Engine")
        alert_types = [
            {"label": "Reconciliation Errors", "count": len(df_rec[df_rec["Status"] == "Exception"]), "color": "#ef4444"},
            {"label": "PSP Delays", "count": len(df_alerts[df_alerts["Title"].str.contains("PSP|Delay", case=False)]), "color": "#f59e0b"},
            {"label": "Cash Imbalance", "count": len(df_alerts[df_alerts["Title"].str.contains("Cash|Imbalance", case=False)]), "color": "#ef4444"},
            {"label": "Threshold Breach", "count": len(df_alerts[df_alerts["Title"].str.contains("Liquidity|Buffer|Spike", case=False)]), "color": "#f59e0b"},
        ]

        for item in alert_types:
            st.markdown(f"""
            <div class="arch-item" style="justify-content:space-between; padding:8px 12px; background:#0f172a; border-radius:6px; margin-bottom:4px;">
                <span>{item["label"]}</span>
                <span style="font-weight:700; color:{item["color"]}">{item["count"]} active</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 7. REPORTING & ANALYTICS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("7️⃣ Reporting & Analytics")
    st.caption("Output layer — dashboards, reports, and monitoring")

    report_items = [
        {"name": "Dashboard", "icon": "📊", "freq": "Real-time", "status": "Live"},
        {"name": "Daily Report", "icon": "📅", "freq": "Daily @ 01:00", "status": "Scheduled"},
        {"name": "Weekly Review", "icon": "📋", "freq": "Monday @ 06:00", "status": "Scheduled"},
        {"name": "Monthly Close", "icon": "📊", "freq": "1st of month", "status": "Scheduled"},
        {"name": "Profitability Analysis", "icon": "💰", "freq": "On demand", "status": "Available"},
        {"name": "KPI Monitoring", "icon": "📈", "freq": "Real-time", "status": "Live"},
    ]

    rpt_cols = st.columns(6)
    for idx, rpt in enumerate(report_items):
        with rpt_cols[idx]:
            live_color = "#22c55e" if rpt["status"] == "Live" else "#3b82f6" if rpt["status"] == "Scheduled" else "#6366f1"
            st.markdown(f"""
            <div class="arch-box" style="text-align:center; min-height:120px">
                <div style="font-size:24px; margin-bottom:4px">{rpt["icon"]}</div>
                <h4 style="font-size:12px">{rpt["name"]}</h4>
                <div style="font-size:10px; color:#64748b">{rpt["freq"]}</div>
                <div style="margin-top:6px; font-size:11px; color:{live_color}; font-weight:600">{rpt["status"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">▼</div>', unsafe_allow_html=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 8. END USERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.subheader("8️⃣ End Users")
    st.caption("Teams and roles consuming platform outputs")

    users = [
        {"role": "CFO", "icon": "👔", "access": "Full dashboard, reports, KPIs", "level": "Executive"},
        {"role": "Finance Manager", "icon": "📊", "access": "Ledger, liquidity, reconciliation", "level": "Management"},
        {"role": "Operations Team", "icon": "⚙️", "access": "Transactions, PSP monitoring", "level": "Operational"},
        {"role": "Reconciliation Team", "icon": "🔄", "access": "Matching, exceptions, cases", "level": "Operational"},
        {"role": "Management", "icon": "🏢", "access": "Reports, analytics, profitability", "level": "Executive"},
    ]

    user_cols = st.columns(5)
    for idx, user in enumerate(users):
        with user_cols[idx]:
            level_color = "#6366f1" if user["level"] == "Executive" else "#22c55e"
            st.markdown(f"""
            <div class="arch-box" style="text-align:center">
                <div style="font-size:28px; margin-bottom:4px">{user["icon"]}</div>
                <h4 style="font-size:13px">{user["role"]}</h4>
                <div style="font-size:11px; color:#94a3b8; margin-bottom:6px">{user["access"]}</div>
                <span class="badge" style="background:{'rgba(99,102,241,0.15)' if user['level']=='Executive' else 'rgba(34,197,94,0.15)'}; color:{level_color}">{user["level"]}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 2. TRANSACTIONS
# ══════════════════════════════════════════════
elif page == "💸 Transactions":
    st.title("💸 Transactions")
    st.caption("All Transactions — CRM Deposits, Withdrawals, PSP Payments, Bank Transfers, Commissions & Bonuses")

    # Source summary KPIs
    src_crm = len(df_txn[df_txn["Source"] == "CRM"])
    src_psp = len(df_txn[df_txn["Source"] == "PSP Payment"])
    src_bank = len(df_txn[df_txn["Source"] == "Bank Transfer"])
    src_comm = len(df_txn[df_txn["Source"] == "Commission"])
    src_bonus = len(df_txn[df_txn["Source"] == "Bonus"])

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    sc1.metric("CRM", src_crm, "deposits & withdrawals")
    sc2.metric("PSP Payments", src_psp, "payment settlements")
    sc3.metric("Bank Transfers", src_bank, "inter-bank")
    sc4.metric("Commissions", src_comm, "IB payouts")
    sc5.metric("Bonuses", src_bonus, "bonus credits")

    # Source tabs
    src_tab_all, src_tab_crm, src_tab_psp, src_tab_bank, src_tab_comm, src_tab_bonus = st.tabs(
        ["All Sources", "CRM", "PSP Payments", "Bank Transfers", "Commissions", "Bonuses"]
    )

    def get_source_df(source_name):
        if source_name == "All":
            return df_txn
        return df_txn[df_txn["Source"] == source_name]

    def render_txn_tab(base_df, tab_key):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            search = st.text_input("🔍 Search", placeholder="ID, client...", key=f"search_{tab_key}")
        with col2:
            type_filter = st.selectbox("Type", ["All"] + sorted(base_df["Type"].unique().tolist()), key=f"type_{tab_key}")
        with col3:
            status_filter = st.selectbox("Status", ["All"] + sorted(base_df["Status"].unique().tolist()), key=f"status_{tab_key}")
        with col4:
            psp_filter = st.selectbox("PSP", ["All"] + sorted(base_df["PSP"].unique().tolist()), key=f"psp_{tab_key}")
        with col5:
            bank_filter = st.selectbox("Bank", ["All"] + sorted(base_df["Bank"].unique().tolist()), key=f"bank_{tab_key}")

        filtered = base_df.copy()
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

        st.caption(f"Showing {len(filtered)} of {len(base_df)} transactions")
        st.dataframe(
            filtered[["ID", "Type", "Source", "Client", "Amount", "Currency", "PSP", "Bank", "Status", "Timestamp"]],
            use_container_width=True,
            hide_index=True,
            column_config={"Amount": st.column_config.NumberColumn(format="$%d")},
        )
        return filtered

    with src_tab_all:
        filtered_all = render_txn_tab(df_txn, "all")
    with src_tab_crm:
        render_txn_tab(get_source_df("CRM"), "crm")
    with src_tab_psp:
        render_txn_tab(get_source_df("PSP Payment"), "psp")
    with src_tab_bank:
        render_txn_tab(get_source_df("Bank Transfer"), "bank")
    with src_tab_comm:
        render_txn_tab(get_source_df("Commission"), "comm")
    with src_tab_bonus:
        render_txn_tab(get_source_df("Bonus"), "bonus")

    st.markdown("---")

    # Source breakdown chart
    col_src_chart, col_src_vol = st.columns(2)
    with col_src_chart:
        st.subheader("Transactions by Source")
        src_counts = df_txn["Source"].value_counts().reset_index()
        src_counts.columns = ["Source", "Count"]
        fig_src = px.pie(src_counts, values="Count", names="Source", hole=0.5, color_discrete_sequence=["#6366f1", "#22c55e", "#3b82f6", "#a855f7", "#f59e0b"])
        fig_src.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0))
        fig_src.update_traces(textinfo="none")
        st.plotly_chart(fig_src, use_container_width=True)

    with col_src_vol:
        st.subheader("Volume by Source")
        src_vol = df_txn.groupby("Source")["Amount"].sum().reset_index().sort_values("Amount", ascending=True)
        fig_vol = px.bar(src_vol, y="Source", x="Amount", orientation="h", color="Source", color_discrete_sequence=["#6366f1", "#22c55e", "#3b82f6", "#a855f7", "#f59e0b"])
        fig_vol.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown("---")
    st.subheader("Transaction Details")
    selected_txn = st.selectbox("Select transaction to view details", df_txn["ID"].tolist())
    if selected_txn:
        tx = df_txn[df_txn["ID"] == selected_txn].iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**ID:** {tx['ID']}")
            st.markdown(f"**Type:** {tx['Type']}")
            st.markdown(f"**Source:** {tx['Source']}")
            st.markdown(f"**Amount:** {fmt(tx['Amount'])} {tx['Currency']}")
        with c2:
            st.markdown(f"**Status:** {tx['Status']}")
            st.markdown(f"**Client:** {tx['Client']}")
            st.markdown(f"**Source System:** {tx['Source System']}")
            st.markdown(f"**PSP:** {tx['PSP']}")
        with c3:
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
