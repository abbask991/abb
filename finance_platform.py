"""
FinanceOps — Finance Automation Platform v2
Standards: IFRS, ISO 20022, ISO 4217, Basel III LCR/NSFR, SOX Compliance
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="FinanceOps", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
.main .block-container{padding-top:1.2rem}
div[data-testid="stMetric"]{background:#0f1729;border:1px solid #1e2d4a;border-radius:10px;padding:14px 18px}
div[data-testid="stMetric"] label{font-size:11px !important;text-transform:uppercase;letter-spacing:.5px}
.kpi-strip{display:flex;gap:8px;margin-bottom:12px}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}
.bg-green{background:rgba(16,185,129,.12);color:#10b981}.bg-red{background:rgba(239,68,68,.12);color:#ef4444}
.bg-yellow{background:rgba(245,158,11,.12);color:#f59e0b}.bg-blue{background:rgba(59,130,246,.12);color:#3b82f6}
.bg-purple{background:rgba(139,92,246,.12);color:#8b5cf6}.bg-cyan{background:rgba(6,182,212,.12);color:#06b6d4}
.bg-gray{background:rgba(148,163,184,.12);color:#94a3b8}
.card{background:#0f1729;border:1px solid #1e2d4a;border-radius:10px;padding:18px 20px;margin-bottom:8px}
.card h4{font-size:14px;font-weight:700;margin-bottom:8px;color:#e2e8f0}
.card-sm{font-size:12px;color:#94a3b8}
.flow-arrow{text-align:center;font-size:20px;color:#6366f1;margin:4px 0;line-height:1}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot-green{background:#10b981}.dot-yellow{background:#f59e0b}.dot-red{background:#ef4444}
.alert-card{background:#0f1729;border:1px solid #1e2d4a;border-radius:8px;padding:14px 18px;margin-bottom:8px}
.alert-title{font-weight:600;font-size:13px;margin-bottom:3px}
.alert-desc{font-size:12px;color:#94a3b8;line-height:1.5}
.metric-box{background:#0f1729;border:1px solid #1e2d4a;border-radius:8px;padding:12px 16px;text-align:center}
.metric-box .val{font-size:20px;font-weight:700;color:#e2e8f0}
.metric-box .lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
div[data-testid="stSidebar"]>div:first-child{padding-top:1rem}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ISO 4217 — CURRENCY DEFINITIONS
# ══════════════════════════════════════════════════════════════
ISO_CCY = {
    "USD": {"name": "US Dollar", "symbol": "$", "code": 840, "minor": 2},
    "EUR": {"name": "Euro", "symbol": "€", "code": 978, "minor": 2},
    "GBP": {"name": "Pound Sterling", "symbol": "£", "code": 826, "minor": 2},
    "CHF": {"name": "Swiss Franc", "symbol": "CHF", "code": 756, "minor": 2},
    "JPY": {"name": "Japanese Yen", "symbol": "¥", "code": 392, "minor": 0},
}

FX_RATES = {"EUR/USD": 1.0845, "GBP/USD": 1.2630, "CHF/USD": 1.1175, "JPY/USD": 0.00645, "USD/USD": 1.0}

def to_usd(amount, ccy):
    if ccy == "USD": return amount
    key = f"{ccy}/USD"
    return amount * FX_RATES.get(key, 1.0)

def fmt(n, ccy="USD"):
    sym = ISO_CCY.get(ccy, {}).get("symbol", "$")
    minor = ISO_CCY.get(ccy, {}).get("minor", 2)
    if minor == 0:
        return f"{sym}{n:,.0f}"
    return f"{sym}{n:,.{minor}f}"

def fmt_usd(n):
    return f"${n:,.0f}"

# ══════════════════════════════════════════════════════════════
# IFRS-ALIGNED CHART OF ACCOUNTS (IAS 1)
# ══════════════════════════════════════════════════════════════
CHART_OF_ACCOUNTS = {
    "1000": {"name": "Cash & Cash Equivalents", "class": "Asset", "ifrs": "IAS 7", "normal": "Debit"},
    "1010": {"name": "Bank Accounts — Operating", "class": "Asset", "ifrs": "IAS 7", "normal": "Debit"},
    "1020": {"name": "Bank Accounts — Reserve", "class": "Asset", "ifrs": "IAS 7", "normal": "Debit"},
    "1100": {"name": "PSP Settlement Receivables", "class": "Asset", "ifrs": "IFRS 9", "normal": "Debit"},
    "1200": {"name": "Trade Receivables", "class": "Asset", "ifrs": "IFRS 15", "normal": "Debit"},
    "2000": {"name": "Client Funds Payable", "class": "Liability", "ifrs": "IAS 37", "normal": "Credit"},
    "2100": {"name": "Commission Payable", "class": "Liability", "ifrs": "IFRS 15", "normal": "Credit"},
    "2200": {"name": "Bonus Liability", "class": "Liability", "ifrs": "IAS 19", "normal": "Credit"},
    "2300": {"name": "Accrued Expenses", "class": "Liability", "ifrs": "IAS 37", "normal": "Credit"},
    "3000": {"name": "Retained Earnings", "class": "Equity", "ifrs": "IAS 1", "normal": "Credit"},
    "4000": {"name": "Transaction Fee Revenue", "class": "Revenue", "ifrs": "IFRS 15", "normal": "Credit"},
    "4100": {"name": "Spread / Markup Revenue", "class": "Revenue", "ifrs": "IFRS 15", "normal": "Credit"},
    "4200": {"name": "FX Conversion Revenue", "class": "Revenue", "ifrs": "IFRS 9", "normal": "Credit"},
    "5000": {"name": "PSP Processing Costs", "class": "Expense", "ifrs": "IAS 1", "normal": "Debit"},
    "5100": {"name": "Bank Charges", "class": "Expense", "ifrs": "IAS 1", "normal": "Debit"},
    "5200": {"name": "Commission Expense", "class": "Expense", "ifrs": "IFRS 15", "normal": "Debit"},
    "5300": {"name": "Bonus & Promotion Expense", "class": "Expense", "ifrs": "IAS 19", "normal": "Debit"},
    "5400": {"name": "Operational Expenses", "class": "Expense", "ifrs": "IAS 1", "normal": "Debit"},
}

# ══════════════════════════════════════════════════════════════
# ISO 20022 — MESSAGE TYPE MAPPING
# ══════════════════════════════════════════════════════════════
ISO20022_MSG = {
    "pacs.008": "FI to FI Customer Credit Transfer",
    "pacs.003": "FI to FI Customer Direct Debit",
    "pacs.004": "Payment Return",
    "pacs.002": "FI to FI Payment Status Report",
    "camt.053": "Bank to Customer Statement",
    "camt.054": "Bank to Customer Debit/Credit Notification",
    "pain.001": "Customer Credit Transfer Initiation",
    "pain.002": "Customer Payment Status Report",
}

# ══════════════════════════════════════════════════════════════
# MOCK DATA — TRANSACTIONS (ISO 20022 aligned)
# ══════════════════════════════════════════════════════════════
transactions_data = [
    {"ID": "TXN-20260518-001", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "CRM", "Amount": 25000.00, "Currency": "USD", "Status": "Settled", "Client": "Acme Corp", "Counterparty": "Stripe (PSP)", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 08:23:14", "Ref": "CRED-2026-0518-001", "Description": "Client deposit — wire transfer"},
    {"ID": "TXN-20260518-002", "ISO_Msg": "pacs.003", "Type": "Withdrawal", "Source": "CRM", "Amount": 12500.00, "Currency": "USD", "Status": "Settled", "Client": "Globe Ltd", "Counterparty": "Adyen (PSP)", "Bank": "HSBC", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 09:15:42", "Ref": "DBIT-2026-0518-001", "Description": "Client withdrawal request"},
    {"ID": "TXN-20260518-003", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "PSP", "Amount": 8700.00, "Currency": "EUR", "Status": "Pending", "Client": "NovaTech", "Counterparty": "Stripe (PSP)", "Bank": "Deutsche Bank", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-19", "Timestamp": "2026-05-18 09:42:08", "Ref": "CRED-2026-0518-002", "Description": "PSP settlement — T+1"},
    {"ID": "TXN-20260518-004", "ISO_Msg": "pacs.008", "Type": "Transfer", "Source": "Bank", "Amount": 50000.00, "Currency": "USD", "Status": "Settled", "Client": "Internal", "Counterparty": "Treasury", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 10:01:33", "Ref": "TRFR-2026-0518-001", "Description": "Internal treasury rebalancing"},
    {"ID": "TXN-20260518-005", "ISO_Msg": "camt.054", "Type": "Fee", "Source": "PSP", "Amount": 125.00, "Currency": "USD", "Status": "Settled", "Client": "Acme Corp", "Counterparty": "Stripe (PSP)", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 08:23:15", "Ref": "FEE-2026-0518-001", "Description": "PSP processing fee — 0.5%"},
    {"ID": "TXN-20260518-006", "ISO_Msg": "pain.001", "Type": "Commission", "Source": "Commission", "Amount": 375.00, "Currency": "USD", "Status": "Settled", "Client": "IB-Partner-Alpha", "Counterparty": "IB Commission Engine", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 10:30:00", "Ref": "COMM-2026-0518-001", "Description": "IB commission — volume-based"},
    {"ID": "TXN-20260517-007", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "CRM", "Amount": 150000.00, "Currency": "USD", "Status": "Settled", "Client": "MegaFund", "Counterparty": "Worldpay (PSP)", "Bank": "Barclays", "Value_Date": "2026-05-17", "Settlement_Date": "2026-05-17", "Timestamp": "2026-05-17 14:22:11", "Ref": "CRED-2026-0517-001", "Description": "High-value client deposit"},
    {"ID": "TXN-20260517-008", "ISO_Msg": "pacs.004", "Type": "Withdrawal", "Source": "CRM", "Amount": 45000.00, "Currency": "GBP", "Status": "Failed", "Client": "BritCo", "Counterparty": "Adyen (PSP)", "Bank": "Barclays", "Value_Date": "2026-05-17", "Settlement_Date": "", "Timestamp": "2026-05-17 16:45:22", "Ref": "DBIT-2026-0517-001", "Description": "Failed — insufficient PSP balance (pacs.004 return)"},
    {"ID": "TXN-20260517-009", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "PSP", "Amount": 32000.00, "Currency": "USD", "Status": "Settled", "Client": "SolarInc", "Counterparty": "Stripe (PSP)", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-17", "Settlement_Date": "2026-05-17", "Timestamp": "2026-05-17 11:10:55", "Ref": "CRED-2026-0517-002", "Description": "PSP settlement batch"},
    {"ID": "TXN-20260518-010", "ISO_Msg": "pacs.003", "Type": "Withdrawal", "Source": "CRM", "Amount": 18200.00, "Currency": "EUR", "Status": "Pending", "Client": "EuroTrade", "Counterparty": "Adyen (PSP)", "Bank": "Deutsche Bank", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-19", "Timestamp": "2026-05-18 07:50:19", "Ref": "DBIT-2026-0518-002", "Description": "Pending withdrawal — T+1"},
    {"ID": "TXN-20260517-011", "ISO_Msg": "camt.054", "Type": "Fee", "Source": "PSP", "Amount": 89.00, "Currency": "USD", "Status": "Settled", "Client": "MegaFund", "Counterparty": "Worldpay (PSP)", "Bank": "Barclays", "Value_Date": "2026-05-17", "Settlement_Date": "2026-05-17", "Timestamp": "2026-05-17 14:22:12", "Ref": "FEE-2026-0517-001", "Description": "PSP processing fee"},
    {"ID": "TXN-20260516-012", "ISO_Msg": "pacs.004", "Type": "Deposit", "Source": "CRM", "Amount": 5600.00, "Currency": "USD", "Status": "Reversed", "Client": "QuickPay", "Counterparty": "Stripe (PSP)", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-16", "Settlement_Date": "2026-05-16", "Timestamp": "2026-05-16 09:30:44", "Ref": "CRED-2026-0516-001", "Description": "Chargeback reversal (pacs.004)"},
    {"ID": "TXN-20260516-013", "ISO_Msg": "pacs.008", "Type": "Transfer", "Source": "Bank", "Amount": 200000.00, "Currency": "USD", "Status": "Settled", "Client": "Internal", "Counterparty": "Treasury", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-16", "Settlement_Date": "2026-05-16", "Timestamp": "2026-05-16 15:00:00", "Ref": "TRFR-2026-0516-001", "Description": "Liquidity rebalancing"},
    {"ID": "TXN-20260516-014", "ISO_Msg": "pain.001", "Type": "Commission", "Source": "Commission", "Amount": 1250.00, "Currency": "USD", "Status": "Settled", "Client": "IB-Partner-Beta", "Counterparty": "IB Commission Engine", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-16", "Settlement_Date": "2026-05-16", "Timestamp": "2026-05-16 16:00:00", "Ref": "COMM-2026-0516-001", "Description": "Monthly IB commission"},
    {"ID": "TXN-20260515-015", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "CRM", "Amount": 72000.00, "Currency": "USD", "Status": "Settled", "Client": "TradeCo", "Counterparty": "Stripe (PSP)", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-15", "Settlement_Date": "2026-05-15", "Timestamp": "2026-05-15 10:20:00", "Ref": "CRED-2026-0515-001", "Description": "Client deposit"},
    {"ID": "TXN-20260518-016", "ISO_Msg": "camt.054", "Type": "Bonus", "Source": "Bonus", "Amount": 15000.00, "Currency": "USD", "Status": "Settled", "Client": "Acme Corp", "Counterparty": "Bonus Engine", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 11:00:00", "Ref": "BNS-2026-0518-001", "Description": "Welcome bonus — IAS 19 provision"},
    {"ID": "TXN-20260517-017", "ISO_Msg": "camt.054", "Type": "Bonus", "Source": "Bonus", "Amount": 5000.00, "Currency": "USD", "Status": "Settled", "Client": "MegaFund", "Counterparty": "Bonus Engine", "Bank": "Barclays", "Value_Date": "2026-05-17", "Settlement_Date": "2026-05-17", "Timestamp": "2026-05-17 09:00:00", "Ref": "BNS-2026-0517-001", "Description": "Loyalty bonus credit"},
    {"ID": "TXN-20260518-018", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "PSP", "Amount": 42000.00, "Currency": "EUR", "Status": "Settled", "Client": "EuroTrade", "Counterparty": "Adyen (PSP)", "Bank": "Deutsche Bank", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 06:30:00", "Ref": "CRED-2026-0518-003", "Description": "PSP batch settlement"},
    {"ID": "TXN-20260518-019", "ISO_Msg": "pacs.008", "Type": "Transfer", "Source": "Bank", "Amount": 85000.00, "Currency": "USD", "Status": "Settled", "Client": "Internal", "Counterparty": "Treasury", "Bank": "HSBC", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 08:00:00", "Ref": "TRFR-2026-0518-002", "Description": "Inter-bank transfer"},
    {"ID": "TXN-20260518-020", "ISO_Msg": "pain.001", "Type": "Commission", "Source": "Commission", "Amount": 2100.00, "Currency": "USD", "Status": "Pending", "Client": "IB-Partner-Gamma", "Counterparty": "IB Commission Engine", "Bank": "JPMorgan Chase", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-19", "Timestamp": "2026-05-18 11:15:00", "Ref": "COMM-2026-0518-002", "Description": "Weekly IB accrual — IFRS 15"},
    {"ID": "TXN-20260518-021", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "CRM", "Amount": 1850000, "Currency": "JPY", "Status": "Settled", "Client": "TokyoFin", "Counterparty": "Stripe (PSP)", "Bank": "MUFG", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 04:10:00", "Ref": "CRED-2026-0518-004", "Description": "JPY deposit — FX converted"},
    {"ID": "TXN-20260518-022", "ISO_Msg": "pacs.008", "Type": "Deposit", "Source": "CRM", "Amount": 38000.00, "Currency": "CHF", "Status": "Settled", "Client": "SwissVault", "Counterparty": "Worldpay (PSP)", "Bank": "UBS", "Value_Date": "2026-05-18", "Settlement_Date": "2026-05-18", "Timestamp": "2026-05-18 07:15:00", "Ref": "CRED-2026-0518-005", "Description": "CHF deposit"},
]

# Reconciliation cases
reconciliation_data = [
    {"Case_ID": "REC-001", "TXN_ID": "TXN-20260518-001", "Status": "Matched", "Method": "ID + Amount", "Score": 100, "Source_Amt": 25000, "Target_Amt": 25000, "Source_CCY": "USD", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.03, "Case_Status": "Closed", "Notes": "Auto-matched — full reconciliation"},
    {"Case_ID": "REC-002", "TXN_ID": "TXN-20260518-003", "Status": "Partial", "Method": "ID + Time Window", "Score": 72, "Source_Amt": 8700, "Target_Amt": 9000, "Source_CCY": "EUR", "Variance": -300, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 4, "Resolved_In": None, "Case_Status": "Investigating", "Notes": "€300 variance — suspected FX rounding (IAS 21)"},
    {"Case_ID": "REC-003", "TXN_ID": "TXN-20260517-008", "Status": "Exception", "Method": "ID Match", "Score": 0, "Source_Amt": 45000, "Target_Amt": 0, "Source_CCY": "GBP", "Variance": -45000, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 4, "Resolved_In": None, "Case_Status": "Open", "Notes": "No bank record — pacs.004 return not reflected"},
    {"Case_ID": "REC-004", "TXN_ID": "TXN-20260517-007", "Status": "Matched", "Method": "ID + Amount", "Score": 100, "Source_Amt": 150000, "Target_Amt": 150000, "Source_CCY": "USD", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.05, "Case_Status": "Closed", "Notes": "Auto-matched"},
    {"Case_ID": "REC-005", "TXN_ID": "TXN-20260516-012", "Status": "Exception", "Method": "Amount + Time", "Score": 45, "Source_Amt": 5600, "Target_Amt": 5600, "Source_CCY": "USD", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 4, "Resolved_In": None, "Case_Status": "Investigating", "Notes": "Chargeback in PSP, not yet in bank (pacs.004 pending)"},
    {"Case_ID": "REC-006", "TXN_ID": "TXN-20260517-009", "Status": "Matched", "Method": "ID + Amount", "Score": 100, "Source_Amt": 32000, "Target_Amt": 32000, "Source_CCY": "USD", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.02, "Case_Status": "Closed", "Notes": "Auto-matched"},
    {"Case_ID": "REC-007", "TXN_ID": "TXN-20260518-010", "Status": "Partial", "Method": "ID + Time Window", "Score": 68, "Source_Amt": 18200, "Target_Amt": 18500, "Source_CCY": "EUR", "Variance": -300, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 4, "Resolved_In": None, "Case_Status": "Open", "Notes": "Amount discrepancy — FX or fee delta"},
    {"Case_ID": "REC-008", "TXN_ID": "TXN-20260518-002", "Status": "Matched", "Method": "ID + Amount", "Score": 100, "Source_Amt": 12500, "Target_Amt": 12500, "Source_CCY": "USD", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.04, "Case_Status": "Closed", "Notes": "Auto-matched"},
    {"Case_ID": "REC-009", "TXN_ID": "TXN-20260518-021", "Status": "Matched", "Method": "ID + Amount + FX", "Score": 98, "Source_Amt": 1850000, "Target_Amt": 1850000, "Source_CCY": "JPY", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.1, "Case_Status": "Closed", "Notes": "JPY matched — FX converted at IAS 21 spot rate"},
    {"Case_ID": "REC-010", "TXN_ID": "TXN-20260518-022", "Status": "Matched", "Method": "ID + Amount", "Score": 100, "Source_Amt": 38000, "Target_Amt": 38000, "Source_CCY": "CHF", "Variance": 0, "Source": "PSP Ledger", "Target": "Bank Statement (camt.053)", "SLA_Hours": 2, "Resolved_In": 0.06, "Case_Status": "Closed", "Notes": "Auto-matched"},
]

# Ledger entries (double-entry, IFRS accounts)
ledger_data = [
    {"ID": "JNL-001A", "Date": "2026-05-18", "Account": "1010", "Entry": "Debit", "Amount": 25000, "CCY": "USD", "Ref": "TXN-20260518-001", "Narration": "Cash received — client deposit", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-001B", "Date": "2026-05-18", "Account": "2000", "Entry": "Credit", "Amount": 25000, "CCY": "USD", "Ref": "TXN-20260518-001", "Narration": "Client funds payable — Acme Corp", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-002A", "Date": "2026-05-18", "Account": "2000", "Entry": "Debit", "Amount": 12500, "CCY": "USD", "Ref": "TXN-20260518-002", "Narration": "Client withdrawal — Globe Ltd", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-002B", "Date": "2026-05-18", "Account": "1010", "Entry": "Credit", "Amount": 12500, "CCY": "USD", "Ref": "TXN-20260518-002", "Narration": "Cash disbursed — withdrawal", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-003A", "Date": "2026-05-18", "Account": "1100", "Entry": "Debit", "Amount": 8700, "CCY": "EUR", "Ref": "TXN-20260518-003", "Narration": "PSP receivable — Stripe T+1", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-003B", "Date": "2026-05-18", "Account": "2000", "Entry": "Credit", "Amount": 8700, "CCY": "EUR", "Ref": "TXN-20260518-003", "Narration": "Client funds — NovaTech", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-004A", "Date": "2026-05-18", "Account": "1010", "Entry": "Debit", "Amount": 50000, "CCY": "USD", "Ref": "TXN-20260518-004", "Narration": "Treasury transfer in — operating", "Posted_By": "Treasury", "Approved_By": "CFO"},
    {"ID": "JNL-004B", "Date": "2026-05-18", "Account": "1020", "Entry": "Credit", "Amount": 50000, "CCY": "USD", "Ref": "TXN-20260518-004", "Narration": "Treasury transfer out — reserve", "Posted_By": "Treasury", "Approved_By": "CFO"},
    {"ID": "JNL-005A", "Date": "2026-05-18", "Account": "4000", "Entry": "Credit", "Amount": 125, "CCY": "USD", "Ref": "TXN-20260518-005", "Narration": "Fee revenue — IFRS 15 recognized", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-005B", "Date": "2026-05-18", "Account": "5000", "Entry": "Debit", "Amount": 62.50, "CCY": "USD", "Ref": "TXN-20260518-005", "Narration": "PSP cost — Stripe processing", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-006A", "Date": "2026-05-18", "Account": "5200", "Entry": "Debit", "Amount": 375, "CCY": "USD", "Ref": "TXN-20260518-006", "Narration": "Commission expense — IB Alpha", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-006B", "Date": "2026-05-18", "Account": "2100", "Entry": "Credit", "Amount": 375, "CCY": "USD", "Ref": "TXN-20260518-006", "Narration": "Commission payable — IB Alpha", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-016A", "Date": "2026-05-18", "Account": "5300", "Entry": "Debit", "Amount": 15000, "CCY": "USD", "Ref": "TXN-20260518-016", "Narration": "Bonus expense — IAS 19 provision", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-016B", "Date": "2026-05-18", "Account": "2200", "Entry": "Credit", "Amount": 15000, "CCY": "USD", "Ref": "TXN-20260518-016", "Narration": "Bonus liability — Acme Corp", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-007A", "Date": "2026-05-17", "Account": "1010", "Entry": "Debit", "Amount": 150000, "CCY": "USD", "Ref": "TXN-20260517-007", "Narration": "Cash received — MegaFund", "Posted_By": "System", "Approved_By": "Auto"},
    {"ID": "JNL-007B", "Date": "2026-05-17", "Account": "2000", "Entry": "Credit", "Amount": 150000, "CCY": "USD", "Ref": "TXN-20260517-007", "Narration": "Client funds — MegaFund", "Posted_By": "System", "Approved_By": "Auto"},
]

# Alerts
alerts_data = [
    {"ID": "ALT-001", "Title": "Cash Position Imbalance", "Category": "Financial", "Severity": "Critical", "Status": "Open", "Standard": "SOX 404", "Description": "GL cash balance deviates from bank statement by $12,400. Requires investigation per SOX internal controls.", "SLA_Hours": 4, "Created": "2026-05-18 10:15", "Linked": "TXN-20260518-004"},
    {"ID": "ALT-002", "Title": "Withdrawal Volume Anomaly", "Category": "Financial", "Severity": "High", "Status": "Investigating", "Standard": "Basel III LCR", "Description": "Withdrawal volume 340% above 30-day rolling average. LCR stress test triggered.", "SLA_Hours": 8, "Created": "2026-05-18 09:30", "Linked": "TXN-20260518-002, TXN-20260518-010"},
    {"ID": "ALT-003", "Title": "PSP Settlement Delay — Adyen", "Category": "Operational", "Severity": "Medium", "Status": "Open", "Standard": "ISO 20022 SLA", "Description": "Adyen pacs.008 batch delayed 4h beyond SLA. Estimated pending: €63,200.", "SLA_Hours": 12, "Created": "2026-05-18 08:00", "Linked": "TXN-20260518-010"},
    {"ID": "ALT-004", "Title": "Reconciliation Exception Rate Breach", "Category": "Operational", "Severity": "High", "Status": "Investigating", "Standard": "SOX 302", "Description": "Exception rate 20% — above 5% SOX threshold. Management attestation at risk.", "SLA_Hours": 4, "Created": "2026-05-18 10:00", "Linked": "REC-003, REC-005"},
    {"ID": "ALT-005", "Title": "Chargeback — IFRS 9 Provision", "Category": "Financial", "Severity": "Medium", "Status": "Resolved", "Standard": "IFRS 9", "Description": "Chargeback on TXN-012 booked. ECL provision updated per IFRS 9 impairment model.", "SLA_Hours": 24, "Created": "2026-05-16 10:30", "Linked": "TXN-20260516-012"},
    {"ID": "ALT-006", "Title": "LCR Below Regulatory Minimum", "Category": "Financial", "Severity": "Critical", "Status": "Open", "Standard": "Basel III", "Description": "Liquidity Coverage Ratio at 92% — below 100% regulatory minimum. Immediate action required.", "SLA_Hours": 2, "Created": "2026-05-18 07:00", "Linked": ""},
    {"ID": "ALT-007", "Title": "IAS 21 FX Revaluation Required", "Category": "Operational", "Severity": "Low", "Status": "Open", "Standard": "IAS 21", "Description": "EUR/USD rate moved >1% since last revaluation. Foreign currency balances require mark-to-market.", "SLA_Hours": 24, "Created": "2026-05-18 06:00", "Linked": ""},
]

# Bank balances
bank_balances = [
    {"Bank": "JPMorgan Chase", "SWIFT": "CHASUS33", "CCY": "USD", "Balance": 2450000, "HQLA_Level": "Level 1"},
    {"Bank": "HSBC", "SWIFT": "HSBCGB2L", "CCY": "USD", "Balance": 890000, "HQLA_Level": "Level 1"},
    {"Bank": "Deutsche Bank", "SWIFT": "DEUTDEFF", "CCY": "EUR", "Balance": 1250000, "HQLA_Level": "Level 1"},
    {"Bank": "Barclays", "SWIFT": "BARCGB22", "CCY": "GBP", "Balance": 675000, "HQLA_Level": "Level 1"},
    {"Bank": "Barclays", "SWIFT": "BARCGB22", "CCY": "USD", "Balance": 320000, "HQLA_Level": "Level 1"},
    {"Bank": "UBS", "SWIFT": "UBSWCHZH", "CCY": "CHF", "Balance": 485000, "HQLA_Level": "Level 1"},
    {"Bank": "MUFG", "SWIFT": "BOTKJPJT", "CCY": "JPY", "Balance": 52000000, "HQLA_Level": "Level 1"},
]

psp_balances = [
    {"PSP": "Stripe", "CCY": "USD", "Balance": 185000, "Pending_In": 32000, "Pending_Out": 8500, "Settlement_Cycle": "T+1"},
    {"PSP": "Stripe", "CCY": "EUR", "Balance": 45000, "Pending_In": 8700, "Pending_Out": 0, "Settlement_Cycle": "T+1"},
    {"PSP": "Adyen", "CCY": "USD", "Balance": 92000, "Pending_In": 0, "Pending_Out": 12500, "Settlement_Cycle": "T+1"},
    {"PSP": "Adyen", "CCY": "EUR", "Balance": 63200, "Pending_In": 0, "Pending_Out": 18200, "Settlement_Cycle": "T+1"},
    {"PSP": "Worldpay", "CCY": "USD", "Balance": 210000, "Pending_In": 0, "Pending_Out": 0, "Settlement_Cycle": "T+2"},
    {"PSP": "Worldpay", "CCY": "GBP", "Balance": 55000, "Pending_In": 0, "Pending_Out": 45000, "Settlement_Cycle": "T+2"},
]

cash_flow_data = [
    {"Date": "May 12", "Deposits": 185000, "Withdrawals": 72000, "Net_Flow": 113000},
    {"Date": "May 13", "Deposits": 210000, "Withdrawals": 95000, "Net_Flow": 115000},
    {"Date": "May 14", "Deposits": 145000, "Withdrawals": 120000, "Net_Flow": 25000},
    {"Date": "May 15", "Deposits": 290000, "Withdrawals": 88000, "Net_Flow": 202000},
    {"Date": "May 16", "Deposits": 178000, "Withdrawals": 156000, "Net_Flow": 22000},
    {"Date": "May 17", "Deposits": 320000, "Withdrawals": 105000, "Net_Flow": 215000},
    {"Date": "May 18", "Deposits": 307700, "Withdrawals": 75700, "Net_Flow": 232000},
]

liquidity_trend = [
    {"Date": "May 12", "Available_USD": 4200000, "Buffer_Pct": 18.2, "LCR": 118},
    {"Date": "May 13", "Available_USD": 4315000, "Buffer_Pct": 17.8, "LCR": 115},
    {"Date": "May 14", "Available_USD": 4050000, "Buffer_Pct": 15.5, "LCR": 108},
    {"Date": "May 15", "Available_USD": 4400000, "Buffer_Pct": 16.1, "LCR": 112},
    {"Date": "May 16", "Available_USD": 4150000, "Buffer_Pct": 14.2, "LCR": 104},
    {"Date": "May 17", "Available_USD": 4580000, "Buffer_Pct": 13.8, "LCR": 98},
    {"Date": "May 18", "Available_USD": 4585000, "Buffer_Pct": 11.2, "LCR": 92},
]

profitability_data = [
    {"Client": "Acme Corp", "Revenue": 45000, "Costs": 12000, "Profit": 33000, "Margin": 73.3},
    {"Client": "Globe Ltd", "Revenue": 28000, "Costs": 9500, "Profit": 18500, "Margin": 66.1},
    {"Client": "MegaFund", "Revenue": 82000, "Costs": 18000, "Profit": 64000, "Margin": 78.0},
    {"Client": "NovaTech", "Revenue": 15000, "Costs": 6200, "Profit": 8800, "Margin": 58.7},
    {"Client": "SolarInc", "Revenue": 32000, "Costs": 8800, "Profit": 23200, "Margin": 72.5},
    {"Client": "TradeCo", "Revenue": 51000, "Costs": 14500, "Profit": 36500, "Margin": 71.6},
]

ib_data = [
    {"Partner": "IB-Partner-Alpha", "Clients": 42, "Volume": 1250000, "Commission": 18750, "Net_Revenue": 62500},
    {"Partner": "IB-Partner-Beta", "Clients": 28, "Volume": 890000, "Commission": 13350, "Net_Revenue": 44500},
    {"Partner": "IB-Partner-Gamma", "Clients": 65, "Volume": 2100000, "Commission": 31500, "Net_Revenue": 105000},
    {"Partner": "IB-Partner-Delta", "Clients": 15, "Volume": 420000, "Commission": 6300, "Net_Revenue": 21000},
]

monthly_kpis = [
    {"Month": "Jan 2026", "Net_Flow": 1200000, "Op_Costs": 85000, "Exceptions": 12, "Recon_Rate": 94.2},
    {"Month": "Feb 2026", "Net_Flow": 980000, "Op_Costs": 78000, "Exceptions": 8, "Recon_Rate": 96.1},
    {"Month": "Mar 2026", "Net_Flow": 1450000, "Op_Costs": 92000, "Exceptions": 15, "Recon_Rate": 93.8},
    {"Month": "Apr 2026", "Net_Flow": 1100000, "Op_Costs": 88000, "Exceptions": 10, "Recon_Rate": 95.5},
    {"Month": "May 2026", "Net_Flow": 1680000, "Op_Costs": 95000, "Exceptions": 18, "Recon_Rate": 91.0},
]

# ══════════════════════════════════════════════════════════════
# DATAFRAMES
# ══════════════════════════════════════════════════════════════
df_txn = pd.DataFrame(transactions_data)
df_rec = pd.DataFrame(reconciliation_data)
df_led = pd.DataFrame(ledger_data)
df_alerts = pd.DataFrame(alerts_data)
df_bank = pd.DataFrame(bank_balances)
df_psp = pd.DataFrame(psp_balances)
df_cash = pd.DataFrame(cash_flow_data)
df_liq = pd.DataFrame(liquidity_trend)
df_profit = pd.DataFrame(profitability_data)
df_ib = pd.DataFrame(ib_data)
df_kpi = pd.DataFrame(monthly_kpis)
df_coa = pd.DataFrame([{"Code": k, **v} for k, v in CHART_OF_ACCOUNTS.items()])

# Enrich ledger with account names
df_led["Account_Name"] = df_led["Account"].map(lambda x: CHART_OF_ACCOUNTS.get(x, {}).get("name", x))
df_led["Account_Class"] = df_led["Account"].map(lambda x: CHART_OF_ACCOUNTS.get(x, {}).get("class", ""))

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def sev_icon(sev):
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🔵"}.get(sev, "⚪")

def badge(text, color="gray"):
    return f'<span class="badge bg-{color}">{text}</span>'

PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=10, b=0))

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏛️ **FinanceOps**")
    st.caption("IFRS · ISO 20022 · Basel III · SOX")
    st.markdown("---")
    open_alerts = len(df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])])
    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🏗️ System Architecture",
        "💸 Transactions",
        "🔄 Reconciliation",
        "📒 General Ledger",
        "💧 Liquidity & Treasury",
        f"⚠️ Alerts & Exceptions ({open_alerts})",
        "📈 Reports & Analytics",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("🟢 **System Online**")
    st.caption(datetime.now().strftime("%a, %b %d, %Y — %H:%M UTC"))
    st.markdown("---")
    st.caption("Standards compliance:")
    st.caption("IFRS / IAS · ISO 20022 · ISO 4217")
    st.caption("Basel III (LCR/NSFR) · SOX 302/404")


# ══════════════════════════════════════════════════════════════
# 1. DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard")
    st.caption("Finance Automation Platform — Real-time Financial Overview")

    opening_balance = 4250000
    cash_in = df_txn[(df_txn["Type"].isin(["Deposit"])) & (df_txn["Value_Date"] == "2026-05-18") & (df_txn["Status"].isin(["Settled", "Pending"]))]["Amount"].apply(lambda a: to_usd(a, df_txn.loc[df_txn["Amount"] == a, "Currency"].iloc[0]) if len(df_txn.loc[df_txn["Amount"] == a]) > 0 else a).sum()
    cash_in_today = df_txn[(df_txn["Type"] == "Deposit") & (df_txn["Value_Date"] == "2026-05-18")]["Amount"].sum()
    cash_out_today = df_txn[(df_txn["Type"].isin(["Withdrawal", "Fee", "Commission"])) & (df_txn["Value_Date"] == "2026-05-18")]["Amount"].sum()
    net_flow = cash_in_today - cash_out_today

    # Financial Overview
    st.subheader("💵 Financial Overview")
    fo1, fo2, fo3, fo4 = st.columns(4)
    fo1.metric("Opening Balance", fmt_usd(opening_balance))
    fo2.metric("Cash In Today", fmt_usd(cash_in_today), f"+{cash_in_today/opening_balance*100:.1f}%")
    fo3.metric("Cash Out Today", fmt_usd(cash_out_today), f"-{cash_out_today/opening_balance*100:.1f}%", delta_color="inverse")
    fo4.metric("Net Flow", fmt_usd(net_flow), f"{'+'if net_flow>=0 else ''}{net_flow/opening_balance*100:.1f}%")

    # Liquidity (Basel III)
    st.subheader("💧 Liquidity — Basel III Metrics")
    total_bank_usd = sum(to_usd(b["Balance"], b["CCY"]) for b in bank_balances)
    total_psp_usd = sum(to_usd(p["Balance"], p["CCY"]) for p in psp_balances)
    available = total_bank_usd + total_psp_usd
    pending_wd = df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Pending")]["Amount"].sum() + df_psp["Pending_Out"].sum()
    lcr_current = df_liq.iloc[-1]["LCR"]

    lq1, lq2, lq3, lq4 = st.columns(4)
    lq1.metric("Available Cash (USD eq.)", fmt_usd(available))
    lq2.metric("Pending Outflows", fmt_usd(pending_wd))
    lq3.metric("LCR (Liquidity Coverage)", f"{lcr_current}%", "-6%" if lcr_current < 100 else "+OK", delta_color="inverse" if lcr_current < 100 else "normal")
    lq4.metric("HQLA (Level 1)", fmt_usd(total_bank_usd), "All bank balances")

    cl, cr = st.columns(2)
    with cl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Deposits"], mode="lines", name="Cash In", fill="tozeroy", line=dict(color="#10b981", width=2), fillcolor="rgba(16,185,129,.12)"))
        fig.add_trace(go.Scatter(x=df_cash["Date"], y=df_cash["Withdrawals"], mode="lines", name="Cash Out", fill="tozeroy", line=dict(color="#ef4444", width=2), fillcolor="rgba(239,68,68,.12)"))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    with cr:
        fig_lcr = go.Figure()
        fig_lcr.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["LCR"], mode="lines+markers", line=dict(color="#6366f1", width=2.5), fill="tozeroy", fillcolor="rgba(99,102,241,.1)"))
        fig_lcr.add_hline(y=100, line_dash="dash", line_color="#ef4444", annotation_text="100% Regulatory Min")
        fig_lcr.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(fig_lcr, use_container_width=True)

    # Alerts
    st.subheader("🚨 Alerts")
    unresolved = len(df_rec[df_rec["Status"] == "Exception"])
    delays = len(df_alerts[df_alerts["Title"].str.contains("Delay|Settlement", case=False) & (df_alerts["Status"] != "Resolved")])
    liq_warn = len(df_alerts[df_alerts["Standard"].str.contains("Basel|LCR", case=False) & (df_alerts["Status"] != "Resolved")])

    a1, a2, a3 = st.columns(3)
    a1.metric("Unresolved Exceptions", unresolved, delta_color="inverse")
    a2.metric("Settlement Delays", delays, delta_color="inverse")
    a3.metric("Liquidity Warnings", liq_warn, delta_color="inverse")

    for _, al in df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])].iterrows():
        st.markdown(f"""<div class="alert-card">
            <div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div>
            <div class="alert-desc">{al["Description"]}</div>
            <div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Standard"],"blue")} {badge(al["Status"],"yellow")}</div>
        </div>""", unsafe_allow_html=True)

    # KPIs
    st.subheader("📈 KPIs")
    daily_vol = len(df_txn[df_txn["Value_Date"] == "2026-05-18"])
    m = len(df_rec[df_rec["Status"] == "Matched"])
    rec_rate = m / len(df_rec) * 100 if len(df_rec) > 0 else 0
    unmatched = len(df_rec[df_rec["Status"].isin(["Partial", "Exception"])])

    k1, k2, k3 = st.columns(3)
    k1.metric("Daily Volume", f"{daily_vol} transactions")
    k2.metric("Reconciliation Rate", f"{rec_rate:.1f}%", f"{m}/{len(df_rec)} matched")
    k3.metric("Unmatched Transactions", unmatched, delta_color="inverse")


# ══════════════════════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════
elif page == "🏗️ System Architecture":
    st.title("🏗️ System Architecture")
    st.caption("End-to-end pipeline — ISO 20022 messaging, IFRS ledger, Basel III liquidity")

    # Pipeline Health
    st.subheader("Pipeline Health")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.markdown(f'<div class="metric-box"><div class="val">6</div><div class="lbl">Connected Systems</div></div>', unsafe_allow_html=True)
    p2.markdown(f'<div class="metric-box"><div class="val">4</div><div class="lbl">API Connectors</div></div>', unsafe_allow_html=True)
    p3.markdown(f'<div class="metric-box"><div class="val">{len(df_txn)}</div><div class="lbl">Txns Processed</div></div>', unsafe_allow_html=True)
    p4.markdown(f'<div class="metric-box"><div class="val">{len(df_rec[df_rec["Status"]=="Matched"])}/{len(df_rec)}</div><div class="lbl">Reconciled</div></div>', unsafe_allow_html=True)
    p5.markdown(f'<div class="metric-box"><div class="val">{len(df_led)}</div><div class="lbl">GL Entries</div></div>', unsafe_allow_html=True)
    p6.markdown(f'<div class="metric-box"><div class="val">{len(df_alerts[df_alerts["Status"].isin(["Open","Investigating"])])}</div><div class="lbl">Active Alerts</div></div>', unsafe_allow_html=True)

    # Connected Systems
    st.subheader("1️⃣ Connected Systems")
    systems = [
        ("CRM", "Client deposits, withdrawals, KYC", "Online", "REST API"),
        ("PSP Gateways", "Stripe, Adyen, Worldpay — ISO 20022 pacs.*", "Online", "REST + Webhook"),
        ("Bank Accounts", "JPMorgan, HSBC, Deutsche, Barclays, UBS, MUFG", "Online", "SWIFT / camt.053"),
        ("Trading Platform", "Client P&L, margin, positions", "Online", "WebSocket"),
        ("Bonus Engine", "Welcome, loyalty, referral — IAS 19", "Delayed", "REST API"),
        ("Commission Engine", "IB commissions — IFRS 15", "Online", "REST API"),
    ]
    sc = st.columns(3)
    for i, (name, desc, status, protocol) in enumerate(systems):
        dot = "dot-green" if status == "Online" else "dot-yellow"
        sc[i % 3].markdown(f'<div class="card"><h4><span class="status-dot {dot}"></span>{name}</h4><div class="card-sm">{desc}</div><div class="card-sm" style="margin-top:6px">Protocol: {protocol}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Data Integration
    st.subheader("2️⃣ Data Integration Layer")
    di = st.columns(4)
    for idx, (title, items) in enumerate([
        ("API Connectors", ["REST (CRM, PSP)", "WebSocket (Trading)", "SWIFT/SFTP (Banks)", "OAuth 2.0 + mTLS"]),
        ("File Imports", ["camt.053 (Bank Stmts)", "MT940 (Legacy SWIFT)", "PSP Settlement CSV", "Commission batch XML"]),
        ("Scheduled Pull", ["Real-time: PSP, CRM", "Every 5 min: Trading", "Every 15 min: Banks", "Hourly: Commissions"]),
        ("Data Validation", ["ISO 20022 schema", "ISO 4217 currency check", "Duplicate detection", "Referential integrity"]),
    ]):
        items_html = "".join([f'<div class="card-sm" style="padding:2px 0">• {x}</div>' for x in items])
        di[idx].markdown(f'<div class="card"><h4>{title}</h4>{items_html}</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Normalization
    st.subheader("3️⃣ Data Normalization (ISO Standards)")
    dn = st.columns(4)
    for idx, (title, items) in enumerate([
        ("Currency (ISO 4217)", ["Base currency: USD", "FX rates: IAS 21 spot", f"{len(ISO_CCY)} currencies supported", "Auto-conversion engine"]),
        ("Time (ISO 8601)", ["UTC normalization", "Value date / Settlement date", "T+0 / T+1 / T+2 cycles", "Timezone offset handling"]),
        ("ID Mapping", ["End-to-end TXN ref", "SWIFT BIC ↔ Internal", "Client ID unification", "PSP ref → GL posting"]),
        ("Type Mapping (ISO 20022)", [f"pacs.008 → Deposit/Transfer", f"pacs.003 → Withdrawal", f"pacs.004 → Return/Reversal", f"camt.054 → Fee/Bonus"]),
    ]):
        items_html = "".join([f'<div class="card-sm" style="padding:2px 0">• {x}</div>' for x in items])
        dn[idx].markdown(f'<div class="card"><h4>{title}</h4>{items_html}</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Reconciliation Engine
    st.subheader("4️⃣ Reconciliation Engine (3-Way Match)")
    re1, re2, re3, re4 = st.columns(4)
    re1.metric("ID Match", f"{len(df_rec[df_rec['Score']==100])}", "Exact ref match")
    re2.metric("Amount Match", f"{len(df_rec[df_rec['Score'].between(60,99)])}", "Within tolerance")
    re3.metric("Time Window", "±2 hours", "Settlement tolerance")
    re4.metric("Exceptions", f"{len(df_rec[df_rec['Status']=='Exception'])}", delta_color="inverse")
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Central Ledger
    st.subheader("5️⃣ Central Financial Ledger (IFRS)")
    classes = ["Asset", "Liability", "Equity", "Revenue", "Expense"]
    lc = st.columns(5)
    for i, cls in enumerate(classes):
        accts = [f"{v['Code']}: {v['name']}" for _, v in df_coa[df_coa["class"] == cls].iterrows()]
        accts_html = "".join([f'<div class="card-sm" style="padding:1px 0">• {a}</div>' for a in accts[:4]])
        lc[i].markdown(f'<div class="card"><h4>{cls}</h4>{accts_html}</div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Liquidity + Alert engines
    el, er = st.columns(2)
    with el:
        st.subheader("6a. Liquidity Engine (Basel III)")
        for label, val, color in [("HQLA (Level 1)", fmt_usd(sum(to_usd(b["Balance"], b["CCY"]) for b in bank_balances)), "#10b981"), ("Net Cash Outflows", fmt_usd(df_psp["Pending_Out"].sum()), "#ef4444"), ("LCR", f"{df_liq.iloc[-1]['LCR']}%", "#ef4444" if df_liq.iloc[-1]["LCR"] < 100 else "#10b981"), ("NSFR (est.)", "105%", "#10b981")]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 12px;background:#0f1729;border-radius:6px;margin-bottom:4px;border:1px solid #1e2d4a"><span class="card-sm">{label}</span><span style="font-weight:700;color:{color}">{val}</span></div>', unsafe_allow_html=True)
    with er:
        st.subheader("6b. Alert & Exception Engine")
        for label, val, color in [("Recon Exceptions", f"{len(df_rec[df_rec['Status']=='Exception'])} active", "#ef4444"), ("PSP SLA Breaches", f"{len(df_alerts[df_alerts['Title'].str.contains('PSP|Delay', case=False)])} active", "#f59e0b"), ("SOX Control Failures", f"{len(df_alerts[df_alerts['Standard'].str.contains('SOX', case=False)])} flagged", "#ef4444"), ("Threshold Breaches", f"{len(df_alerts[df_alerts['Standard'].str.contains('Basel', case=False)])} active", "#f59e0b")]:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 12px;background:#0f1729;border-radius:6px;margin-bottom:4px;border:1px solid #1e2d4a"><span class="card-sm">{label}</span><span style="font-weight:700;color:{color}">{val}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # Reporting
    st.subheader("7️⃣ Reporting & Analytics")
    rpts = [("📊", "Dashboard", "Real-time"), ("📅", "Daily Report", "T+1 @ 01:00"), ("📋", "Weekly Review", "Mon @ 06:00"), ("📊", "Monthly Close", "1st of month"), ("💰", "Profitability", "On demand"), ("📈", "KPI Monitor", "Real-time")]
    rc = st.columns(6)
    for i, (icon, name, freq) in enumerate(rpts):
        rc[i].markdown(f'<div class="card" style="text-align:center;min-height:100px"><div style="font-size:22px">{icon}</div><h4 style="font-size:11px">{name}</h4><div class="card-sm">{freq}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    # End Users
    st.subheader("8️⃣ End Users")
    users = [("👔", "CFO", "Executive"), ("📊", "Finance Manager", "Management"), ("⚙️", "Operations", "Operational"), ("🔄", "Recon Team", "Operational"), ("🏢", "Management", "Executive")]
    uc = st.columns(5)
    for i, (icon, role, level) in enumerate(users):
        uc[i].markdown(f'<div class="card" style="text-align:center"><div style="font-size:26px">{icon}</div><h4 style="font-size:12px">{role}</h4>{badge(level, "purple" if level=="Executive" else "green")}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 3. TRANSACTIONS
# ══════════════════════════════════════════════════════════════
elif page == "💸 Transactions":
    st.title("💸 Transactions")
    st.caption("ISO 20022 compliant — pacs.008, pacs.003, pacs.004, camt.054, pain.001")

    # Source KPIs
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("CRM", len(df_txn[df_txn["Source"] == "CRM"]))
    s2.metric("PSP Payments", len(df_txn[df_txn["Source"] == "PSP"]))
    s3.metric("Bank Transfers", len(df_txn[df_txn["Source"] == "Bank"]))
    s4.metric("Commissions", len(df_txn[df_txn["Source"] == "Commission"]))
    s5.metric("Bonuses", len(df_txn[df_txn["Source"] == "Bonus"]))

    tabs = st.tabs(["All", "CRM", "PSP", "Bank", "Commission", "Bonus"])
    sources = ["All", "CRM", "PSP", "Bank", "Commission", "Bonus"]

    for idx, tab in enumerate(tabs):
        with tab:
            base = df_txn if sources[idx] == "All" else df_txn[df_txn["Source"] == sources[idx]]
            f1, f2, f3, f4 = st.columns(4)
            with f1: search = st.text_input("Search", key=f"s_{idx}", placeholder="ID, client...")
            with f2: type_f = st.selectbox("Type", ["All"] + sorted(base["Type"].unique().tolist()), key=f"t_{idx}")
            with f3: status_f = st.selectbox("Status", ["All"] + sorted(base["Status"].unique().tolist()), key=f"st_{idx}")
            with f4: ccy_f = st.selectbox("Currency", ["All"] + sorted(base["Currency"].unique().tolist()), key=f"c_{idx}")

            filt = base.copy()
            if search:
                filt = filt[filt.apply(lambda r: search.lower() in r["ID"].lower() or search.lower() in r["Client"].lower(), axis=1)]
            if type_f != "All": filt = filt[filt["Type"] == type_f]
            if status_f != "All": filt = filt[filt["Status"] == status_f]
            if ccy_f != "All": filt = filt[filt["Currency"] == ccy_f]

            st.dataframe(filt[["ID", "ISO_Msg", "Type", "Source", "Client", "Amount", "Currency", "Counterparty", "Status", "Value_Date", "Ref"]], use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="%.2f")})

    st.markdown("---")
    st.subheader("ISO 20022 Message Types")
    msg_df = pd.DataFrame([{"Code": k, "Description": v, "Count": len(df_txn[df_txn["ISO_Msg"] == k])} for k, v in ISO20022_MSG.items() if len(df_txn[df_txn["ISO_Msg"] == k]) > 0])
    if len(msg_df) > 0:
        st.dataframe(msg_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Transaction Details")
    sel = st.selectbox("Select transaction", df_txn["ID"].tolist())
    if sel:
        tx = df_txn[df_txn["ID"] == sel].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**ID:** {tx['ID']}\n\n**ISO Msg:** {tx['ISO_Msg']} ({ISO20022_MSG.get(tx['ISO_Msg'], '')})\n\n**Type:** {tx['Type']}\n\n**Amount:** {fmt(tx['Amount'], tx['Currency'])}")
        c2.markdown(f"**Status:** {tx['Status']}\n\n**Client:** {tx['Client']}\n\n**Counterparty:** {tx['Counterparty']}\n\n**Source:** {tx['Source']}")
        c3.markdown(f"**Bank:** {tx['Bank']}\n\n**Value Date:** {tx['Value_Date']}\n\n**Settlement:** {tx['Settlement_Date']}\n\n**Reference:** {tx['Ref']}")


# ══════════════════════════════════════════════════════════════
# 4. RECONCILIATION
# ══════════════════════════════════════════════════════════════
elif page == "🔄 Reconciliation":
    st.title("🔄 Reconciliation")
    st.caption("3-way matching — PSP Ledger ↔ Bank Statement (camt.053) ↔ Internal GL")

    matched = len(df_rec[df_rec["Status"] == "Matched"])
    partial = len(df_rec[df_rec["Status"] == "Partial"])
    exceptions = len(df_rec[df_rec["Status"] == "Exception"])
    avg_sla = df_rec[df_rec["Resolved_In"].notna()]["Resolved_In"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ Matched", matched, f"{matched/len(df_rec)*100:.0f}%")
    c2.metric("⚠️ Partial Match", partial, f"{partial/len(df_rec)*100:.0f}%")
    c3.metric("❌ Exceptions", exceptions, delta_color="inverse")
    c4.metric("Avg Resolution", f"{avg_sla:.2f} hours" if avg_sla else "N/A")

    tabs = st.tabs(["All Cases", "Matched", "Partial", "Exceptions"])
    for i, tab in enumerate(tabs):
        with tab:
            data = df_rec if i == 0 else df_rec[df_rec["Status"] == ["Matched", "Matched", "Partial", "Exception"][i]]
            st.dataframe(data[["Case_ID", "TXN_ID", "Status", "Method", "Score", "Source_Amt", "Target_Amt", "Source_CCY", "Variance", "Source", "Target", "Case_Status"]], use_container_width=True, hide_index=True,
                column_config={"Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%"), "Source_Amt": st.column_config.NumberColumn(format="%.0f"), "Target_Amt": st.column_config.NumberColumn(format="%.0f"), "Variance": st.column_config.NumberColumn(format="%.0f")})

    st.markdown("---")
    st.subheader("Case Actions")
    open_cases = df_rec[df_rec["Case_Status"].isin(["Open", "Investigating"])]
    if len(open_cases) > 0:
        sel_case = st.selectbox("Select case", open_cases["Case_ID"].tolist())
        cs = df_rec[df_rec["Case_ID"] == sel_case].iloc[0]
        st.info(f"**{cs['Case_ID']}** — {cs['Notes']} (Score: {cs['Score']}%, Variance: {fmt(abs(cs['Variance']), cs['Source_CCY'])})")
        bc1, bc2, bc3 = st.columns(3)
        bc1.button("🔍 Investigate", key="inv_r", use_container_width=True)
        bc2.button("✅ Approve", key="app_r", use_container_width=True)
        bc3.button("❌ Reject", key="rej_r", use_container_width=True)

    st.subheader("SLA Performance")
    resolved = df_rec[df_rec["Resolved_In"].notna()]
    if len(resolved) > 0:
        fig_sla = px.bar(resolved, x="Case_ID", y="Resolved_In", color="Status", color_discrete_map={"Matched": "#10b981"}, labels={"Resolved_In": "Hours"})
        fig_sla.add_hline(y=2, line_dash="dash", line_color="#f59e0b", annotation_text="2h SLA Target")
        fig_sla.update_layout(**PLOTLY_LAYOUT, height=240, showlegend=False)
        st.plotly_chart(fig_sla, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 5. GENERAL LEDGER
# ══════════════════════════════════════════════════════════════
elif page == "📒 General Ledger":
    st.title("📒 General Ledger")
    st.caption("IFRS-compliant double-entry bookkeeping — IAS 1 presentation")

    total_dr = df_led[df_led["Entry"] == "Debit"]["Amount"].sum()
    total_cr = df_led[df_led["Entry"] == "Credit"]["Amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Debits", fmt_usd(total_dr))
    c2.metric("Total Credits", fmt_usd(total_cr))
    c3.metric("Balance Check", "✅ Balanced" if abs(total_dr - total_cr) < 0.01 else "❌ Imbalanced")
    c4.metric("Journal Entries", len(df_led))

    # Chart of Accounts
    st.subheader("IFRS Chart of Accounts")
    st.dataframe(df_coa[["Code", "name", "class", "ifrs", "normal"]], use_container_width=True, hide_index=True, column_config={"Code": "Account Code", "name": "Account Name", "class": "Class", "ifrs": "IFRS Standard", "normal": "Normal Balance"})

    # Journal entries
    st.subheader("Journal Entries")
    tab_all, tab_dr, tab_cr = st.tabs(["All Entries", "Debits", "Credits"])

    cls_filter = st.selectbox("Filter by Class", ["All"] + sorted(df_led["Account_Class"].unique().tolist()))
    led_view = df_led if cls_filter == "All" else df_led[df_led["Account_Class"] == cls_filter]

    with tab_all: st.dataframe(led_view[["ID", "Date", "Account", "Account_Name", "Entry", "Amount", "CCY", "Ref", "Narration", "Posted_By", "Approved_By"]], use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="%.2f")})
    with tab_dr: st.dataframe(led_view[led_view["Entry"] == "Debit"][["ID", "Date", "Account", "Account_Name", "Amount", "CCY", "Ref", "Narration"]], use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="%.2f")})
    with tab_cr: st.dataframe(led_view[led_view["Entry"] == "Credit"][["ID", "Date", "Account", "Account_Name", "Amount", "CCY", "Ref", "Narration"]], use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn(format="%.2f")})

    # Trial Balance
    st.subheader("Trial Balance")
    tb = df_led.groupby(["Account", "Account_Name", "Account_Class"]).apply(lambda g: pd.Series({"Debits": g[g["Entry"] == "Debit"]["Amount"].sum(), "Credits": g[g["Entry"] == "Credit"]["Amount"].sum()})).reset_index()
    tb["Net"] = tb["Debits"] - tb["Credits"]
    st.dataframe(tb, use_container_width=True, hide_index=True, column_config={"Debits": st.column_config.NumberColumn(format="$%.2f"), "Credits": st.column_config.NumberColumn(format="$%.2f"), "Net": st.column_config.NumberColumn(format="$%.2f")})

    st.subheader("Audit Trail (SOX 404)")
    st.dataframe(df_led.sort_values("Date", ascending=False).head(10)[["Date", "ID", "Account_Name", "Entry", "Amount", "CCY", "Posted_By", "Approved_By", "Narration"]], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 6. LIQUIDITY & TREASURY
# ══════════════════════════════════════════════════════════════
elif page == "💧 Liquidity & Treasury":
    st.title("💧 Liquidity & Treasury")
    st.caption("Basel III LCR / NSFR — Multi-currency treasury management")

    total_bank_usd = sum(to_usd(b["Balance"], b["CCY"]) for b in bank_balances)
    total_psp_usd = sum(to_usd(p["Balance"], p["CCY"]) for p in psp_balances)
    available = total_bank_usd + total_psp_usd
    pending_out = df_psp["Pending_Out"].sum() + df_txn[(df_txn["Type"] == "Withdrawal") & (df_txn["Status"] == "Pending")]["Amount"].sum()
    bonus_liability = 125000
    commission_liability = 48500
    net_liq = available - pending_out - bonus_liability - commission_liability
    lcr = df_liq.iloc[-1]["LCR"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total HQLA (USD eq.)", fmt_usd(total_bank_usd))
    c2.metric("LCR", f"{lcr}%", "Below 100%" if lcr < 100 else "Above min", delta_color="inverse" if lcr < 100 else "normal")
    c3.metric("Net Liquidity", fmt_usd(net_liq))
    c4.metric("NSFR (est.)", "105%", "Above 100%")

    # Multi-currency bank balances
    st.subheader("🏦 Bank Balances — Multi-Currency (ISO 4217)")
    bank_display = df_bank.copy()
    bank_display["Balance_USD"] = [to_usd(r["Balance"], r["CCY"]) for _, r in df_bank.iterrows()]
    st.dataframe(bank_display[["Bank", "SWIFT", "CCY", "Balance", "Balance_USD", "HQLA_Level"]], use_container_width=True, hide_index=True,
        column_config={"Balance": st.column_config.NumberColumn(format="%.0f"), "Balance_USD": st.column_config.NumberColumn(format="$%.0f")})

    st.subheader("💳 PSP Balances")
    st.dataframe(df_psp, use_container_width=True, hide_index=True,
        column_config={"Balance": st.column_config.NumberColumn(format="%.0f"), "Pending_In": st.column_config.NumberColumn(format="%.0f"), "Pending_Out": st.column_config.NumberColumn(format="%.0f")})

    col_lcr, col_buf = st.columns(2)
    with col_lcr:
        st.subheader("LCR Trend")
        fig_lcr = go.Figure()
        fig_lcr.add_trace(go.Scatter(x=df_liq["Date"], y=df_liq["LCR"], mode="lines+markers", line=dict(color="#6366f1", width=2.5), fill="tozeroy", fillcolor="rgba(99,102,241,.1)"))
        fig_lcr.add_hline(y=100, line_dash="dash", line_color="#ef4444", annotation_text="100% Basel III Minimum")
        fig_lcr.update_layout(**PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_lcr, use_container_width=True)
    with col_buf:
        st.subheader("Liquidity Calculation")
        calc = pd.DataFrame({"Item": ["HQLA (Level 1)", "PSP Balances", "Pending Outflows", "Bonus Liability (IAS 19)", "Commission Liability (IFRS 15)", "Net Liquidity"], "Amount": [total_bank_usd, total_psp_usd, -pending_out, -bonus_liability, -commission_liability, net_liq]})
        fig_calc = px.bar(calc, y="Item", x="Amount", orientation="h", color=calc["Amount"].apply(lambda x: "Positive" if x >= 0 else "Negative"), color_discrete_map={"Positive": "#10b981", "Negative": "#ef4444"})
        fig_calc.update_layout(**PLOTLY_LAYOUT, height=280, showlegend=False)
        st.plotly_chart(fig_calc, use_container_width=True)

    st.subheader("FX Rates (IAS 21)")
    fx_df = pd.DataFrame([{"Pair": k, "Rate": v} for k, v in FX_RATES.items() if k != "USD/USD"])
    st.dataframe(fx_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# 7. ALERTS & EXCEPTIONS
# ══════════════════════════════════════════════════════════════
elif "Alerts" in page:
    st.title("⚠️ Alerts & Exceptions")
    st.caption("SLA-driven alert management — SOX 302/404 compliance")

    critical = len(df_alerts[(df_alerts["Severity"] == "Critical") & (df_alerts["Status"] != "Resolved")])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Critical", critical)
    c2.metric("🟡 Open", len(df_alerts[df_alerts["Status"] == "Open"]))
    c3.metric("🔵 Investigating", len(df_alerts[df_alerts["Status"] == "Investigating"]))
    c4.metric("🟢 Resolved", len(df_alerts[df_alerts["Status"] == "Resolved"]))

    tab_all, tab_fin, tab_ops = st.tabs(["All", "Financial", "Operational"])

    sev_f = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"])
    stat_f = st.selectbox("Status", ["All", "Open", "Investigating", "Resolved"], key="a_stat")

    def filter_show(data):
        d = data.copy()
        if sev_f != "All": d = d[d["Severity"] == sev_f]
        if stat_f != "All": d = d[d["Status"] == stat_f]
        if len(d) == 0:
            st.success("No alerts match filters")
            return
        for _, al in d.iterrows():
            st.markdown(f"""<div class="alert-card">
                <div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div>
                <div class="alert-desc">{al["Description"]}</div>
                <div style="margin-top:6px">{badge(al["Severity"], "red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Standard"], "blue")} {badge(al["Status"], "yellow" if al["Status"] != "Resolved" else "green")} <span class="card-sm" style="margin-left:6px">SLA: {al["SLA_Hours"]}h</span></div>
            </div>""", unsafe_allow_html=True)

    with tab_all: filter_show(df_alerts)
    with tab_fin: filter_show(df_alerts[df_alerts["Category"] == "Financial"])
    with tab_ops: filter_show(df_alerts[df_alerts["Category"] == "Operational"])

    st.markdown("---")
    st.subheader("Case Management")
    active = df_alerts[df_alerts["Status"].isin(["Open", "Investigating"])]
    if len(active) > 0:
        sel = st.selectbox("Select alert", active["ID"].tolist())
        al = df_alerts[df_alerts["ID"] == sel].iloc[0]
        st.info(f"**{al['Title']}** — {al['Description']}")
        st.markdown(f"**Standard:** {al['Standard']} | **SLA:** {al['SLA_Hours']}h | **Linked:** {al['Linked']}")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.button("📋 Open Case", use_container_width=True, key="oc")
        bc2.button("🔍 Investigate", use_container_width=True, key="inv_a")
        bc3.button("✅ Resolve", use_container_width=True, key="res_a")
        bc4.button("❌ Dismiss", use_container_width=True, key="dis_a")


# ══════════════════════════════════════════════════════════════
# 8. REPORTS & ANALYTICS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Reports & Analytics":
    st.title("📈 Reports & Analytics")
    st.caption("IFRS financial reporting, profitability analysis, KPI monitoring")

    tab_profit, tab_kpis, tab_reports = st.tabs(["Profitability Analysis", "Financial KPIs", "Reports"])

    with tab_profit:
        total_rev = df_profit["Revenue"].sum()
        total_cost = df_profit["Costs"].sum()
        total_prof = df_profit["Profit"].sum()
        margin = total_prof / total_rev * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Revenue (IFRS 15)", fmt_usd(total_rev), "+8.2%")
        c2.metric("Costs", fmt_usd(total_cost), "-3.1%", delta_color="inverse")
        c3.metric("Net Profit", fmt_usd(total_prof), "+12.4%")
        c4.metric("Margin", f"{margin:.1f}%")

        pt1, pt2, pt3 = st.tabs(["Client", "IB Partners", "Campaigns"])
        with pt1:
            cl, cr = st.columns(2)
            with cl:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Revenue"], name="Revenue", marker_color="#10b981"))
                fig.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Costs"], name="Costs", marker_color="#ef4444"))
                fig.add_trace(go.Bar(x=df_profit["Client"], y=df_profit["Profit"], name="Profit", marker_color="#6366f1"))
                fig.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group")
                st.plotly_chart(fig, use_container_width=True)
            with cr:
                st.dataframe(df_profit.sort_values("Profit", ascending=False), use_container_width=True, hide_index=True, column_config={"Revenue": st.column_config.NumberColumn(format="$%d"), "Costs": st.column_config.NumberColumn(format="$%d"), "Profit": st.column_config.NumberColumn(format="$%d"), "Margin": st.column_config.NumberColumn(format="%.1f%%")})
        with pt2:
            cl2, cr2 = st.columns(2)
            with cl2:
                fig_ib = go.Figure()
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"], y=df_ib["Net_Revenue"], name="Net Revenue", marker_color="#10b981"))
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"], y=df_ib["Commission"], name="Commission", marker_color="#8b5cf6"))
                fig_ib.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group")
                st.plotly_chart(fig_ib, use_container_width=True)
            with cr2:
                st.dataframe(df_ib.sort_values("Net_Revenue", ascending=False), use_container_width=True, hide_index=True, column_config={"Volume": st.column_config.NumberColumn(format="$%d"), "Commission": st.column_config.NumberColumn(format="$%d"), "Net_Revenue": st.column_config.NumberColumn(format="$%d")})
        with pt3:
            campaigns = pd.DataFrame([
                {"Campaign": "Welcome Bonus", "Spend": 45000, "Revenue": 180000, "ROI": 300},
                {"Campaign": "Loyalty Program", "Spend": 22000, "Revenue": 95000, "ROI": 332},
                {"Campaign": "Referral Bonus", "Spend": 15000, "Revenue": 72000, "ROI": 380},
                {"Campaign": "VIP Cashback", "Spend": 35000, "Revenue": 110000, "ROI": 214},
            ])
            cl3, cr3 = st.columns(2)
            with cl3:
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(x=campaigns["Campaign"], y=campaigns["Spend"], name="Spend", marker_color="#ef4444"))
                fig_c.add_trace(go.Bar(x=campaigns["Campaign"], y=campaigns["Revenue"], name="Revenue", marker_color="#10b981"))
                fig_c.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group")
                st.plotly_chart(fig_c, use_container_width=True)
            with cr3:
                st.dataframe(campaigns.sort_values("ROI", ascending=False), use_container_width=True, hide_index=True, column_config={"Spend": st.column_config.NumberColumn(format="$%d"), "Revenue": st.column_config.NumberColumn(format="$%d"), "ROI": st.column_config.ProgressColumn(min_value=0, max_value=400, format="%d%%")})

    with tab_kpis:
        cl, cr = st.columns(2)
        with cl:
            st.subheader("Net Flow Trend")
            fig_nf = px.line(df_kpi, x="Month", y="Net_Flow", markers=True, color_discrete_sequence=["#6366f1"])
            fig_nf.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig_nf, use_container_width=True)
        with cr:
            st.subheader("Operating Costs")
            fig_oc = px.bar(df_kpi, x="Month", y="Op_Costs", color_discrete_sequence=["#ef4444"])
            fig_oc.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
            st.plotly_chart(fig_oc, use_container_width=True)

        cl2, cr2 = st.columns(2)
        with cl2:
            st.subheader("Reconciliation Rate")
            fig_rr = px.line(df_kpi, x="Month", y="Recon_Rate", markers=True, color_discrete_sequence=["#10b981"])
            fig_rr.add_hline(y=95, line_dash="dash", line_color="#f59e0b", annotation_text="95% Target")
            fig_rr.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig_rr, use_container_width=True)
        with cr2:
            st.subheader("Exception Trend")
            fig_ex = px.line(df_kpi, x="Month", y="Exceptions", markers=True, color_discrete_sequence=["#f59e0b"])
            fig_ex.update_layout(**PLOTLY_LAYOUT, height=260)
            st.plotly_chart(fig_ex, use_container_width=True)

        st.subheader("Monthly KPI Summary")
        kpi_d = df_kpi.copy()
        kpi_d["Cost_Ratio"] = (kpi_d["Op_Costs"] / kpi_d["Net_Flow"] * 100).round(1).astype(str) + "%"
        st.dataframe(kpi_d, use_container_width=True, hide_index=True, column_config={"Net_Flow": st.column_config.NumberColumn(format="$%d"), "Op_Costs": st.column_config.NumberColumn(format="$%d"), "Recon_Rate": st.column_config.NumberColumn(format="%.1f%%")})

    with tab_reports:
        st.subheader("Report Templates")
        for name, desc, freq in [
            ("📅 Daily Finance Report (IAS 1)", "End-of-day P&L, cash position, reconciliation status", "Daily @ 01:00 UTC"),
            ("📋 Weekly Control Report (SOX 302)", "Internal controls attestation, exception analysis", "Monday @ 06:00 UTC"),
            ("📊 Monthly Financial Statements (IFRS)", "Balance sheet, income statement, cash flow per IAS 1/7", "1st of month"),
            ("🏛️ Regulatory Report (Basel III)", "LCR, NSFR, HQLA composition for supervisor", "Monthly"),
        ]:
            rc1, rc2 = st.columns([3, 1])
            with rc1:
                st.markdown(f"**{name}**")
                st.caption(f"{desc} — {freq}")
            with rc2:
                st.button("📥 Generate", key=f"gen_{name}", use_container_width=True)
            st.markdown("---")

        st.subheader("Recent Reports")
        st.dataframe(pd.DataFrame([
            {"Report": "Daily Finance Report", "Period": "May 17, 2026", "Standard": "IAS 1", "Status": "Ready"},
            {"Report": "Weekly Control Report", "Period": "May 12–18, 2026", "Standard": "SOX 302", "Status": "Ready"},
            {"Report": "Monthly Financial Statements", "Period": "April 2026", "Standard": "IFRS", "Status": "Ready"},
            {"Report": "Basel III Regulatory Report", "Period": "Q1 2026", "Standard": "Basel III", "Status": "Ready"},
            {"Report": "Daily Finance Report", "Period": "May 18, 2026", "Standard": "IAS 1", "Status": "Scheduled"},
        ]), use_container_width=True, hide_index=True)
