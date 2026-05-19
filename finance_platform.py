"""
FinanceOps — Financial Control & Intelligence Platform v3
All improvements: Auto-refresh, Theme toggle, Export, Date range, Notifications,
Client 360, PSP Scorecard, Cash Forecast, Audit Log, Multi-currency, Comparison, Comments
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io, json

st.set_page_config(page_title="FinanceOps", page_icon="F", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ══════════════════════════════════════════════
if "theme" not in st.session_state: st.session_state.theme = "dark"
if "base_ccy" not in st.session_state: st.session_state.base_ccy = "USD"
if "comments" not in st.session_state: st.session_state.comments = {}
if "notifications_read" not in st.session_state: st.session_state.notifications_read = set()
if "audit_log" not in st.session_state:
    st.session_state.audit_log = [
        {"Time": "2026-05-18 11:15", "User": "System", "Action": "Auto-matched REC-001", "Section": "Reconciliation"},
        {"Time": "2026-05-18 10:30", "User": "Ahmed K.", "Action": "Viewed Dashboard", "Section": "Dashboard"},
        {"Time": "2026-05-18 10:15", "User": "System", "Action": "Alert ALT-001 created", "Section": "Alerts"},
        {"Time": "2026-05-18 09:45", "User": "Sarah M.", "Action": "Exported Daily Report", "Section": "Reports"},
        {"Time": "2026-05-18 09:30", "User": "System", "Action": "Alert ALT-002 created", "Section": "Alerts"},
        {"Time": "2026-05-18 09:00", "User": "Omar R.", "Action": "Investigated REC-003", "Section": "Reconciliation"},
        {"Time": "2026-05-18 08:30", "User": "Lina T.", "Action": "Approved REC-008", "Section": "Reconciliation"},
        {"Time": "2026-05-18 08:00", "User": "System", "Action": "PSP Settlement processed", "Section": "Transactions"},
        {"Time": "2026-05-18 07:00", "User": "System", "Action": "Alert ALT-006 — LCR breach", "Section": "Alerts"},
        {"Time": "2026-05-17 23:00", "User": "System", "Action": "Daily reconciliation completed", "Section": "Reconciliation"},
    ]

# ══════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════
dark = st.session_state.theme == "dark"
BG = "#0f1729" if dark else "#ffffff"
BG2 = "#1e2d4a" if dark else "#e2e8f0"
TXT = "#e2e8f0" if dark else "#1e293b"
TXT2 = "#94a3b8" if dark else "#64748b"
CARD_BG = "#0f1729" if dark else "#f8fafc"
CARD_BD = "#1e2d4a" if dark else "#e2e8f0"

st.markdown(f"""<style>
.main .block-container{{padding-top:1rem}}
div[data-testid="stMetric"]{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:10px;padding:14px 18px}}
div[data-testid="stMetric"] label{{font-size:11px!important;text-transform:uppercase;letter-spacing:.5px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}}
.bg-green{{background:rgba(16,185,129,.12);color:#10b981}}.bg-red{{background:rgba(239,68,68,.12);color:#ef4444}}
.bg-yellow{{background:rgba(245,158,11,.12);color:#f59e0b}}.bg-blue{{background:rgba(59,130,246,.12);color:#3b82f6}}
.bg-purple{{background:rgba(139,92,246,.12);color:#8b5cf6}}.bg-cyan{{background:rgba(6,182,212,.12);color:#06b6d4}}
.card{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:10px;padding:16px 18px;margin-bottom:8px}}
.card h4{{font-size:14px;font-weight:700;margin-bottom:8px;color:{TXT}}}
.sm{{font-size:12px;color:{TXT2}}}
.alert-card{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:8px;padding:14px 18px;margin-bottom:8px}}
.alert-title{{font-weight:600;font-size:13px;margin-bottom:3px;color:{TXT}}}
.alert-desc{{font-size:12px;color:{TXT2};line-height:1.5}}
.risk-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:{CARD_BG};border:1px solid {CARD_BD};border-radius:8px;margin-bottom:6px}}
.risk-label{{font-size:13px;color:{TXT}}}.risk-val{{font-weight:700;font-size:13px}}
.flow-arrow{{text-align:center;font-size:20px;color:#6366f1;margin:4px 0}}
.metric-box{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:8px;padding:12px;text-align:center}}
.metric-box .val{{font-size:20px;font-weight:700;color:{TXT}}}.metric-box .lbl{{font-size:10px;color:{TXT2};text-transform:uppercase;letter-spacing:.5px}}
.status-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.dot-g{{background:#10b981}}.dot-y{{background:#f59e0b}}.dot-r{{background:#ef4444}}
.notif-badge{{background:#ef4444;color:#fff;border-radius:50%;font-size:10px;font-weight:700;padding:1px 5px;position:relative;top:-8px}}
.comment-box{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:6px;padding:10px 14px;margin:4px 0;font-size:12px}}
div[data-testid="stSidebar"]>div:first-child{{padding-top:.8rem}}
</style>""", unsafe_allow_html=True)

PLT = "plotly_dark" if dark else "plotly_white"
PL = dict(template=PLT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0))

# ══════════════════════════════════════════════
# FX RATES & HELPERS
# ══════════════════════════════════════════════
FX = {"USD":1.0,"EUR":1.0845,"GBP":1.2630,"CHF":1.1175,"JPY":0.00645}

def to_base(amount, ccy):
    base = st.session_state.base_ccy
    usd_val = amount * FX.get(ccy, 1.0)
    if base == "USD": return usd_val
    return usd_val / FX.get(base, 1.0)

def fmt(n):
    sym = {"USD":"$","EUR":"€","GBP":"£","CHF":"CHF ","JPY":"¥"}.get(st.session_state.base_ccy, "$")
    return f"{sym}{n:,.0f}"

def fmt_usd(n): return f"${n:,.0f}"

def badge(t, c="blue"): return f'<span class="badge bg-{c}">{t}</span>'
def sev_icon(s): return {"Critical":"[!]","High":"[!]","Medium":"[~]","Low":"[i]"}.get(s,"[-]")

def export_df(df, name, key):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"Export {name}", csv, f"{name.lower().replace(' ','_')}.csv", "text/csv", key=key)

def add_audit(user, action, section):
    st.session_state.audit_log.insert(0, {"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "User": user, "Action": action, "Section": section})

def add_comment(case_id, comment_text, user="Current User"):
    if case_id not in st.session_state.comments:
        st.session_state.comments[case_id] = []
    st.session_state.comments[case_id].append({"user": user, "text": comment_text, "time": datetime.now().strftime("%H:%M")})

def show_comments(case_id):
    comments = st.session_state.comments.get(case_id, [])
    if comments:
        for c in comments:
            st.markdown(f'<div class="comment-box"><b>{c["user"]}</b> <span class="sm">({c["time"]})</span><br>{c["text"]}</div>', unsafe_allow_html=True)
    new = st.text_input(f"Add comment to {case_id}", key=f"cmt_{case_id}", placeholder="Type a comment...")
    if new:
        add_comment(case_id, new)
        add_audit("Current User", f"Commented on {case_id}", "Comments")
        st.rerun()

# ══════════════════════════════════════════════
# MOCK DATA
# ══════════════════════════════════════════════
transactions_data = [
    {"ID":"TXN-001","Type":"Deposit","Source":"CRM","Amount":25000,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Client deposit"},
    {"ID":"TXN-002","Type":"Withdrawal","Source":"CRM","Amount":12500,"CCY":"USD","Status":"Settled","Client":"Globe Ltd","Counterparty":"Adyen","Bank":"HSBC","Timestamp":"2026-05-18 09:15","Value_Date":"2026-05-18","Description":"Client withdrawal"},
    {"ID":"TXN-003","Type":"Deposit","Source":"PSP","Amount":8700,"CCY":"EUR","Status":"Pending","Client":"NovaTech","Counterparty":"Stripe","Bank":"Deutsche Bank","Timestamp":"2026-05-18 09:42","Value_Date":"2026-05-18","Description":"PSP settlement T+1"},
    {"ID":"TXN-004","Type":"Transfer","Source":"Bank","Amount":50000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"JPMorgan","Timestamp":"2026-05-18 10:01","Value_Date":"2026-05-18","Description":"Treasury rebalancing"},
    {"ID":"TXN-005","Type":"Fee","Source":"PSP","Amount":125,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Processing fee"},
    {"ID":"TXN-006","Type":"Commission","Source":"Commission","Amount":375,"CCY":"USD","Status":"Settled","Client":"IB-Alpha","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 10:30","Value_Date":"2026-05-18","Description":"IB commission"},
    {"ID":"TXN-007","Type":"Deposit","Source":"CRM","Amount":150000,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Worldpay","Bank":"Barclays","Timestamp":"2026-05-17 14:22","Value_Date":"2026-05-17","Description":"High-value deposit"},
    {"ID":"TXN-008","Type":"Withdrawal","Source":"CRM","Amount":45000,"CCY":"GBP","Status":"Failed","Client":"BritCo","Counterparty":"Adyen","Bank":"Barclays","Timestamp":"2026-05-17 16:45","Value_Date":"2026-05-17","Description":"Failed — insufficient PSP"},
    {"ID":"TXN-009","Type":"Deposit","Source":"PSP","Amount":32000,"CCY":"USD","Status":"Settled","Client":"SolarInc","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-17 11:10","Value_Date":"2026-05-17","Description":"PSP settlement"},
    {"ID":"TXN-010","Type":"Withdrawal","Source":"CRM","Amount":18200,"CCY":"EUR","Status":"Pending","Client":"EuroTrade","Counterparty":"Adyen","Bank":"Deutsche Bank","Timestamp":"2026-05-18 07:50","Value_Date":"2026-05-18","Description":"Pending withdrawal"},
    {"ID":"TXN-011","Type":"Fee","Source":"PSP","Amount":89,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Worldpay","Bank":"Barclays","Timestamp":"2026-05-17 14:22","Value_Date":"2026-05-17","Description":"Processing fee"},
    {"ID":"TXN-012","Type":"Deposit","Source":"CRM","Amount":5600,"CCY":"USD","Status":"Reversed","Client":"QuickPay","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-16 09:30","Value_Date":"2026-05-16","Description":"Chargeback"},
    {"ID":"TXN-013","Type":"Transfer","Source":"Bank","Amount":200000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"JPMorgan","Timestamp":"2026-05-16 15:00","Value_Date":"2026-05-16","Description":"Liquidity rebalancing"},
    {"ID":"TXN-014","Type":"Commission","Source":"Commission","Amount":1250,"CCY":"USD","Status":"Settled","Client":"IB-Beta","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-16 16:00","Value_Date":"2026-05-16","Description":"Monthly IB commission"},
    {"ID":"TXN-015","Type":"Deposit","Source":"CRM","Amount":72000,"CCY":"USD","Status":"Settled","Client":"TradeCo","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-15 10:20","Value_Date":"2026-05-15","Description":"Client deposit"},
    {"ID":"TXN-016","Type":"Bonus","Source":"Bonus","Amount":15000,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Bonus Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 11:00","Value_Date":"2026-05-18","Description":"Welcome bonus"},
    {"ID":"TXN-017","Type":"Bonus","Source":"Bonus","Amount":5000,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Bonus Engine","Bank":"Barclays","Timestamp":"2026-05-17 09:00","Value_Date":"2026-05-17","Description":"Loyalty bonus"},
    {"ID":"TXN-018","Type":"Deposit","Source":"PSP","Amount":42000,"CCY":"EUR","Status":"Settled","Client":"EuroTrade","Counterparty":"Adyen","Bank":"Deutsche Bank","Timestamp":"2026-05-18 06:30","Value_Date":"2026-05-18","Description":"PSP settlement"},
    {"ID":"TXN-019","Type":"Transfer","Source":"Bank","Amount":85000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"HSBC","Timestamp":"2026-05-18 08:00","Value_Date":"2026-05-18","Description":"Inter-bank transfer"},
    {"ID":"TXN-020","Type":"Commission","Source":"Commission","Amount":2100,"CCY":"USD","Status":"Pending","Client":"IB-Gamma","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 11:15","Value_Date":"2026-05-18","Description":"Weekly accrual"},
]

reconciliation_data = [
    {"Case_ID":"REC-001","TXN_ID":"TXN-001","CRM_Amt":25000,"PSP_Amt":25000,"Bank_Amt":25000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-002","TXN_ID":"TXN-003","CRM_Amt":8700,"PSP_Amt":8700,"Bank_Amt":9000,"CCY":"EUR","Status":"Partial","Score":72,"Case_Status":"Investigating","Method":"ID+Time","Notes":"€300 variance — FX suspected"},
    {"Case_ID":"REC-003","TXN_ID":"TXN-008","CRM_Amt":45000,"PSP_Amt":45000,"Bank_Amt":0,"CCY":"GBP","Status":"Exception","Score":0,"Case_Status":"Open","Method":"ID Match","Notes":"No bank record"},
    {"Case_ID":"REC-004","TXN_ID":"TXN-007","CRM_Amt":150000,"PSP_Amt":150000,"Bank_Amt":150000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-005","TXN_ID":"TXN-012","CRM_Amt":5600,"PSP_Amt":5600,"Bank_Amt":5600,"CCY":"USD","Status":"Exception","Score":45,"Case_Status":"Investigating","Method":"Amount+Time","Notes":"Chargeback pending"},
    {"Case_ID":"REC-006","TXN_ID":"TXN-009","CRM_Amt":32000,"PSP_Amt":32000,"Bank_Amt":32000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-007","TXN_ID":"TXN-010","CRM_Amt":18200,"PSP_Amt":18200,"Bank_Amt":18500,"CCY":"EUR","Status":"Partial","Score":68,"Case_Status":"Open","Method":"ID+Time","Notes":"€300 discrepancy"},
    {"Case_ID":"REC-008","TXN_ID":"TXN-002","CRM_Amt":12500,"PSP_Amt":12500,"Bank_Amt":12500,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-009","TXN_ID":"TXN-015","CRM_Amt":72000,"PSP_Amt":72000,"Bank_Amt":72000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-010","TXN_ID":"TXN-018","CRM_Amt":42000,"PSP_Amt":42000,"Bank_Amt":42000,"CCY":"EUR","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount+FX","Notes":"EUR matched"},
]

ledger_data = [
    {"Date":"2026-05-18","Account":"Cash Account","Debit":25000,"Credit":0,"CCY":"USD","Ref":"TXN-001","Narration":"Deposit — Acme Corp"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":25000,"CCY":"USD","Ref":"TXN-001","Narration":"Client funds credited"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":12500,"Credit":0,"CCY":"USD","Ref":"TXN-002","Narration":"Withdrawal — Globe Ltd"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":12500,"CCY":"USD","Ref":"TXN-002","Narration":"Cash disbursed"},
    {"Date":"2026-05-18","Account":"PSP Account","Debit":8700,"Credit":0,"CCY":"EUR","Ref":"TXN-003","Narration":"PSP receivable — Stripe"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":8700,"CCY":"EUR","Ref":"TXN-003","Narration":"Client funds — NovaTech"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":50000,"Credit":0,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer in"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":50000,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer out"},
    {"Date":"2026-05-18","Account":"Fee Account","Debit":0,"Credit":125,"CCY":"USD","Ref":"TXN-005","Narration":"Fee revenue"},
    {"Date":"2026-05-18","Account":"Commission Account","Debit":375,"Credit":0,"CCY":"USD","Ref":"TXN-006","Narration":"IB commission expense"},
    {"Date":"2026-05-18","Account":"Commission Account","Debit":0,"Credit":375,"CCY":"USD","Ref":"TXN-006","Narration":"Commission payable"},
    {"Date":"2026-05-17","Account":"Cash Account","Debit":150000,"Credit":0,"CCY":"USD","Ref":"TXN-007","Narration":"Deposit — MegaFund"},
    {"Date":"2026-05-17","Account":"Client Liability","Debit":0,"Credit":150000,"CCY":"USD","Ref":"TXN-007","Narration":"Client funds — MegaFund"},
    {"Date":"2026-05-17","Account":"Cash Account","Debit":32000,"Credit":0,"CCY":"USD","Ref":"TXN-009","Narration":"PSP settlement — SolarInc"},
    {"Date":"2026-05-17","Account":"Client Liability","Debit":0,"Credit":32000,"CCY":"USD","Ref":"TXN-009","Narration":"Client funds — SolarInc"},
]

alerts_data = [
    {"ID":"ALT-001","Title":"Cash Position Imbalance","Category":"Financial","Severity":"Critical","Status":"Open","Description":"GL cash deviates from bank by $12,400.","SLA":"4h","Linked":"TXN-004","Created":"2026-05-18 10:15"},
    {"ID":"ALT-002","Title":"Withdrawal Spike","Category":"Financial","Severity":"High","Status":"Investigating","Description":"Withdrawal volume 340% above average.","SLA":"8h","Linked":"TXN-002, TXN-010","Created":"2026-05-18 09:30"},
    {"ID":"ALT-003","Title":"PSP Delay — Adyen","Category":"Operational","Severity":"Medium","Status":"Open","Description":"Adyen batch delayed 4h. Pending: €63,200.","SLA":"12h","Linked":"TXN-010","Created":"2026-05-18 08:00"},
    {"ID":"ALT-004","Title":"Recon Exception Rate 20%","Category":"Operational","Severity":"High","Status":"Investigating","Description":"Above 5% threshold.","SLA":"4h","Linked":"REC-003, REC-005","Created":"2026-05-18 10:00"},
    {"ID":"ALT-005","Title":"Chargeback Provision","Category":"Financial","Severity":"Medium","Status":"Resolved","Description":"TXN-012 chargeback booked.","SLA":"24h","Linked":"TXN-012","Created":"2026-05-16 10:30"},
    {"ID":"ALT-006","Title":"Liquidity Buffer Low","Category":"Financial","Severity":"Critical","Status":"Open","Description":"Buffer at 11.2% — below 15%.","SLA":"2h","Linked":"","Created":"2026-05-18 07:00"},
    {"ID":"ALT-007","Title":"Unexplained Transaction","Category":"Compliance","Severity":"High","Status":"Open","Description":"$35K credit with no CRM match.","SLA":"4h","Linked":"","Created":"2026-05-18 11:00"},
]

bank_balances = [
    {"Bank":"JPMorgan","CCY":"USD","Balance":2450000},{"Bank":"HSBC","CCY":"USD","Balance":890000},
    {"Bank":"Deutsche Bank","CCY":"EUR","Balance":1250000},{"Bank":"Barclays","CCY":"GBP","Balance":675000},
    {"Bank":"Barclays","CCY":"USD","Balance":320000},
]

psp_balances = [
    {"PSP":"Stripe","CCY":"USD","Balance":185000,"Pending_In":32000,"Pending_Out":8500,"Success_Rate":99.2,"Avg_Settlement":"4.2h","Cost_Per_Txn":0.52},
    {"PSP":"Stripe","CCY":"EUR","Balance":45000,"Pending_In":8700,"Pending_Out":0,"Success_Rate":98.8,"Avg_Settlement":"5.1h","Cost_Per_Txn":0.48},
    {"PSP":"Adyen","CCY":"USD","Balance":92000,"Pending_In":0,"Pending_Out":12500,"Success_Rate":97.5,"Avg_Settlement":"6.8h","Cost_Per_Txn":0.61},
    {"PSP":"Adyen","CCY":"EUR","Balance":63200,"Pending_In":0,"Pending_Out":18200,"Success_Rate":96.1,"Avg_Settlement":"8.2h","Cost_Per_Txn":0.58},
    {"PSP":"Worldpay","CCY":"USD","Balance":210000,"Pending_In":0,"Pending_Out":0,"Success_Rate":99.5,"Avg_Settlement":"3.5h","Cost_Per_Txn":0.45},
    {"PSP":"Worldpay","CCY":"GBP","Balance":55000,"Pending_In":0,"Pending_Out":45000,"Success_Rate":98.0,"Avg_Settlement":"5.0h","Cost_Per_Txn":0.55},
]

cash_flow_this = [
    {"Date":"May 12","Deposits":185000,"Withdrawals":72000,"Net":113000},
    {"Date":"May 13","Deposits":210000,"Withdrawals":95000,"Net":115000},
    {"Date":"May 14","Deposits":145000,"Withdrawals":120000,"Net":25000},
    {"Date":"May 15","Deposits":290000,"Withdrawals":88000,"Net":202000},
    {"Date":"May 16","Deposits":178000,"Withdrawals":156000,"Net":22000},
    {"Date":"May 17","Deposits":320000,"Withdrawals":105000,"Net":215000},
    {"Date":"May 18","Deposits":307700,"Withdrawals":75700,"Net":232000},
]
cash_flow_last = [
    {"Date":"May 5","Deposits":162000,"Withdrawals":68000,"Net":94000},
    {"Date":"May 6","Deposits":195000,"Withdrawals":82000,"Net":113000},
    {"Date":"May 7","Deposits":130000,"Withdrawals":105000,"Net":25000},
    {"Date":"May 8","Deposits":250000,"Withdrawals":78000,"Net":172000},
    {"Date":"May 9","Deposits":160000,"Withdrawals":140000,"Net":20000},
    {"Date":"May 10","Deposits":280000,"Withdrawals":92000,"Net":188000},
    {"Date":"May 11","Deposits":275000,"Withdrawals":70000,"Net":205000},
]

profitability = [
    {"Client":"Acme Corp","Revenue":45000,"Costs":12000,"Profit":33000,"Deposits":6,"Withdrawals":1,"Since":"2024-03-15"},
    {"Client":"Globe Ltd","Revenue":28000,"Costs":9500,"Profit":18500,"Deposits":3,"Withdrawals":2,"Since":"2024-06-20"},
    {"Client":"MegaFund","Revenue":82000,"Costs":18000,"Profit":64000,"Deposits":4,"Withdrawals":0,"Since":"2023-11-01"},
    {"Client":"NovaTech","Revenue":15000,"Costs":6200,"Profit":8800,"Deposits":2,"Withdrawals":0,"Since":"2025-01-10"},
    {"Client":"SolarInc","Revenue":32000,"Costs":8800,"Profit":23200,"Deposits":3,"Withdrawals":0,"Since":"2024-09-05"},
    {"Client":"TradeCo","Revenue":51000,"Costs":14500,"Profit":36500,"Deposits":5,"Withdrawals":1,"Since":"2024-01-22"},
]

ib_data = [
    {"Partner":"IB-Alpha","Clients":42,"Volume":1250000,"Commission":18750,"Net_Revenue":62500},
    {"Partner":"IB-Beta","Clients":28,"Volume":890000,"Commission":13350,"Net_Revenue":44500},
    {"Partner":"IB-Gamma","Clients":65,"Volume":2100000,"Commission":31500,"Net_Revenue":105000},
    {"Partner":"IB-Delta","Clients":15,"Volume":420000,"Commission":6300,"Net_Revenue":21000},
]

monthly_kpis = [
    {"Month":"Jan","Net_Flow":1200000,"Op_Costs":85000,"Exceptions":12,"Recon_Rate":94.2},
    {"Month":"Feb","Net_Flow":980000,"Op_Costs":78000,"Exceptions":8,"Recon_Rate":96.1},
    {"Month":"Mar","Net_Flow":1450000,"Op_Costs":92000,"Exceptions":15,"Recon_Rate":93.8},
    {"Month":"Apr","Net_Flow":1100000,"Op_Costs":88000,"Exceptions":10,"Recon_Rate":95.5},
    {"Month":"May","Net_Flow":1680000,"Op_Costs":95000,"Exceptions":18,"Recon_Rate":91.0},
]

# DataFrames
df_txn = pd.DataFrame(transactions_data)
df_rec = pd.DataFrame(reconciliation_data)
df_led = pd.DataFrame(ledger_data)
df_alerts = pd.DataFrame(alerts_data)
df_bank = pd.DataFrame(bank_balances)
df_psp = pd.DataFrame(psp_balances)
df_cash = pd.DataFrame(cash_flow_this)
df_cash_last = pd.DataFrame(cash_flow_last)
df_profit = pd.DataFrame(profitability)
df_ib = pd.DataFrame(ib_data)
df_kpi = pd.DataFrame(monthly_kpis)

# Computed
opening_balance = 4250000
today = "2026-05-18"
cash_in = df_txn[(df_txn["Type"]=="Deposit")&(df_txn["Value_Date"]==today)&(df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
cash_out = df_txn[(df_txn["Type"].isin(["Withdrawal","Fee","Commission"]))&(df_txn["Value_Date"]==today)&(df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
net_flow = cash_in - cash_out
total_bank = df_bank["Balance"].sum()
total_psp = df_psp["Balance"].sum()
available_cash = total_bank + total_psp
pending_wd = df_txn[(df_txn["Type"]=="Withdrawal")&(df_txn["Status"]=="Pending")]["Amount"].sum() + df_psp["Pending_Out"].sum()
bonus_liability = 125000; commission_liability = 48500
net_liquidity = available_cash - pending_wd - bonus_liability - commission_liability
buffer_pct = (net_liquidity / available_cash * 100) if available_cash > 0 else 0
matched_count = len(df_rec[df_rec["Status"]=="Matched"])
recon_rate = matched_count / len(df_rec) * 100
unmatched = len(df_rec[df_rec["Status"].isin(["Partial","Exception"])])
open_alerts = len(df_alerts[df_alerts["Status"].isin(["Open","Investigating"])])

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### **FinanceOps**")
    st.caption("Financial Control & Intelligence")
    st.markdown("---")

    # Theme toggle
    theme_label = "Dark" if dark else "Light"
    if st.button(f"Theme: {theme_label}", use_container_width=True):
        st.session_state.theme = "light" if dark else "dark"
        st.rerun()

    # Multi-currency
    st.session_state.base_ccy = st.selectbox("Base Currency", ["USD","EUR","GBP"], index=["USD","EUR","GBP"].index(st.session_state.base_ccy))

    # Auto-refresh
    refresh = st.selectbox("Auto-Refresh", ["Off","30s","60s","5m"], index=0)
    if refresh != "Off":
        secs = {"30s":30,"60s":60,"5m":300}[refresh]
        st.caption(f"Refreshing every {refresh}")
        import time
        st_autorefresh = st.empty()

    st.markdown("---")

    # Notification bell
    unread = open_alerts - len(st.session_state.notifications_read)
    notif_label = f"Notifications ({unread})" if unread > 0 else "Notifications"

    page = st.radio("", [
        "Dashboard",
        notif_label,
        "Client 360",
        "Transactions",
        "Reconciliation",
        "Ledger",
        "Liquidity",
        "PSP Scorecard",
        f"Alerts ({open_alerts})",
        "Reports",
        "Cash Forecast",
        "Risk Monitor",
        "Audit Log",
        "Architecture",
        "Integrations",
        "Settings",
        "File Upload",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**System Online**")
    st.caption(datetime.now().strftime("%a, %b %d, %Y — %H:%M"))


# ══════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════

# ──────────── DASHBOARD ────────────
if page == "Dashboard":
    st.title("Dashboard — Financial Control Tower")
    role = st.radio("View as:", ["CFO","Finance Manager","Operations"], horizontal=True)

    # Comparison toggle
    compare = st.toggle("Compare with last week", value=False)

    st.subheader("Financial Overview")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Opening Balance", fmt(opening_balance))
    f2.metric("Cash In Today", fmt(cash_in), f"+{cash_in/opening_balance*100:.1f}%")
    f3.metric("Cash Out Today", fmt(cash_out), f"-{cash_out/opening_balance*100:.1f}%", delta_color="inverse")
    f4.metric("Net Flow", fmt(net_flow))

    if role in ["CFO","Finance Manager"]:
        st.subheader("Liquidity")
        l1,l2,l3,l4 = st.columns(4)
        l1.metric("Available Cash", fmt(available_cash))
        l2.metric("Pending WD", fmt(pending_wd), delta_color="inverse")
        l3.metric("PSP Balances", fmt(total_psp))
        l4.metric("Net Liquidity", fmt(net_liquidity), f"Buffer: {buffer_pct:.1f}%")

    # Charts with comparison
    cl,cr = st.columns(2)
    with cl:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Deposits"],mode="lines",name="Cash In",fill="tozeroy",line=dict(color="#10b981",width=2),fillcolor="rgba(16,185,129,.12)"))
        fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Withdrawals"],mode="lines",name="Cash Out",fill="tozeroy",line=dict(color="#ef4444",width=2),fillcolor="rgba(239,68,68,.12)"))
        if compare:
            fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash_last["Deposits"].values,mode="lines",name="Last Wk In",line=dict(color="#10b981",width=1,dash="dot")))
            fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash_last["Withdrawals"].values,mode="lines",name="Last Wk Out",line=dict(color="#ef4444",width=1,dash="dot")))
        fig.update_layout(**PL,height=260,legend=dict(orientation="h",y=1.15))
        st.plotly_chart(fig,use_container_width=True)
    with cr:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=df_cash["Date"],y=df_cash["Net"],name="This Week",marker_color="#6366f1"))
        if compare:
            fig2.add_trace(go.Bar(x=df_cash["Date"],y=df_cash_last["Net"].values,name="Last Week",marker_color="#94a3b8"))
        fig2.update_layout(**PL,height=260,barmode="group",showlegend=compare)
        st.plotly_chart(fig2,use_container_width=True)

    # Quick charts
    dc1,dc2,dc3 = st.columns(3)
    with dc1:
        tc = df_txn["Type"].value_counts().reset_index(); tc.columns=["Type","Count"]
        fig_m = px.pie(tc,values="Count",names="Type",hole=0.5,color_discrete_sequence=["#10b981","#ef4444","#6366f1","#f59e0b","#8b5cf6","#06b6d4"])
        fig_m.update_layout(**PL,height=200); fig_m.update_traces(textinfo="value+percent")
        st.caption("Transaction Mix"); st.plotly_chart(fig_m,use_container_width=True)
    with dc2:
        sc = df_txn["Status"].value_counts().reset_index(); sc.columns=["Status","Count"]
        fig_s = px.pie(sc,values="Count",names="Status",hole=0.5,color_discrete_map={"Settled":"#10b981","Pending":"#f59e0b","Failed":"#ef4444","Reversed":"#8b5cf6"})
        fig_s.update_layout(**PL,height=200); fig_s.update_traces(textinfo="value+percent")
        st.caption("Status"); st.plotly_chart(fig_s,use_container_width=True)
    with dc3:
        pb = df_psp.groupby("PSP")["Balance"].sum().reset_index()
        fig_p = px.bar(pb,x="PSP",y="Balance",color="PSP",color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig_p.update_layout(**PL,height=200,showlegend=False)
        st.caption("PSP Balances"); st.plotly_chart(fig_p,use_container_width=True)

    st.subheader("Alerts")
    a1,a2,a3 = st.columns(3)
    a1.metric("Exceptions", len(df_rec[df_rec["Status"]=="Exception"]), delta_color="inverse")
    a2.metric("Delays", len(df_alerts[df_alerts["Title"].str.contains("Delay",case=False)&(df_alerts["Status"]!="Resolved")]), delta_color="inverse")
    a3.metric("Liquidity Warnings", len(df_alerts[df_alerts["Title"].str.contains("Liquidity|Buffer",case=False)&(df_alerts["Status"]!="Resolved")]), delta_color="inverse")

    for _,al in df_alerts[df_alerts["Status"].isin(["Open","Investigating"])].iterrows():
        st.markdown(f'<div class="alert-card"><div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow")}</div></div>',unsafe_allow_html=True)

    st.subheader("KPIs")
    k1,k2,k3 = st.columns(3)
    k1.metric("Daily Volume", f"{len(df_txn[df_txn['Value_Date']==today])} txns")
    k2.metric("Recon Rate", f"{recon_rate:.1f}%", f"{matched_count}/{len(df_rec)}")
    k3.metric("Unmatched", unmatched, delta_color="inverse")

    if role == "CFO":
        st.subheader("Financial Intelligence")
        ci1,ci2 = st.columns(2)
        with ci1:
            st.markdown("**Top Profitable Clients**")
            for _,r in df_profit.nlargest(3,"Profit").iterrows():
                st.markdown(f'<div class="risk-row"><span class="risk-label">{r["Client"]}</span><span class="risk-val" style="color:#10b981">{fmt_usd(r["Profit"])}</span></div>',unsafe_allow_html=True)
        with ci2:
            st.markdown("**Top Performing IBs**")
            for _,r in df_ib.nlargest(3,"Net_Revenue").iterrows():
                st.markdown(f'<div class="risk-row"><span class="risk-label">{r["Partner"]}</span><span class="risk-val" style="color:#6366f1">{fmt_usd(r["Net_Revenue"])}</span></div>',unsafe_allow_html=True)


# ──────────── NOTIFICATIONS ────────────
elif "Notifications" in page:
    st.title("Notification Center")
    for _,al in df_alerts.iterrows():
        is_read = al["ID"] in st.session_state.notifications_read
        icon = "NEW" if not is_read else "READ"
        st.markdown(f'<div class="alert-card" style="opacity:{"0.6" if is_read else "1"}"><div class="alert-title">{icon} {al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Status"],"yellow" if al["Status"]!="Resolved" else "green")} <span class="sm">{al["Created"]}</span></div></div>',unsafe_allow_html=True)
        if not is_read:
            if st.button(f"Mark as read", key=f"read_{al['ID']}"):
                st.session_state.notifications_read.add(al["ID"])
                st.rerun()
    if st.button("Mark all as read", use_container_width=True):
        for _,al in df_alerts.iterrows():
            st.session_state.notifications_read.add(al["ID"])
        st.rerun()


# ──────────── CLIENT 360 ────────────
elif page == "Client 360":
    st.title("Client 360 View")
    clients = sorted(df_txn[df_txn["Client"]!="Internal"]["Client"].unique().tolist())
    sel_client = st.selectbox("Select Client", clients)
    client_txns = df_txn[df_txn["Client"]==sel_client]
    client_profit = df_profit[df_profit["Client"]==sel_client]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Transactions", len(client_txns))
    c2.metric("Total Volume", fmt_usd(client_txns["Amount"].sum()))
    if len(client_profit) > 0:
        cp = client_profit.iloc[0]
        c3.metric("Revenue", fmt_usd(cp["Revenue"]))
        c4.metric("Profit", fmt_usd(cp["Profit"]), f"Margin: {cp['Profit']/cp['Revenue']*100:.0f}%")
    else:
        c3.metric("Revenue", "N/A"); c4.metric("Profit", "N/A")

    cl,cr = st.columns(2)
    with cl:
        st.subheader("Transaction History")
        st.dataframe(client_txns[["ID","Type","Amount","CCY","Status","Timestamp"]],use_container_width=True,hide_index=True)
        export_df(client_txns, f"{sel_client} Transactions", f"exp_c360_{sel_client}")
    with cr:
        st.subheader("Activity by Type")
        ct = client_txns["Type"].value_counts().reset_index(); ct.columns=["Type","Count"]
        fig_ct = px.pie(ct,values="Count",names="Type",hole=0.5,color_discrete_sequence=["#10b981","#ef4444","#6366f1","#f59e0b","#8b5cf6"])
        fig_ct.update_layout(**PL,height=250)
        st.plotly_chart(fig_ct,use_container_width=True)

    # Volume over time
    if len(client_txns) > 1:
        vol_time = client_txns.groupby("Value_Date")["Amount"].sum().reset_index()
        fig_vt = px.bar(vol_time,x="Value_Date",y="Amount",color_discrete_sequence=["#6366f1"])
        fig_vt.update_layout(**PL,height=200,title="Volume Over Time",showlegend=False)
        st.plotly_chart(fig_vt,use_container_width=True)

    # Reconciliation status for this client
    client_rec = df_rec[df_rec["TXN_ID"].isin(client_txns["ID"])]
    if len(client_rec) > 0:
        st.subheader("Reconciliation Status")
        st.dataframe(client_rec[["Case_ID","TXN_ID","Status","Score","Case_Status","Notes"]],use_container_width=True,hide_index=True)

    # Risk score
    st.subheader("Client Risk Assessment")
    failed = len(client_txns[client_txns["Status"]=="Failed"])
    reversed_count = len(client_txns[client_txns["Status"]=="Reversed"])
    risk_score = max(0, 100 - failed*20 - reversed_count*15)
    fig_g = go.Figure(go.Indicator(mode="gauge+number",value=risk_score,title={"text":"Risk Score"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#10b981" if risk_score>70 else "#f59e0b" if risk_score>40 else "#ef4444"},"steps":[{"range":[0,40],"color":"rgba(239,68,68,.1)"},{"range":[40,70],"color":"rgba(245,158,11,.1)"},{"range":[70,100],"color":"rgba(16,185,129,.1)"}]}))
    fig_g.update_layout(**PL,height=250)
    st.plotly_chart(fig_g,use_container_width=True)


# ──────────── TRANSACTIONS ────────────
elif page == "Transactions":
    st.title("Transactions")
    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("CRM",len(df_txn[df_txn["Source"]=="CRM"])); s2.metric("PSP",len(df_txn[df_txn["Source"]=="PSP"]))
    s3.metric("Bank",len(df_txn[df_txn["Source"]=="Bank"])); s4.metric("Commission",len(df_txn[df_txn["Source"]=="Commission"]))
    s5.metric("Bonus",len(df_txn[df_txn["Source"]=="Bonus"]))

    tabs = st.tabs(["All","CRM","PSP","Bank","Commission","Bonus"])
    sources = ["All","CRM","PSP","Bank","Commission","Bonus"]
    for idx,tab in enumerate(tabs):
        with tab:
            base = df_txn if sources[idx]=="All" else df_txn[df_txn["Source"]==sources[idx]]
            fc1,fc2,fc3,fc4 = st.columns(4)
            with fc1: search=st.text_input("Search",key=f"s{idx}",placeholder="ID, client...")
            with fc2: tf=st.selectbox("Type",["All"]+sorted(base["Type"].unique().tolist()),key=f"t{idx}")
            with fc3: sf=st.selectbox("Status",["All"]+sorted(base["Status"].unique().tolist()),key=f"st{idx}")
            with fc4: cf=st.selectbox("Currency",["All"]+sorted(base["CCY"].unique().tolist()),key=f"c{idx}")
            f = base.copy()
            if search: f=f[f.apply(lambda r:search.lower() in r["ID"].lower() or search.lower() in r["Client"].lower(),axis=1)]
            if tf!="All": f=f[f["Type"]==tf]
            if sf!="All": f=f[f["Status"]==sf]
            if cf!="All": f=f[f["CCY"]==cf]
            st.dataframe(f[["ID","Type","Source","Client","Amount","CCY","Counterparty","Bank","Status","Timestamp"]],use_container_width=True,hide_index=True)
            export_df(f, f"Transactions_{sources[idx]}", f"exp_txn_{idx}")

    # Charts
    tc1,tc2 = st.columns(2)
    with tc1:
        sv = df_txn.groupby("Source")["Amount"].sum().reset_index().sort_values("Amount",ascending=True)
        fig=px.bar(sv,y="Source",x="Amount",orientation="h",color="Source",color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#8b5cf6","#06b6d4"])
        fig.update_layout(**PL,height=220,showlegend=False,title="Volume by Source"); st.plotly_chart(fig,use_container_width=True)
    with tc2:
        fig_h=px.histogram(df_txn,x="Amount",nbins=15,color_discrete_sequence=["#8b5cf6"])
        fig_h.update_layout(**PL,height=220,title="Amount Distribution"); st.plotly_chart(fig_h,use_container_width=True)


# ──────────── RECONCILIATION ────────────
elif page == "Reconciliation":
    st.title("Smart Reconciliation Engine")
    st.caption("3-way match: CRM ↔ PSP ↔ Bank")
    m=len(df_rec[df_rec["Status"]=="Matched"]); p=len(df_rec[df_rec["Status"]=="Partial"]); e=len(df_rec[df_rec["Status"]=="Exception"])
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Matched",m); c2.metric("Partial",p); c3.metric("Exceptions",e,delta_color="inverse"); c4.metric("Rate",f"{m/len(df_rec)*100:.1f}%")

    # Charts
    rc1,rc2 = st.columns(2)
    with rc1:
        rp=pd.DataFrame({"Status":["Matched","Partial","Exception"],"Count":[m,p,e]})
        fig_rp=px.pie(rp,values="Count",names="Status",hole=0.55,color_discrete_map={"Matched":"#10b981","Partial":"#f59e0b","Exception":"#ef4444"})
        fig_rp.update_layout(**PL,height=220); fig_rp.update_traces(textinfo="value+percent"); st.plotly_chart(fig_rp,use_container_width=True)
    with rc2:
        fig_sc=px.histogram(df_rec,x="Score",nbins=10,color_discrete_sequence=["#6366f1"])
        fig_sc.update_layout(**PL,height=220,title="Score Distribution"); st.plotly_chart(fig_sc,use_container_width=True)

    tab1,tab2,tab3=st.tabs(["Matched","Partial","Exceptions"])
    for tab,status in [(tab1,"Matched"),(tab2,"Partial"),(tab3,"Exception")]:
        with tab:
            data=df_rec[df_rec["Status"]==status]
            st.dataframe(data[["Case_ID","TXN_ID","CRM_Amt","PSP_Amt","Bank_Amt","CCY","Score","Method","Case_Status","Notes"]],use_container_width=True,hide_index=True,column_config={"Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d%%")})
            export_df(data, f"Recon_{status}", f"exp_rec_{status}")

    st.markdown("---")
    st.subheader("Case Actions & Comments")
    open_cases=df_rec[df_rec["Case_Status"].isin(["Open","Investigating"])]
    if len(open_cases)>0:
        sel=st.selectbox("Select case",open_cases["Case_ID"].tolist())
        cs=df_rec[df_rec["Case_ID"]==sel].iloc[0]
        st.info(f"**{cs['Case_ID']}** — {cs['Notes']}")
        bc1,bc2,bc3=st.columns(3)
        if bc1.button("Investigate",key="inv",use_container_width=True): add_audit("Current User",f"Investigated {sel}","Reconciliation")
        if bc2.button("Approve",key="app",use_container_width=True): add_audit("Current User",f"Approved {sel}","Reconciliation")
        if bc3.button("Reject",key="rej",use_container_width=True): add_audit("Current User",f"Rejected {sel}","Reconciliation")
        show_comments(sel)


# ──────────── LEDGER ────────────
elif page == "Ledger":
    st.title("Ledger — Single Source of Truth")
    dr=df_led["Debit"].sum(); cr_total=df_led["Credit"].sum()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Debits",fmt_usd(dr)); c2.metric("Credits",fmt_usd(cr_total))
    c3.metric("Balance","OK" if abs(dr-cr_total)<0.01 else "IMBALANCED"); c4.metric("Entries",len(df_led))

    lc1,lc2 = st.columns(2)
    with lc1:
        adr=df_led.groupby("Account")["Debit"].sum().reset_index(); acr=df_led.groupby("Account")["Credit"].sum().reset_index()
        fig=go.Figure(); fig.add_trace(go.Bar(x=adr["Account"],y=adr["Debit"],name="Debit",marker_color="#06b6d4")); fig.add_trace(go.Bar(x=acr["Account"],y=acr["Credit"],name="Credit",marker_color="#8b5cf6"))
        fig.update_layout(**PL,height=240,barmode="group",title="DR vs CR by Account"); st.plotly_chart(fig,use_container_width=True)
    with lc2:
        ac=df_led["Account"].value_counts().reset_index(); ac.columns=["Account","Count"]
        fig_ac=px.pie(ac,values="Count",names="Account",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6"])
        fig_ac.update_layout(**PL,height=240); fig_ac.update_traces(textinfo="value+percent"); st.plotly_chart(fig_ac,use_container_width=True)

    af=st.selectbox("Filter Account",["All","Client Liability","Cash Account","PSP Account","Fee Account","Commission Account"])
    view=df_led if af=="All" else df_led[df_led["Account"]==af]
    st.dataframe(view,use_container_width=True,hide_index=True,column_config={"Debit":st.column_config.NumberColumn(format="$%.2f"),"Credit":st.column_config.NumberColumn(format="$%.2f")})
    export_df(view, "Ledger", "exp_led")

    tb=df_led.groupby("Account").agg(Debits=("Debit","sum"),Credits=("Credit","sum"),Entries=("Ref","count")).reset_index()
    tb["Net"]=tb["Debits"]-tb["Credits"]
    st.subheader("Trial Balance"); st.dataframe(tb,use_container_width=True,hide_index=True)


# ──────────── LIQUIDITY ────────────
elif page == "Liquidity":
    st.title("Liquidity Intelligence")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Available",fmt(available_cash)); c2.metric("Pending WD",fmt(pending_wd),delta_color="inverse")
    c3.metric("Liabilities",fmt(bonus_liability+commission_liability),delta_color="inverse"); c4.metric("Net Liquidity",fmt(net_liquidity),f"Buffer: {buffer_pct:.1f}%")

    lq1,lq2,lq3=st.columns(3)
    with lq1:
        bc=df_bank.groupby("CCY")["Balance"].sum().reset_index()
        fig=px.pie(bc,values="Balance",names="CCY",hole=0.55,color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig.update_layout(**PL,height=200); fig.update_traces(textinfo="value+percent"); st.caption("Bank by CCY"); st.plotly_chart(fig,use_container_width=True)
    with lq2:
        pc=df_psp.groupby("PSP").agg(Bal=("Balance","sum"),Pend=("Pending_Out","sum")).reset_index()
        fig2=go.Figure(); fig2.add_trace(go.Bar(x=pc["PSP"],y=pc["Bal"],name="Balance",marker_color="#10b981")); fig2.add_trace(go.Bar(x=pc["PSP"],y=pc["Pend"],name="Pending",marker_color="#ef4444"))
        fig2.update_layout(**PL,height=200,barmode="group"); st.caption("PSP Balance vs Pending"); st.plotly_chart(fig2,use_container_width=True)
    with lq3:
        fig_g=go.Figure(go.Indicator(mode="gauge+number",value=buffer_pct,title={"text":"Buffer %"},gauge={"axis":{"range":[0,30]},"bar":{"color":"#ef4444" if buffer_pct<15 else "#10b981"},"steps":[{"range":[0,15],"color":"rgba(239,68,68,.1)"},{"range":[15,30],"color":"rgba(16,185,129,.1)"}],"threshold":{"line":{"color":"#f59e0b","width":3},"thickness":0.8,"value":15}}))
        fig_g.update_layout(**PL,height=200); st.caption("Buffer Gauge"); st.plotly_chart(fig_g,use_container_width=True)

    # Waterfall
    wf_items=["Banks","PSP","Pending Out","Bonus","Commission","Net"]
    wf_vals=[total_bank,total_psp,-pending_wd,-bonus_liability,-commission_liability,net_liquidity]
    fig_wf=go.Figure(go.Waterfall(x=wf_items,y=wf_vals,measure=["relative","relative","relative","relative","relative","total"],increasing={"marker":{"color":"#10b981"}},decreasing={"marker":{"color":"#ef4444"}},totals={"marker":{"color":"#6366f1"}},connector={"line":{"color":"#334155"}}))
    fig_wf.update_layout(**PL,height=260,title="Liquidity Waterfall"); st.plotly_chart(fig_wf,use_container_width=True)

    cl,cr=st.columns(2)
    with cl: st.subheader("Banks"); st.dataframe(df_bank,use_container_width=True,hide_index=True); export_df(df_bank,"Bank_Balances","exp_bank")
    with cr: st.subheader("PSPs"); st.dataframe(df_psp[["PSP","CCY","Balance","Pending_In","Pending_Out"]],use_container_width=True,hide_index=True); export_df(df_psp,"PSP_Balances","exp_psp")


# ──────────── PSP SCORECARD ────────────
elif page == "PSP Scorecard":
    st.title("PSP Performance Scorecard")
    psp_summary = df_psp.groupby("PSP").agg(Total_Balance=("Balance","sum"),Total_Pending=("Pending_Out","sum"),Avg_Success=("Success_Rate","mean"),Avg_Settlement=("Avg_Settlement","first"),Avg_Cost=("Cost_Per_Txn","mean")).reset_index()
    psp_txns = df_txn[df_txn["Counterparty"].isin(df_psp["PSP"].unique())].groupby("Counterparty").size().reset_index(name="Txn_Count")
    psp_summary = psp_summary.merge(psp_txns, left_on="PSP", right_on="Counterparty", how="left").drop("Counterparty",axis=1).fillna(0)

    for _,psp in psp_summary.iterrows():
        st.subheader(f"{psp['PSP']}")
        p1,p2,p3,p4,p5=st.columns(5)
        p1.metric("Balance",fmt_usd(psp["Total_Balance"]))
        p2.metric("Pending Out",fmt_usd(psp["Total_Pending"]))
        p3.metric("Success Rate",f"{psp['Avg_Success']:.1f}%")
        p4.metric("Avg Settlement",psp["Avg_Settlement"])
        p5.metric("Cost/Txn",f"${psp['Avg_Cost']:.2f}")

    # Comparison charts
    st.subheader("PSP Comparison")
    pc1,pc2=st.columns(2)
    with pc1:
        fig=px.bar(psp_summary,x="PSP",y="Avg_Success",color="PSP",color_discrete_sequence=["#10b981","#6366f1","#f59e0b"])
        fig.update_layout(**PL,height=240,showlegend=False,title="Success Rate %"); st.plotly_chart(fig,use_container_width=True)
    with pc2:
        fig2=px.bar(psp_summary,x="PSP",y="Avg_Cost",color="PSP",color_discrete_sequence=["#ef4444","#f59e0b","#10b981"])
        fig2.update_layout(**PL,height=240,showlegend=False,title="Cost per Transaction $"); st.plotly_chart(fig2,use_container_width=True)


# ──────────── ALERTS ────────────
elif "Alerts" in page:
    st.title("Alerts & Exceptions")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Critical",len(df_alerts[(df_alerts["Severity"]=="Critical")&(df_alerts["Status"]!="Resolved")]))
    c2.metric("Open",len(df_alerts[df_alerts["Status"]=="Open"]))
    c3.metric("Investigating",len(df_alerts[df_alerts["Status"]=="Investigating"]))
    c4.metric("Resolved",len(df_alerts[df_alerts["Status"]=="Resolved"]))

    # Charts
    ac1,ac2=st.columns(2)
    with ac1:
        sv=df_alerts["Severity"].value_counts().reset_index(); sv.columns=["Severity","Count"]
        fig=px.pie(sv,values="Count",names="Severity",hole=0.55,color_discrete_map={"Critical":"#ef4444","High":"#f59e0b","Medium":"#f59e0b","Low":"#3b82f6"})
        fig.update_layout(**PL,height=200); fig.update_traces(textinfo="value+percent"); st.plotly_chart(fig,use_container_width=True)
    with ac2:
        cc=df_alerts["Category"].value_counts().reset_index(); cc.columns=["Category","Count"]
        fig2=px.bar(cc,x="Category",y="Count",color="Category",color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig2.update_layout(**PL,height=200,showlegend=False); st.plotly_chart(fig2,use_container_width=True)

    tab_all,tab_fin,tab_ops,tab_comp=st.tabs(["All","Financial","Operational","Compliance"])
    svf=st.selectbox("Severity",["All","Critical","High","Medium","Low"]); stf=st.selectbox("Status",["All","Open","Investigating","Resolved"],key="as2")
    def show_al(data):
        d=data.copy()
        if svf!="All": d=d[d["Severity"]==svf]
        if stf!="All": d=d[d["Status"]==stf]
        if len(d)==0: st.success("No alerts"); return
        for _,al in d.iterrows():
            st.markdown(f'<div class="alert-card"><div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow" if al["Status"]!="Resolved" else "green")}</div></div>',unsafe_allow_html=True)
    with tab_all: show_al(df_alerts)
    with tab_fin: show_al(df_alerts[df_alerts["Category"]=="Financial"])
    with tab_ops: show_al(df_alerts[df_alerts["Category"]=="Operational"])
    with tab_comp: show_al(df_alerts[df_alerts["Category"]=="Compliance"])

    st.subheader("Case Management & Comments")
    active=df_alerts[df_alerts["Status"].isin(["Open","Investigating"])]
    if len(active)>0:
        sel=st.selectbox("Select alert",active["ID"].tolist())
        al=df_alerts[df_alerts["ID"]==sel].iloc[0]
        st.info(f"**{al['Title']}** — {al['Description']}")
        bc1,bc2,bc3,bc4=st.columns(4)
        if bc1.button("Open Case",use_container_width=True,key="oc2"): add_audit("Current User",f"Opened case {sel}","Alerts")
        if bc2.button("Investigate",use_container_width=True,key="inv3"): add_audit("Current User",f"Investigating {sel}","Alerts")
        if bc3.button("Resolve",use_container_width=True,key="res2"): add_audit("Current User",f"Resolved {sel}","Alerts")
        bc4.button("Dismiss",use_container_width=True,key="dis2")
        show_comments(sel)


# ──────────── REPORTS ────────────
elif page == "Reports":
    st.title("Reports & Analytics")
    tab_daily,tab_profit,tab_kpis=st.tabs(["Daily/Weekly/Monthly","Profitability","KPIs"])

    with tab_daily:
        rpt=st.radio("Report",["Daily","Weekly","Monthly"],horizontal=True)
        if rpt=="Daily":
            d1,d2,d3,d4=st.columns(4); d1.metric("Opening",fmt(opening_balance)); d2.metric("Cash In",fmt(cash_in)); d3.metric("Cash Out",fmt(cash_out)); d4.metric("Net Flow",fmt(net_flow))
            fig=go.Figure(); fig.add_trace(go.Bar(x=["Cash In","Cash Out","Net"],y=[cash_in,cash_out,net_flow],marker_color=["#10b981","#ef4444","#6366f1"]))
            fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
        elif rpt=="Weekly":
            w1,w2,w3=st.columns(3); w1.metric("Recon Rate",f"{recon_rate:.1f}%"); w2.metric("Exceptions",len(df_rec[df_rec["Status"]=="Exception"])); w3.metric("Total Volume",fmt(df_cash["Net"].sum()))
            fig=go.Figure(); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Net"],mode="lines+markers",line=dict(color="#6366f1",width=2.5)))
            fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
        else:
            tr=df_profit["Revenue"].sum(); tc=df_profit["Costs"].sum(); tp=df_profit["Profit"].sum()
            m1,m2,m3=st.columns(3); m1.metric("Revenue",fmt(tr)); m2.metric("Costs",fmt(tc)); m3.metric("Profit",fmt(tp))
            fig_tree=px.treemap(df_profit,path=["Client"],values="Profit",color="Profit",color_continuous_scale=["#ef4444","#f59e0b","#10b981"])
            fig_tree.update_layout(**PL,height=300,title="Profitability Map"); st.plotly_chart(fig_tree,use_container_width=True)

    with tab_profit:
        pt1,pt2,pt3=st.tabs(["Client","IB Partners","Campaigns"])
        with pt1:
            cl,cr=st.columns(2)
            with cl:
                fig=go.Figure(); fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Revenue"],name="Revenue",marker_color="#10b981")); fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Costs"],name="Costs",marker_color="#ef4444")); fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Profit"],name="Profit",marker_color="#6366f1"))
                fig.update_layout(**PL,height=300,barmode="group"); st.plotly_chart(fig,use_container_width=True)
            with cr:
                dp=df_profit.copy(); dp["Margin"]=(dp["Profit"]/dp["Revenue"]*100).round(1).astype(str)+"%"
                st.dataframe(dp.sort_values("Profit",ascending=False)[["Client","Revenue","Costs","Profit","Margin"]],use_container_width=True,hide_index=True); export_df(dp,"Client_Profitability","exp_cp")
        with pt2:
            cl2,cr2=st.columns(2)
            with cl2:
                fig=go.Figure(); fig.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Net_Revenue"],name="Net Rev",marker_color="#10b981")); fig.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Commission"],name="Commission",marker_color="#8b5cf6"))
                fig.update_layout(**PL,height=300,barmode="group"); st.plotly_chart(fig,use_container_width=True)
            with cr2: st.dataframe(df_ib.sort_values("Net_Revenue",ascending=False),use_container_width=True,hide_index=True); export_df(df_ib,"IB_Profitability","exp_ib")
        with pt3:
            camps=pd.DataFrame([{"Campaign":"Welcome Bonus","Spend":45000,"Revenue":180000,"ROI":300},{"Campaign":"Loyalty","Spend":22000,"Revenue":95000,"ROI":332},{"Campaign":"Referral","Spend":15000,"Revenue":72000,"ROI":380},{"Campaign":"VIP Cashback","Spend":35000,"Revenue":110000,"ROI":214}])
            cl3,cr3=st.columns(2)
            with cl3:
                fig=go.Figure(); fig.add_trace(go.Bar(x=camps["Campaign"],y=camps["Spend"],name="Spend",marker_color="#ef4444")); fig.add_trace(go.Bar(x=camps["Campaign"],y=camps["Revenue"],name="Revenue",marker_color="#10b981"))
                fig.update_layout(**PL,height=300,barmode="group"); st.plotly_chart(fig,use_container_width=True)
            with cr3: st.dataframe(camps,use_container_width=True,hide_index=True)

    with tab_kpis:
        kc1,kc2=st.columns(2)
        with kc1: fig=px.line(df_kpi,x="Month",y="Net_Flow",markers=True,color_discrete_sequence=["#6366f1"]); fig.update_layout(**PL,height=240,title="Net Flow"); st.plotly_chart(fig,use_container_width=True)
        with kc2: fig2=px.line(df_kpi,x="Month",y="Recon_Rate",markers=True,color_discrete_sequence=["#10b981"]); fig2.add_hline(y=95,line_dash="dash",line_color="#f59e0b"); fig2.update_layout(**PL,height=240,title="Recon Rate"); st.plotly_chart(fig2,use_container_width=True)
        kc3,kc4=st.columns(2)
        with kc3: fig3=px.bar(df_kpi,x="Month",y="Op_Costs",color_discrete_sequence=["#ef4444"]); fig3.update_layout(**PL,height=240,showlegend=False,title="Op Costs"); st.plotly_chart(fig3,use_container_width=True)
        with kc4: fig4=px.line(df_kpi,x="Month",y="Exceptions",markers=True,color_discrete_sequence=["#f59e0b"]); fig4.update_layout(**PL,height=240,title="Exceptions"); st.plotly_chart(fig4,use_container_width=True)
        st.dataframe(df_kpi,use_container_width=True,hide_index=True); export_df(df_kpi,"Monthly_KPIs","exp_kpi")


# ──────────── CASH FORECAST ────────────
elif page == "Cash Forecast":
    st.title("Cash Flow Forecast")
    st.caption("7-day projection based on historical trend")
    avg_dep = df_cash["Deposits"].mean(); avg_wd = df_cash["Withdrawals"].mean()
    std_dep = df_cash["Deposits"].std(); std_wd = df_cash["Withdrawals"].std()

    forecast_days = ["May 19","May 20","May 21","May 22","May 23","May 24","May 25"]
    np.random.seed(42)
    forecast = []
    for d in forecast_days:
        dep = avg_dep + np.random.normal(0, std_dep*0.3)
        wd = avg_wd + np.random.normal(0, std_wd*0.3)
        forecast.append({"Date":d,"Deposits":dep,"Withdrawals":wd,"Net":dep-wd,"Type":"Forecast"})

    hist = [dict(row, Type="Actual") for row in cash_flow_this]
    combined = pd.DataFrame(hist + forecast)

    fig = go.Figure()
    actual = combined[combined["Type"]=="Actual"]
    fcast = combined[combined["Type"]=="Forecast"]
    fig.add_trace(go.Scatter(x=actual["Date"],y=actual["Net"],mode="lines+markers",name="Actual",line=dict(color="#6366f1",width=2.5)))
    fig.add_trace(go.Scatter(x=fcast["Date"],y=fcast["Net"],mode="lines+markers",name="Forecast",line=dict(color="#6366f1",width=2,dash="dot")))
    fig.add_trace(go.Scatter(x=fcast["Date"],y=fcast["Net"]*1.15,mode="lines",name="Upper Band",line=dict(color="#10b981",width=1,dash="dot"),showlegend=False))
    fig.add_trace(go.Scatter(x=fcast["Date"],y=fcast["Net"]*0.85,mode="lines",name="Lower Band",line=dict(color="#ef4444",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(99,102,241,.08)",showlegend=False))
    fig.update_layout(**PL,height=350,title="Net Flow — Actual + 7-Day Forecast")
    st.plotly_chart(fig,use_container_width=True)

    fc1,fc2,fc3 = st.columns(3)
    fc1.metric("Avg Daily Deposit (forecast)", fmt_usd(avg_dep))
    fc2.metric("Avg Daily Withdrawal (forecast)", fmt_usd(avg_wd))
    fc3.metric("Projected Net (7-day)", fmt_usd(sum([f["Net"] for f in forecast])))

    st.subheader("Forecast Detail")
    st.dataframe(pd.DataFrame(forecast)[["Date","Deposits","Withdrawals","Net"]],use_container_width=True,hide_index=True,column_config={"Deposits":st.column_config.NumberColumn(format="$%.0f"),"Withdrawals":st.column_config.NumberColumn(format="$%.0f"),"Net":st.column_config.NumberColumn(format="$%.0f")})


# ──────────── RISK MONITOR ────────────
elif page == "Risk Monitor":
    st.title("Risk Early Warning System")
    liq_risk="Red" if buffer_pct<15 else "Yellow" if buffer_pct<20 else "Green"
    settle_risk="Yellow" if len(df_alerts[df_alerts["Title"].str.contains("Delay",case=False)&(df_alerts["Status"]!="Resolved")])>0 else "Green"
    ops_risk="Red" if unmatched>3 else "Yellow" if unmatched>1 else "Green"
    psp_v=df_txn.groupby("Counterparty").size(); max_psp=(psp_v.max()/psp_v.sum()*100) if len(psp_v)>0 else 0
    conc_risk="Red" if max_psp>50 else "Yellow" if max_psp>35 else "Green"

    r1,r2,r3,r4=st.columns(4)
    for col,(name,status,detail) in zip([r1,r2,r3,r4],[("Liquidity",liq_risk,f"Buffer: {buffer_pct:.1f}%"),("Settlement",settle_risk,"PSP timing"),("Operational",ops_risk,f"{unmatched} unmatched"),("Concentration",conc_risk,f"Top: {max_psp:.0f}%")]):
        emoji="G" if status=="Green" else "Y" if status=="Yellow" else "R"
        rcolor = {"Green":"#10b981","Yellow":"#f59e0b","Red":"#ef4444"}.get(status,"#94a3b8")
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:28px">{emoji}</div><h4 style="font-size:12px">{name}</h4><div style="font-size:16px;font-weight:700;color:{rcolor}">{status}</div><div class="sm">{detail}</div></div>',unsafe_allow_html=True)

    # Radar
    scores={"Liquidity":85 if liq_risk=="Green" else 50 if liq_risk=="Yellow" else 20,"Settlement":80 if settle_risk=="Green" else 50,"Operational":80 if ops_risk=="Green" else 50 if ops_risk=="Yellow" else 20,"Concentration":70 if conc_risk=="Green" else 45 if conc_risk=="Yellow" else 20,"Compliance":60}
    cats=list(scores.keys()); vals=list(scores.values())
    fig=go.Figure(); fig.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",fillcolor="rgba(99,102,241,.15)",line=dict(color="#6366f1",width=2)))
    fig.add_trace(go.Scatterpolar(r=[80]*6,theta=cats+[cats[0]],line=dict(color="#10b981",width=1,dash="dash"),name="Target"))
    fig.update_layout(**PL,height=300,polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,100])))
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Scenario Analysis")
    inc=st.slider("Withdrawal increase %",0,200,50,10)
    nwd=pending_wd*(1+inc/100); nliq=available_cash-nwd-bonus_liability-commission_liability; nbuf=nliq/available_cash*100
    sc1,sc2,sc3=st.columns(3)
    sc1.metric("New Pending",fmt_usd(nwd),f"+{inc}%",delta_color="inverse")
    sc2.metric("New Net Liq",fmt_usd(nliq))
    sc3.metric("New Buffer",f"{nbuf:.1f}%","BREACH" if nbuf<15 else "OK",delta_color="inverse" if nbuf<15 else "normal")


# ──────────── AUDIT LOG ────────────
elif page == "Audit Log":
    st.title("Audit Log — SOX Compliance")
    st.caption("Complete trail of all user and system actions")
    df_audit = pd.DataFrame(st.session_state.audit_log)
    sf=st.selectbox("Filter by Section",["All"]+sorted(df_audit["Section"].unique().tolist()))
    uf=st.selectbox("Filter by User",["All"]+sorted(df_audit["User"].unique().tolist()))
    view=df_audit.copy()
    if sf!="All": view=view[view["Section"]==sf]
    if uf!="All": view=view[view["User"]==uf]
    st.dataframe(view,use_container_width=True,hide_index=True)
    export_df(view,"Audit_Log","exp_audit")

    st.subheader("Activity by User")
    user_acts=df_audit["User"].value_counts().reset_index(); user_acts.columns=["User","Actions"]
    fig=px.bar(user_acts,x="User",y="Actions",color_discrete_sequence=["#6366f1"])
    fig.update_layout(**PL,height=220,showlegend=False); st.plotly_chart(fig,use_container_width=True)

    st.subheader("Activity by Section")
    sec_acts=df_audit["Section"].value_counts().reset_index(); sec_acts.columns=["Section","Actions"]
    fig2=px.pie(sec_acts,values="Actions",names="Section",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6"])
    fig2.update_layout(**PL,height=220); st.plotly_chart(fig2,use_container_width=True)


# ──────────── ARCHITECTURE ────────────
elif page == "Architecture":
    st.title("System Architecture")
    layers=[("1. Connected Systems",["CRM","PSP Gateways","Bank Accounts","Trading Platform","Bonus Engine","Commission Engine"]),("2. Data Integration",["API Connectors","File Imports","Scheduled Pull","Validation"]),("3. Normalization",["Currency (ISO 4217)","Time (UTC)","ID Mapping","Type Mapping"]),("4. Reconciliation Engine",["ID Match","Amount Match","Time Window","Exception Detection"]),("5. Financial Ledger",["Client Funds","Company Cash","Fees","Commissions","Adjustments"])]
    for title,items in layers:
        st.subheader(title)
        cols=st.columns(len(items) if len(items)<=4 else 3)
        for i,item in enumerate(items):
            cols[i%len(cols)].markdown(f'<div class="card" style="text-align:center;min-height:50px"><div class="sm" style="color:{TXT};font-weight:500">{item}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="flow-arrow">▼</div>',unsafe_allow_html=True)
    st.subheader("6. → Liquidity + Alerts → 7. Reports → 8. End Users")
    uc=st.columns(5)
    for col,(i,r) in zip(uc,[("","CFO"),("","Finance"),("","Ops"),("","Recon"),("","Mgmt")]):
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:22px">{i}</div><h4 style="font-size:11px">{r}</h4></div>',unsafe_allow_html=True)


# ──────────── INTEGRATIONS ────────────
elif page == "Integrations":
    st.title("Integrations")
    intg=[("CRM","REST API","Online","2 min"),("Stripe","REST+Webhook","Online","30s"),("Adyen","REST API","Warning","4h"),("Worldpay","REST API","Online","5 min"),("JPMorgan","SWIFT/SFTP","Online","15 min"),("HSBC","SWIFT/SFTP","Online","15 min"),("Deutsche Bank","SWIFT/SFTP","Online","15 min"),("Barclays","SWIFT/SFTP","Online","15 min"),("Trading Platform","WebSocket","Online","Real-time"),("Bonus Engine","REST API","Online","18 min"),("Commission Engine","REST API","Online","10 min")]
    for name,proto,health,sync in intg:
        dot="dot-g" if health=="Online" else "dot-y"
        st.markdown(f'<div class="risk-row"><span class="risk-label"><span class="status-dot {dot}"></span>{name}</span><span class="sm">{proto} · {sync}</span></div>',unsafe_allow_html=True)


# ──────────── SETTINGS ────────────
elif page == "Settings":
    st.title("Settings")
    tab_gen,tab_users,tab_alerts_cfg,tab_recon_cfg=st.tabs(["General","Users","Alert Rules","Recon Rules"])
    with tab_gen:
        st.text_input("Company",value="FinanceOps Corp"); st.selectbox("Base CCY",["USD","EUR","GBP"],key="set_ccy"); st.number_input("Buffer Threshold %",value=15.0,step=0.5)
        st.button("Save",use_container_width=True,key="save_gen")
    with tab_users:
        st.dataframe(pd.DataFrame([{"User":"Ahmed K.","Role":"CFO","Access":"Full"},{"User":"Sarah M.","Role":"Finance Mgr","Access":"Finance+Reports"},{"User":"Omar R.","Role":"Ops Mgr","Access":"Txns+Alerts"},{"User":"Lina T.","Role":"Recon Analyst","Access":"Recon+Ledger"},{"User":"David P.","Role":"Compliance","Access":"Alerts+Reports"}]),use_container_width=True,hide_index=True)
    with tab_alerts_cfg:
        st.number_input("WD Spike %",value=200,step=10); st.number_input("Cash Imbalance $",value=5000,step=500); st.number_input("Recon Threshold %",value=5.0,step=0.5)
        st.button("Save",use_container_width=True,key="save_alert")
    with tab_recon_cfg:
        st.number_input("Amount Tolerance %",value=1.0,step=0.1); st.number_input("Time Window (h)",value=24,step=1)
        st.checkbox("Auto-create cases",value=True); st.checkbox("Auto-match ID+Amount",value=True)
        st.button("Save",use_container_width=True,key="save_recon")


# ──────────── FILE UPLOAD ────────────
elif page == "File Upload":
    st.title("File Upload")
    uploaded=st.file_uploader("Drop files",type=["docx","xlsx","xls","csv","pdf","txt"],accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            ext=uf.name.rsplit(".",1)[-1].lower()
            st.markdown("---"); st.subheader(f"{uf.name}")
            try:
                if ext=="csv":
                    df=pd.read_csv(uf); st.success(f"{len(df)} rows"); st.dataframe(df,use_container_width=True,hide_index=True); export_df(df,uf.name,f"exp_{uf.name}")
                elif ext in ["xlsx","xls"]:
                    xls=pd.ExcelFile(uf); sheet=st.selectbox("Sheet",xls.sheet_names,key=f"sh_{uf.name}")
                    df=pd.read_excel(uf,sheet_name=sheet); st.dataframe(df,use_container_width=True,hide_index=True)
                elif ext=="docx":
                    from docx import Document; doc=Document(uf)
                    for p in doc.paragraphs:
                        if p.text.strip():
                            if p.style and "Heading" in (p.style.name or ""): st.markdown(f"### {p.text}")
                            else: st.markdown(p.text)
                    for ti,table in enumerate(doc.tables):
                        rows=[[c.text.strip() for c in r.cells] for r in table.rows]
                        if len(rows)>1: st.dataframe(pd.DataFrame(rows[1:],columns=rows[0]),use_container_width=True,hide_index=True)
                elif ext=="pdf":
                    from PyPDF2 import PdfReader; reader=PdfReader(uf)
                    for i,pg in enumerate(reader.pages):
                        txt=pg.extract_text()
                        if txt: st.markdown(f"**Page {i+1}**"); st.text(txt)
                elif ext=="txt": st.text(uf.read().decode("utf-8",errors="replace"))
            except Exception as e: st.error(f"Error: {e}")
    else:
        st.markdown(f'<div class="card" style="text-align:center;padding:40px"><div style="font-size:28px;color:#6366f1;font-weight:700">Upload</div><h4>Upload files</h4><div class="sm">.docx · .xlsx · .csv · .pdf · .txt</div></div>',unsafe_allow_html=True)
