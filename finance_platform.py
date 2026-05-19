"""
FinanceOps v4 — Financial Control & Intelligence Platform
Production features: Auth, RBAC, Workflows, AI Anomaly, Case Mgmt, Data Quality
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib, io, json, time

st.set_page_config(page_title="FinanceOps", page_icon="F", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# USER DATABASE & AUTH
# ============================================================
USERS = {
    "ahmed": {"password": hashlib.sha256("admin123".encode()).hexdigest(), "name": "Ahmed K.", "role": "CFO", "access": ["all"]},
    "sarah": {"password": hashlib.sha256("finance1".encode()).hexdigest(), "name": "Sarah M.", "role": "Finance Manager", "access": ["Dashboard","Transactions","Reconciliation","Ledger","Liquidity","Reports","Cash Forecast","PSP Scorecard","File Upload"]},
    "omar": {"password": hashlib.sha256("ops123".encode()).hexdigest(), "name": "Omar R.", "role": "Operations Manager", "access": ["Dashboard","Transactions","Alerts","Activity Feed","Integrations","Data Quality"]},
    "lina": {"password": hashlib.sha256("recon1".encode()).hexdigest(), "name": "Lina T.", "role": "Recon Analyst", "access": ["Dashboard","Transactions","Reconciliation","Ledger","Alerts"]},
    "david": {"password": hashlib.sha256("comp123".encode()).hexdigest(), "name": "David P.", "role": "Compliance", "access": ["Dashboard","Alerts","Audit Log","Reports","Data Quality"]},
    "demo": {"password": hashlib.sha256("demo".encode()).hexdigest(), "name": "Demo User", "role": "CFO", "access": ["all"]},
}

SESSION_TIMEOUT_MIN = 30

# ============================================================
# SESSION STATE
# ============================================================
defaults = {"authenticated": False, "user": None, "role": None, "user_name": None, "last_activity": None,
    "theme": "dark", "base_ccy": "USD", "comments": {}, "notifications_read": set(),
    "case_assignments": {}, "case_approvals": {}, "custom_dashboards": [],
    "audit_log": [
        {"Time":"2026-05-18 11:15","User":"System","Action":"Auto-matched REC-001","Section":"Reconciliation"},
        {"Time":"2026-05-18 10:30","User":"Ahmed K.","Action":"Viewed Dashboard","Section":"Dashboard"},
        {"Time":"2026-05-18 10:15","User":"System","Action":"Alert ALT-001 created","Section":"Alerts"},
        {"Time":"2026-05-18 09:45","User":"Sarah M.","Action":"Exported Daily Report","Section":"Reports"},
        {"Time":"2026-05-18 09:30","User":"System","Action":"Alert ALT-002 created","Section":"Alerts"},
        {"Time":"2026-05-18 09:00","User":"Omar R.","Action":"Investigated REC-003","Section":"Reconciliation"},
        {"Time":"2026-05-18 08:30","User":"Lina T.","Action":"Approved REC-008","Section":"Reconciliation"},
        {"Time":"2026-05-18 08:00","User":"System","Action":"PSP Settlement processed","Section":"Transactions"},
        {"Time":"2026-05-18 07:00","User":"System","Action":"Alert ALT-006 created","Section":"Alerts"},
        {"Time":"2026-05-17 23:00","User":"System","Action":"Daily reconciliation completed","Section":"Reconciliation"},
    ],
    "activity_feed": [
        {"time":"11:15","user":"System","event":"REC-001 auto-matched","type":"recon"},
        {"time":"10:30","user":"Ahmed K.","event":"Logged in","type":"auth"},
        {"time":"10:15","user":"System","event":"Cash imbalance alert raised","type":"alert"},
        {"time":"09:45","user":"Sarah M.","event":"Generated daily report","type":"report"},
        {"time":"09:30","user":"System","event":"Withdrawal spike detected","type":"alert"},
        {"time":"09:00","user":"Omar R.","event":"Investigating REC-003","type":"recon"},
        {"time":"08:30","user":"Lina T.","event":"Approved REC-008","type":"recon"},
        {"time":"08:00","user":"System","event":"19 transactions processed","type":"txn"},
        {"time":"07:00","user":"System","event":"Liquidity buffer breach","type":"alert"},
    ],
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SESSION TIMEOUT CHECK
# ============================================================
if st.session_state.authenticated and st.session_state.last_activity:
    elapsed = (datetime.now() - st.session_state.last_activity).total_seconds() / 60
    if elapsed > SESSION_TIMEOUT_MIN:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.warning("Session expired due to inactivity. Please log in again.")

if st.session_state.authenticated:
    st.session_state.last_activity = datetime.now()

# ============================================================
# LOGIN PAGE
# ============================================================
if not st.session_state.authenticated:
    st.markdown("""<style>
    .main .block-container{max-width:450px;padding-top:10vh}
    div[data-testid="stMetric"]{background:#0f1729;border:1px solid #1e2d4a;border-radius:10px;padding:14px}
    </style>""", unsafe_allow_html=True)

    st.title("FinanceOps")
    st.caption("Financial Control & Intelligence Platform")
    st.markdown("---")

    username = st.text_input("Username", placeholder="Enter username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    if st.button("Sign In", use_container_width=True, type="primary"):
        if username in USERS:
            pw_hash = hashlib.sha256(password.encode()).hexdigest()
            if USERS[username]["password"] == pw_hash:
                st.session_state.authenticated = True
                st.session_state.user = username
                st.session_state.role = USERS[username]["role"]
                st.session_state.user_name = USERS[username]["name"]
                st.session_state.last_activity = datetime.now()
                st.session_state.activity_feed.insert(0, {"time": datetime.now().strftime("%H:%M"), "user": USERS[username]["name"], "event": "Logged in", "type": "auth"})
                st.session_state.audit_log.insert(0, {"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "User": USERS[username]["name"], "Action": "Logged in", "Section": "Auth"})
                st.rerun()
            else:
                st.error("Invalid password")
        else:
            st.error("User not found")

    st.markdown("---")
    st.caption("Demo accounts:")
    st.caption("demo / demo (full access)")
    st.caption("ahmed / admin123 (CFO) | sarah / finance1 (Finance)")
    st.caption("omar / ops123 (Ops) | lina / recon1 (Recon) | david / comp123 (Compliance)")
    st.stop()

# ============================================================
# RBAC HELPER
# ============================================================
user_access = USERS.get(st.session_state.user, {}).get("access", [])
def has_access(page_name):
    if "all" in user_access:
        return True
    return page_name in user_access

def add_audit(action, section):
    st.session_state.audit_log.insert(0, {"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "User": st.session_state.user_name, "Action": action, "Section": section})

def add_feed(event, etype="action"):
    st.session_state.activity_feed.insert(0, {"time": datetime.now().strftime("%H:%M"), "user": st.session_state.user_name, "event": event, "type": etype})

# ============================================================
# THEME & STYLES
# ============================================================
dark = st.session_state.theme == "dark"
CARD_BG = "#0f1729" if dark else "#f8fafc"
CARD_BD = "#1e2d4a" if dark else "#e2e8f0"
TXT = "#e2e8f0" if dark else "#1e293b"
TXT2 = "#94a3b8" if dark else "#64748b"

st.markdown(f"""<style>
.main .block-container{{padding-top:1rem}}
div[data-testid="stMetric"]{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:10px;padding:14px 18px}}
div[data-testid="stMetric"] label{{font-size:11px!important;text-transform:uppercase;letter-spacing:.5px}}
.badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}}
.bg-green{{background:rgba(16,185,129,.12);color:#10b981}}.bg-red{{background:rgba(239,68,68,.12);color:#ef4444}}
.bg-yellow{{background:rgba(245,158,11,.12);color:#f59e0b}}.bg-blue{{background:rgba(59,130,246,.12);color:#3b82f6}}
.bg-purple{{background:rgba(139,92,246,.12);color:#8b5cf6}}
.card{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:10px;padding:16px 18px;margin-bottom:8px}}
.card h4{{font-size:14px;font-weight:700;margin-bottom:8px;color:{TXT}}}
.sm{{font-size:12px;color:{TXT2}}}
.alert-card{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:8px;padding:14px 18px;margin-bottom:8px}}
.alert-title{{font-weight:600;font-size:13px;margin-bottom:3px;color:{TXT}}}
.alert-desc{{font-size:12px;color:{TXT2};line-height:1.5}}
.risk-row{{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:{CARD_BG};border:1px solid {CARD_BD};border-radius:8px;margin-bottom:6px}}
.risk-label{{font-size:13px;color:{TXT}}}.risk-val{{font-weight:700;font-size:13px}}
.flow-arrow{{text-align:center;font-size:20px;color:#6366f1;margin:4px 0}}
.status-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
.dot-g{{background:#10b981}}.dot-y{{background:#f59e0b}}.dot-r{{background:#ef4444}}
.feed-item{{padding:8px 12px;border-left:3px solid #6366f1;margin-bottom:6px;background:{CARD_BG};border-radius:0 6px 6px 0}}
.feed-time{{font-size:10px;color:{TXT2}}}
.comment-box{{background:{CARD_BG};border:1px solid {CARD_BD};border-radius:6px;padding:10px 14px;margin:4px 0;font-size:12px}}
.sla-bar{{height:6px;border-radius:3px;background:#1e2d4a;overflow:hidden;margin-top:4px}}
.sla-fill{{height:100%;border-radius:3px}}
div[data-testid="stSidebar"]>div:first-child{{padding-top:.8rem}}
</style>""", unsafe_allow_html=True)

PLT = "plotly_dark" if dark else "plotly_white"
PL = dict(template=PLT, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0))

def fmt(n): return f"${n:,.0f}"
def badge(t,c="blue"): return f'<span class="badge bg-{c}">{t}</span>'
def export_df(df,name,key): st.download_button(f"Export {name}",df.to_csv(index=False).encode(),f"{name}.csv","text/csv",key=key)

def show_comments(cid):
    for c in st.session_state.comments.get(cid,[]):
        st.markdown(f'<div class="comment-box"><b>{c["user"]}</b> <span class="sm">({c["time"]})</span><br>{c["text"]}</div>',unsafe_allow_html=True)
    new=st.text_input(f"Add comment",key=f"cmt_{cid}",placeholder="Type a comment...")
    if new:
        if cid not in st.session_state.comments: st.session_state.comments[cid]=[]
        st.session_state.comments[cid].append({"user":st.session_state.user_name,"text":new,"time":datetime.now().strftime("%H:%M")})
        add_audit(f"Commented on {cid}","Comments"); add_feed(f"Commented on {cid}","comment")
        st.rerun()

def sla_bar(created_str, sla_hours):
    created = datetime.strptime(created_str, "%Y-%m-%d %H:%M")
    elapsed = (datetime.now() - created).total_seconds() / 3600
    pct = min(elapsed / sla_hours * 100, 100)
    color = "#10b981" if pct < 60 else "#f59e0b" if pct < 90 else "#ef4444"
    remaining = max(0, sla_hours - elapsed)
    st.markdown(f'<div class="sm">SLA: {remaining:.1f}h remaining of {sla_hours}h</div><div class="sla-bar"><div class="sla-fill" style="width:{pct}%;background:{color}"></div></div>', unsafe_allow_html=True)

# ============================================================
# MOCK DATA
# ============================================================
transactions_data = [
    {"ID":"TXN-001","Type":"Deposit","Source":"CRM","Amount":25000,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Client deposit"},
    {"ID":"TXN-002","Type":"Withdrawal","Source":"CRM","Amount":12500,"CCY":"USD","Status":"Settled","Client":"Globe Ltd","Counterparty":"Adyen","Bank":"HSBC","Timestamp":"2026-05-18 09:15","Value_Date":"2026-05-18","Description":"Client withdrawal"},
    {"ID":"TXN-003","Type":"Deposit","Source":"PSP","Amount":8700,"CCY":"EUR","Status":"Pending","Client":"NovaTech","Counterparty":"Stripe","Bank":"Deutsche Bank","Timestamp":"2026-05-18 09:42","Value_Date":"2026-05-18","Description":"PSP settlement T+1"},
    {"ID":"TXN-004","Type":"Transfer","Source":"Bank","Amount":50000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"JPMorgan","Timestamp":"2026-05-18 10:01","Value_Date":"2026-05-18","Description":"Treasury rebalancing"},
    {"ID":"TXN-005","Type":"Fee","Source":"PSP","Amount":125,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Processing fee"},
    {"ID":"TXN-006","Type":"Commission","Source":"Commission","Amount":375,"CCY":"USD","Status":"Settled","Client":"IB-Alpha","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 10:30","Value_Date":"2026-05-18","Description":"IB commission"},
    {"ID":"TXN-007","Type":"Deposit","Source":"CRM","Amount":150000,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Worldpay","Bank":"Barclays","Timestamp":"2026-05-17 14:22","Value_Date":"2026-05-17","Description":"High-value deposit"},
    {"ID":"TXN-008","Type":"Withdrawal","Source":"CRM","Amount":45000,"CCY":"GBP","Status":"Failed","Client":"BritCo","Counterparty":"Adyen","Bank":"Barclays","Timestamp":"2026-05-17 16:45","Value_Date":"2026-05-17","Description":"Failed - insufficient PSP"},
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
    {"Case_ID":"REC-001","TXN_ID":"TXN-001","CRM_Amt":25000,"PSP_Amt":25000,"Bank_Amt":25000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched","Created":"2026-05-18 08:25","SLA_Hours":4},
    {"Case_ID":"REC-002","TXN_ID":"TXN-003","CRM_Amt":8700,"PSP_Amt":8700,"Bank_Amt":9000,"CCY":"EUR","Status":"Partial","Score":72,"Case_Status":"Investigating","Method":"ID+Time","Notes":"FX variance suspected","Created":"2026-05-18 09:45","SLA_Hours":4},
    {"Case_ID":"REC-003","TXN_ID":"TXN-008","CRM_Amt":45000,"PSP_Amt":45000,"Bank_Amt":0,"CCY":"GBP","Status":"Exception","Score":0,"Case_Status":"Open","Method":"ID Match","Notes":"No bank record","Created":"2026-05-17 17:00","SLA_Hours":4},
    {"Case_ID":"REC-004","TXN_ID":"TXN-007","CRM_Amt":150000,"PSP_Amt":150000,"Bank_Amt":150000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched","Created":"2026-05-17 14:30","SLA_Hours":4},
    {"Case_ID":"REC-005","TXN_ID":"TXN-012","CRM_Amt":5600,"PSP_Amt":5600,"Bank_Amt":5600,"CCY":"USD","Status":"Exception","Score":45,"Case_Status":"Investigating","Method":"Amount+Time","Notes":"Chargeback pending","Created":"2026-05-16 10:00","SLA_Hours":8},
    {"Case_ID":"REC-006","TXN_ID":"TXN-009","CRM_Amt":32000,"PSP_Amt":32000,"Bank_Amt":32000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched","Created":"2026-05-17 11:15","SLA_Hours":4},
    {"Case_ID":"REC-007","TXN_ID":"TXN-010","CRM_Amt":18200,"PSP_Amt":18200,"Bank_Amt":18500,"CCY":"EUR","Status":"Partial","Score":68,"Case_Status":"Open","Method":"ID+Time","Notes":"Amount discrepancy","Created":"2026-05-18 08:00","SLA_Hours":4},
    {"Case_ID":"REC-008","TXN_ID":"TXN-002","CRM_Amt":12500,"PSP_Amt":12500,"Bank_Amt":12500,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched","Created":"2026-05-18 09:20","SLA_Hours":4},
]

ledger_data = [
    {"Date":"2026-05-18","Account":"Cash Account","Debit":25000,"Credit":0,"CCY":"USD","Ref":"TXN-001","Narration":"Deposit - Acme Corp"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":25000,"CCY":"USD","Ref":"TXN-001","Narration":"Client funds credited"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":12500,"Credit":0,"CCY":"USD","Ref":"TXN-002","Narration":"Withdrawal - Globe Ltd"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":12500,"CCY":"USD","Ref":"TXN-002","Narration":"Cash disbursed"},
    {"Date":"2026-05-18","Account":"PSP Account","Debit":8700,"Credit":0,"CCY":"EUR","Ref":"TXN-003","Narration":"PSP receivable"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":8700,"CCY":"EUR","Ref":"TXN-003","Narration":"Client funds - NovaTech"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":50000,"Credit":0,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer in"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":50000,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer out"},
    {"Date":"2026-05-18","Account":"Fee Account","Debit":0,"Credit":125,"CCY":"USD","Ref":"TXN-005","Narration":"Fee revenue"},
    {"Date":"2026-05-18","Account":"Commission Account","Debit":375,"Credit":0,"CCY":"USD","Ref":"TXN-006","Narration":"Commission expense"},
    {"Date":"2026-05-17","Account":"Cash Account","Debit":150000,"Credit":0,"CCY":"USD","Ref":"TXN-007","Narration":"Deposit - MegaFund"},
    {"Date":"2026-05-17","Account":"Client Liability","Debit":0,"Credit":150000,"CCY":"USD","Ref":"TXN-007","Narration":"Client funds - MegaFund"},
]

alerts_data = [
    {"ID":"ALT-001","Title":"Cash Position Imbalance","Category":"Financial","Severity":"Critical","Status":"Open","Description":"GL cash deviates from bank by $12,400.","SLA_Hours":4,"Linked":"TXN-004","Created":"2026-05-18 10:15"},
    {"ID":"ALT-002","Title":"Withdrawal Spike","Category":"Financial","Severity":"High","Status":"Investigating","Description":"Withdrawal volume 340% above average.","SLA_Hours":8,"Linked":"TXN-002, TXN-010","Created":"2026-05-18 09:30"},
    {"ID":"ALT-003","Title":"PSP Delay - Adyen","Category":"Operational","Severity":"Medium","Status":"Open","Description":"Adyen batch delayed 4h.","SLA_Hours":12,"Linked":"TXN-010","Created":"2026-05-18 08:00"},
    {"ID":"ALT-004","Title":"Recon Exception Rate 20%","Category":"Operational","Severity":"High","Status":"Investigating","Description":"Above 5% threshold.","SLA_Hours":4,"Linked":"REC-003, REC-005","Created":"2026-05-18 10:00"},
    {"ID":"ALT-005","Title":"Chargeback Provision","Category":"Financial","Severity":"Medium","Status":"Resolved","Description":"TXN-012 chargeback booked.","SLA_Hours":24,"Linked":"TXN-012","Created":"2026-05-16 10:30"},
    {"ID":"ALT-006","Title":"Liquidity Buffer Low","Category":"Financial","Severity":"Critical","Status":"Open","Description":"Buffer at 11.2% below 15%.","SLA_Hours":2,"Linked":"","Created":"2026-05-18 07:00"},
    {"ID":"ALT-007","Title":"Unexplained Transaction","Category":"Compliance","Severity":"High","Status":"Open","Description":"$35K credit no CRM match.","SLA_Hours":4,"Linked":"","Created":"2026-05-18 11:00"},
]

bank_balances = [{"Bank":"JPMorgan","CCY":"USD","Balance":2450000},{"Bank":"HSBC","CCY":"USD","Balance":890000},{"Bank":"Deutsche Bank","CCY":"EUR","Balance":1250000},{"Bank":"Barclays","CCY":"GBP","Balance":675000},{"Bank":"Barclays","CCY":"USD","Balance":320000}]
psp_balances = [{"PSP":"Stripe","CCY":"USD","Balance":185000,"Pending_In":32000,"Pending_Out":8500,"Success_Rate":99.2,"Avg_Settle":"4.2h","Cost":0.52},{"PSP":"Stripe","CCY":"EUR","Balance":45000,"Pending_In":8700,"Pending_Out":0,"Success_Rate":98.8,"Avg_Settle":"5.1h","Cost":0.48},{"PSP":"Adyen","CCY":"USD","Balance":92000,"Pending_In":0,"Pending_Out":12500,"Success_Rate":97.5,"Avg_Settle":"6.8h","Cost":0.61},{"PSP":"Adyen","CCY":"EUR","Balance":63200,"Pending_In":0,"Pending_Out":18200,"Success_Rate":96.1,"Avg_Settle":"8.2h","Cost":0.58},{"PSP":"Worldpay","CCY":"USD","Balance":210000,"Pending_In":0,"Pending_Out":0,"Success_Rate":99.5,"Avg_Settle":"3.5h","Cost":0.45},{"PSP":"Worldpay","CCY":"GBP","Balance":55000,"Pending_In":0,"Pending_Out":45000,"Success_Rate":98.0,"Avg_Settle":"5.0h","Cost":0.55}]
cash_flow = [{"Date":"May 12","Deposits":185000,"Withdrawals":72000,"Net":113000},{"Date":"May 13","Deposits":210000,"Withdrawals":95000,"Net":115000},{"Date":"May 14","Deposits":145000,"Withdrawals":120000,"Net":25000},{"Date":"May 15","Deposits":290000,"Withdrawals":88000,"Net":202000},{"Date":"May 16","Deposits":178000,"Withdrawals":156000,"Net":22000},{"Date":"May 17","Deposits":320000,"Withdrawals":105000,"Net":215000},{"Date":"May 18","Deposits":307700,"Withdrawals":75700,"Net":232000}]
profitability = [{"Client":"Acme Corp","Revenue":45000,"Costs":12000,"Profit":33000},{"Client":"Globe Ltd","Revenue":28000,"Costs":9500,"Profit":18500},{"Client":"MegaFund","Revenue":82000,"Costs":18000,"Profit":64000},{"Client":"NovaTech","Revenue":15000,"Costs":6200,"Profit":8800},{"Client":"SolarInc","Revenue":32000,"Costs":8800,"Profit":23200},{"Client":"TradeCo","Revenue":51000,"Costs":14500,"Profit":36500}]
ib_data = [{"Partner":"IB-Alpha","Clients":42,"Volume":1250000,"Commission":18750,"Net_Revenue":62500},{"Partner":"IB-Beta","Clients":28,"Volume":890000,"Commission":13350,"Net_Revenue":44500},{"Partner":"IB-Gamma","Clients":65,"Volume":2100000,"Commission":31500,"Net_Revenue":105000},{"Partner":"IB-Delta","Clients":15,"Volume":420000,"Commission":6300,"Net_Revenue":21000}]
monthly_kpis = [{"Month":"Jan","Net_Flow":1200000,"Op_Costs":85000,"Exceptions":12,"Recon_Rate":94.2},{"Month":"Feb","Net_Flow":980000,"Op_Costs":78000,"Exceptions":8,"Recon_Rate":96.1},{"Month":"Mar","Net_Flow":1450000,"Op_Costs":92000,"Exceptions":15,"Recon_Rate":93.8},{"Month":"Apr","Net_Flow":1100000,"Op_Costs":88000,"Exceptions":10,"Recon_Rate":95.5},{"Month":"May","Net_Flow":1680000,"Op_Costs":95000,"Exceptions":18,"Recon_Rate":91.0}]

scheduled_jobs = [{"Job":"CRM Data Pull","Schedule":"Every 5 min","Last_Run":"2026-05-18 11:15","Status":"Success","Duration":"2.1s","Records":142},{"Job":"PSP Settlement Import","Schedule":"Every 15 min","Last_Run":"2026-05-18 11:00","Status":"Success","Duration":"4.8s","Records":38},{"Job":"Bank Statement (SWIFT)","Schedule":"Every 30 min","Last_Run":"2026-05-18 11:00","Status":"Success","Duration":"8.2s","Records":15},{"Job":"Auto-Reconciliation","Schedule":"Every 10 min","Last_Run":"2026-05-18 11:10","Status":"Success","Duration":"12.5s","Records":10},{"Job":"Commission Calc","Schedule":"Hourly","Last_Run":"2026-05-18 11:00","Status":"Success","Duration":"3.1s","Records":4},{"Job":"Liquidity Check","Schedule":"Every 5 min","Last_Run":"2026-05-18 11:15","Status":"Warning","Duration":"1.2s","Records":1},{"Job":"Daily Report Gen","Schedule":"Daily 01:00","Last_Run":"2026-05-18 01:00","Status":"Success","Duration":"45s","Records":1}]

# DataFrames
df_txn=pd.DataFrame(transactions_data); df_rec=pd.DataFrame(reconciliation_data); df_led=pd.DataFrame(ledger_data)
df_alerts=pd.DataFrame(alerts_data); df_bank=pd.DataFrame(bank_balances); df_psp=pd.DataFrame(psp_balances)
df_cash=pd.DataFrame(cash_flow); df_profit=pd.DataFrame(profitability); df_ib=pd.DataFrame(ib_data); df_kpi=pd.DataFrame(monthly_kpis)

# Computed
opening_balance=4250000; today="2026-05-18"
cash_in=df_txn[(df_txn["Type"]=="Deposit")&(df_txn["Value_Date"]==today)&(df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
cash_out=df_txn[(df_txn["Type"].isin(["Withdrawal","Fee","Commission"]))&(df_txn["Value_Date"]==today)&(df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
net_flow=cash_in-cash_out; total_bank=df_bank["Balance"].sum(); total_psp=df_psp["Balance"].sum()
available_cash=total_bank+total_psp; pending_wd=df_txn[(df_txn["Type"]=="Withdrawal")&(df_txn["Status"]=="Pending")]["Amount"].sum()+df_psp["Pending_Out"].sum()
bonus_liability=125000; commission_liability=48500; net_liquidity=available_cash-pending_wd-bonus_liability-commission_liability
buffer_pct=(net_liquidity/available_cash*100) if available_cash>0 else 0
matched_count=len(df_rec[df_rec["Status"]=="Matched"]); recon_rate=matched_count/len(df_rec)*100; unmatched=len(df_rec[df_rec["Status"].isin(["Partial","Exception"])])
open_alert_count=len(df_alerts[df_alerts["Status"].isin(["Open","Investigating"])])

# AI Anomaly Detection
def detect_anomalies(df):
    anomalies = []
    mean_amt = df["Amount"].mean(); std_amt = df["Amount"].std()
    for _,row in df.iterrows():
        flags = []
        if row["Amount"] > mean_amt + 2*std_amt: flags.append("Unusually high amount")
        if row["Status"] == "Failed": flags.append("Failed transaction")
        if row["Status"] == "Reversed": flags.append("Reversed/Chargeback")
        if row["Type"] == "Withdrawal" and row["Amount"] > 30000: flags.append("Large withdrawal")
        if flags:
            anomalies.append({"ID":row["ID"],"Client":row["Client"],"Amount":row["Amount"],"Flags":", ".join(flags),"Risk":"High" if len(flags)>1 else "Medium"})
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

# Client Scoring
def client_scores(df_t, df_p):
    scores = []
    for client in df_t[df_t["Client"]!="Internal"]["Client"].unique():
        ct = df_t[df_t["Client"]==client]
        failed = len(ct[ct["Status"]=="Failed"]); reversed_c = len(ct[ct["Status"]=="Reversed"])
        total = len(ct); volume = ct["Amount"].sum()
        cp = df_p[df_p["Client"]==client]
        profit = cp["Profit"].sum() if len(cp)>0 else 0
        risk = max(0, 100 - failed*25 - reversed_c*15)
        churn = "Low" if profit>20000 and failed==0 else "Medium" if profit>5000 else "High"
        scores.append({"Client":client,"Transactions":total,"Volume":volume,"Risk_Score":risk,"Churn_Risk":churn,"Failed":failed,"Reversed":reversed_c})
    return pd.DataFrame(scores)

# Data Quality
def check_data_quality(df, name):
    issues = []
    total = len(df)
    for col in df.columns:
        nulls = df[col].isna().sum()
        if nulls > 0: issues.append({"Dataset":name,"Field":col,"Issue":"Missing values","Count":nulls,"Pct":f"{nulls/total*100:.1f}%"})
        if df[col].dtype == "object":
            empty = (df[col].str.strip()=="").sum()
            if empty > 0: issues.append({"Dataset":name,"Field":col,"Issue":"Empty strings","Count":empty,"Pct":f"{empty/total*100:.1f}%"})
    dupes = df.duplicated().sum()
    if dupes > 0: issues.append({"Dataset":name,"Field":"ALL","Issue":"Duplicate rows","Count":dupes,"Pct":f"{dupes/total*100:.1f}%"})
    return issues

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"### **FinanceOps**")
    st.caption(f"Signed in: **{st.session_state.user_name}** ({st.session_state.role})")
    if st.button("Sign Out", use_container_width=True):
        add_audit("Logged out","Auth"); add_feed("Logged out","auth")
        for k in ["authenticated","user","role","user_name","last_activity"]: st.session_state[k] = defaults[k]
        st.rerun()
    st.markdown("---")

    theme_label = "Dark" if dark else "Light"
    if st.button(f"Theme: {theme_label}", use_container_width=True):
        st.session_state.theme = "light" if dark else "dark"; st.rerun()

    st.session_state.base_ccy = st.selectbox("Base Currency",["USD","EUR","GBP"],index=["USD","EUR","GBP"].index(st.session_state.base_ccy))
    st.markdown("---")

    all_pages = ["Dashboard","Activity Feed","Client 360","Transactions","Reconciliation","Ledger","Liquidity","PSP Scorecard",f"Alerts ({open_alert_count})","Reports","Cash Forecast","Risk Monitor","AI Anomaly","Client Scoring","Data Quality","Scheduled Jobs","Custom Dashboard","Audit Log","Architecture","Integrations","Data Import","Settings","File Upload"]
    visible = [p for p in all_pages if has_access(p.split(" (")[0]) or has_access(p)]
    page = st.radio("",visible,label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"Session: {SESSION_TIMEOUT_MIN}min timeout")
    st.caption(datetime.now().strftime("%a, %b %d, %Y %H:%M"))

# ============================================================
# PAGE ROUTING
# ============================================================
page_key = page.split(" (")[0]

if page_key == "Dashboard":
    st.title("Dashboard - Financial Control Tower")
    st.caption(f"Welcome back, {st.session_state.user_name} ({st.session_state.role})")
    f1,f2,f3,f4=st.columns(4)
    f1.metric("Opening Balance",fmt(opening_balance)); f2.metric("Cash In Today",fmt(cash_in),f"+{cash_in/opening_balance*100:.1f}%")
    f3.metric("Cash Out Today",fmt(cash_out),f"-{cash_out/opening_balance*100:.1f}%",delta_color="inverse"); f4.metric("Net Flow",fmt(net_flow))

    l1,l2,l3,l4=st.columns(4)
    l1.metric("Available Cash",fmt(available_cash)); l2.metric("Pending WD",fmt(pending_wd),delta_color="inverse")
    l3.metric("Net Liquidity",fmt(net_liquidity)); l4.metric("Buffer",f"{buffer_pct:.1f}%","Below 15%" if buffer_pct<15 else "OK",delta_color="inverse" if buffer_pct<15 else "normal")

    cl,cr=st.columns(2)
    with cl:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Deposits"],mode="lines",name="In",fill="tozeroy",line=dict(color="#10b981",width=2),fillcolor="rgba(16,185,129,.12)"))
        fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Withdrawals"],mode="lines",name="Out",fill="tozeroy",line=dict(color="#ef4444",width=2),fillcolor="rgba(239,68,68,.12)"))
        fig.update_layout(**PL,height=240,legend=dict(orientation="h",y=1.1)); st.plotly_chart(fig,use_container_width=True)
    with cr:
        fig2=px.bar(df_cash,x="Date",y="Net",color_discrete_sequence=["#6366f1"]); fig2.update_layout(**PL,height=240,showlegend=False); st.plotly_chart(fig2,use_container_width=True)

    st.subheader("Alerts")
    for _,al in df_alerts[df_alerts["Status"].isin(["Open","Investigating"])].iterrows():
        sev_color = "red" if al["Severity"] in ["Critical","High"] else "yellow"
        st.markdown(f'<div class="alert-card"><div class="alert-title">{al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],sev_color)} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow")}</div></div>',unsafe_allow_html=True)

    k1,k2,k3=st.columns(3)
    k1.metric("Daily Volume",f"{len(df_txn[df_txn['Value_Date']==today])} txns"); k2.metric("Recon Rate",f"{recon_rate:.1f}%"); k3.metric("Unmatched",unmatched,delta_color="inverse")

elif page_key == "Activity Feed":
    st.title("Activity Feed")
    st.caption("Live stream of system and user events")
    type_filter = st.selectbox("Filter",["All","auth","alert","recon","txn","report","comment","action"])
    for item in st.session_state.activity_feed:
        if type_filter != "All" and item["type"] != type_filter: continue
        border_color = {"alert":"#ef4444","auth":"#6366f1","recon":"#10b981","txn":"#f59e0b","report":"#8b5cf6"}.get(item["type"],"#6366f1")
        st.markdown(f'<div class="feed-item" style="border-left-color:{border_color}"><div class="feed-time">{item["time"]}</div><b>{item["user"]}</b>: {item["event"]}</div>',unsafe_allow_html=True)

elif page_key == "Client 360":
    st.title("Client 360 View")
    clients=sorted(df_txn[df_txn["Client"]!="Internal"]["Client"].unique().tolist())
    sel=st.selectbox("Select Client",clients); ct=df_txn[df_txn["Client"]==sel]; cp=df_profit[df_profit["Client"]==sel]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Transactions",len(ct)); c2.metric("Volume",fmt(ct["Amount"].sum()))
    if len(cp)>0: c3.metric("Revenue",fmt(cp.iloc[0]["Revenue"])); c4.metric("Profit",fmt(cp.iloc[0]["Profit"]))
    else: c3.metric("Revenue","N/A"); c4.metric("Profit","N/A")
    st.dataframe(ct[["ID","Type","Amount","CCY","Status","Timestamp"]],use_container_width=True,hide_index=True)
    export_df(ct,f"{sel}_txns",f"exp360_{sel}")
    scores=client_scores(df_txn,df_profit)
    cs=scores[scores["Client"]==sel]
    if len(cs)>0:
        st.subheader("Risk Assessment")
        rs=cs.iloc[0]
        fig=go.Figure(go.Indicator(mode="gauge+number",value=rs["Risk_Score"],title={"text":"Risk Score"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#10b981" if rs["Risk_Score"]>70 else "#f59e0b" if rs["Risk_Score"]>40 else "#ef4444"}}))
        fig.update_layout(**PL,height=220); st.plotly_chart(fig,use_container_width=True)
        st.metric("Churn Risk",rs["Churn_Risk"])

elif page_key == "Transactions":
    st.title("Transactions")
    tabs=st.tabs(["All","CRM","PSP","Bank","Commission","Bonus"]); sources=["All","CRM","PSP","Bank","Commission","Bonus"]
    for idx,tab in enumerate(tabs):
        with tab:
            base=df_txn if sources[idx]=="All" else df_txn[df_txn["Source"]==sources[idx]]
            fc1,fc2,fc3=st.columns(3)
            with fc1: search=st.text_input("Search",key=f"s{idx}")
            with fc2: tf=st.selectbox("Type",["All"]+sorted(base["Type"].unique().tolist()),key=f"t{idx}")
            with fc3: sf=st.selectbox("Status",["All"]+sorted(base["Status"].unique().tolist()),key=f"st{idx}")
            f=base.copy()
            if search: f=f[f.apply(lambda r:search.lower() in r["ID"].lower() or search.lower() in r["Client"].lower(),axis=1)]
            if tf!="All": f=f[f["Type"]==tf]
            if sf!="All": f=f[f["Status"]==sf]
            st.dataframe(f[["ID","Type","Source","Client","Amount","CCY","Status","Timestamp"]],use_container_width=True,hide_index=True)
            export_df(f,f"txn_{sources[idx]}",f"exp_t{idx}")

elif page_key == "Reconciliation":
    st.title("Smart Reconciliation Engine")
    st.caption("3-way match: CRM / PSP / Bank")
    m=len(df_rec[df_rec["Status"]=="Matched"]); p=len(df_rec[df_rec["Status"]=="Partial"]); e=len(df_rec[df_rec["Status"]=="Exception"])
    c1,c2,c3,c4=st.columns(4); c1.metric("Matched",m); c2.metric("Partial",p); c3.metric("Exceptions",e,delta_color="inverse"); c4.metric("Rate",f"{m/len(df_rec)*100:.1f}%")

    tab1,tab2,tab3=st.tabs(["Matched","Partial","Exceptions"])
    for tab,status in [(tab1,"Matched"),(tab2,"Partial"),(tab3,"Exception")]:
        with tab:
            data=df_rec[df_rec["Status"]==status]
            st.dataframe(data[["Case_ID","TXN_ID","CRM_Amt","PSP_Amt","Bank_Amt","CCY","Score","Method","Case_Status","Notes"]],use_container_width=True,hide_index=True,column_config={"Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d%%")})

    st.markdown("---")
    st.subheader("Case Actions")
    open_cases=df_rec[df_rec["Case_Status"].isin(["Open","Investigating"])]
    if len(open_cases)>0:
        sel=st.selectbox("Select case",open_cases["Case_ID"].tolist())
        cs=df_rec[df_rec["Case_ID"]==sel].iloc[0]
        st.info(f"**{cs['Case_ID']}** - {cs['Notes']}")
        sla_bar(cs["Created"],cs["SLA_Hours"])

        # Assignment
        team=[u["name"] for u in USERS.values()]
        assigned=st.session_state.case_assignments.get(sel,"Unassigned")
        new_assign=st.selectbox("Assign to",["Unassigned"]+team,index=(["Unassigned"]+team).index(assigned) if assigned in ["Unassigned"]+team else 0,key=f"assign_{sel}")
        if new_assign!=assigned:
            st.session_state.case_assignments[sel]=new_assign
            add_audit(f"Assigned {sel} to {new_assign}","Reconciliation"); add_feed(f"Assigned {sel} to {new_assign}","recon")

        # Bulk actions
        bc1,bc2,bc3=st.columns(3)
        if bc1.button("Investigate",key="inv",use_container_width=True): add_audit(f"Investigated {sel}","Reconciliation"); add_feed(f"Investigating {sel}","recon")
        if bc2.button("Approve",key="app",use_container_width=True):
            if st.session_state.role in ["CFO","Finance Manager"]:
                st.session_state.case_approvals[sel]={"approved_by":st.session_state.user_name,"time":datetime.now().strftime("%H:%M")}
                add_audit(f"Approved {sel}","Reconciliation"); add_feed(f"Approved {sel}","recon"); st.success("Approved")
            else: st.error("Manager approval required")
        if bc3.button("Reject",key="rej",use_container_width=True): add_audit(f"Rejected {sel}","Reconciliation"); add_feed(f"Rejected {sel}","recon")

        # Approval status
        if sel in st.session_state.case_approvals:
            ap=st.session_state.case_approvals[sel]
            st.success(f"Approved by {ap['approved_by']} at {ap['time']}")

        show_comments(sel)

    # Bulk actions
    st.subheader("Bulk Actions")
    bulk_cases=st.multiselect("Select multiple cases",df_rec["Case_ID"].tolist())
    if bulk_cases:
        bc1,bc2=st.columns(2)
        if bc1.button("Bulk Investigate",use_container_width=True):
            for c in bulk_cases: add_audit(f"Bulk investigated {c}","Reconciliation")
            add_feed(f"Bulk investigated {len(bulk_cases)} cases","recon"); st.success(f"Investigated {len(bulk_cases)} cases")
        if bc2.button("Bulk Close",use_container_width=True):
            for c in bulk_cases: add_audit(f"Bulk closed {c}","Reconciliation")
            add_feed(f"Bulk closed {len(bulk_cases)} cases","recon"); st.success(f"Closed {len(bulk_cases)} cases")

elif page_key == "Ledger":
    st.title("Ledger - Single Source of Truth")
    dr=df_led["Debit"].sum(); cr=df_led["Credit"].sum()
    c1,c2,c3=st.columns(3); c1.metric("Debits",fmt(dr)); c2.metric("Credits",fmt(cr)); c3.metric("Balance","OK" if abs(dr-cr)<0.01 else "IMBALANCED")
    af=st.selectbox("Account",["All"]+sorted(df_led["Account"].unique().tolist()))
    view=df_led if af=="All" else df_led[df_led["Account"]==af]
    st.dataframe(view,use_container_width=True,hide_index=True); export_df(view,"Ledger","exp_led")
    tb=df_led.groupby("Account").agg(Debits=("Debit","sum"),Credits=("Credit","sum")).reset_index(); tb["Net"]=tb["Debits"]-tb["Credits"]
    st.subheader("Trial Balance"); st.dataframe(tb,use_container_width=True,hide_index=True)

elif page_key == "Liquidity":
    st.title("Liquidity Intelligence")
    c1,c2,c3,c4=st.columns(4); c1.metric("Available",fmt(available_cash)); c2.metric("Pending",fmt(pending_wd),delta_color="inverse")
    c3.metric("Liabilities",fmt(bonus_liability+commission_liability),delta_color="inverse"); c4.metric("Net Liquidity",fmt(net_liquidity))
    wf=["Banks","PSP","Pending","Bonus","Commission","Net"]
    wv=[total_bank,total_psp,-pending_wd,-bonus_liability,-commission_liability,net_liquidity]
    fig=go.Figure(go.Waterfall(x=wf,y=wv,measure=["relative"]*5+["total"],increasing={"marker":{"color":"#10b981"}},decreasing={"marker":{"color":"#ef4444"}},totals={"marker":{"color":"#6366f1"}}))
    fig.update_layout(**PL,height=280); st.plotly_chart(fig,use_container_width=True)
    cl,cr=st.columns(2)
    with cl: st.subheader("Banks"); st.dataframe(df_bank,use_container_width=True,hide_index=True)
    with cr: st.subheader("PSPs"); st.dataframe(df_psp[["PSP","CCY","Balance","Pending_In","Pending_Out"]],use_container_width=True,hide_index=True)

elif page_key == "PSP Scorecard":
    st.title("PSP Performance Scorecard")
    ps=df_psp.groupby("PSP").agg(Balance=("Balance","sum"),Pending=("Pending_Out","sum"),Success=("Success_Rate","mean"),Settle=("Avg_Settle","first"),Cost=("Cost","mean")).reset_index()
    for _,p in ps.iterrows():
        st.subheader(p["PSP"])
        p1,p2,p3,p4,p5=st.columns(5); p1.metric("Balance",fmt(p["Balance"])); p2.metric("Pending",fmt(p["Pending"])); p3.metric("Success",f"{p['Success']:.1f}%"); p4.metric("Settle",p["Settle"]); p5.metric("Cost/Txn",f"${p['Cost']:.2f}")
    pc1,pc2=st.columns(2)
    with pc1: fig=px.bar(ps,x="PSP",y="Success",color="PSP",color_discrete_sequence=["#10b981","#6366f1","#f59e0b"]); fig.update_layout(**PL,height=240,showlegend=False,title="Success Rate"); st.plotly_chart(fig,use_container_width=True)
    with pc2: fig2=px.bar(ps,x="PSP",y="Cost",color="PSP",color_discrete_sequence=["#ef4444","#f59e0b","#10b981"]); fig2.update_layout(**PL,height=240,showlegend=False,title="Cost/Txn"); st.plotly_chart(fig2,use_container_width=True)

elif "Alerts" in page_key:
    st.title("Alerts & Exceptions")
    c1,c2,c3,c4=st.columns(4); c1.metric("Critical",len(df_alerts[(df_alerts["Severity"]=="Critical")&(df_alerts["Status"]!="Resolved")])); c2.metric("Open",len(df_alerts[df_alerts["Status"]=="Open"])); c3.metric("Investigating",len(df_alerts[df_alerts["Status"]=="Investigating"])); c4.metric("Resolved",len(df_alerts[df_alerts["Status"]=="Resolved"]))
    tab_all,tab_fin,tab_ops,tab_comp=st.tabs(["All","Financial","Operational","Compliance"])
    def show_al(data):
        for _,al in data.iterrows():
            sev_color="red" if al["Severity"] in ["Critical","High"] else "yellow"
            st.markdown(f'<div class="alert-card"><div class="alert-title">{al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],sev_color)} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow" if al["Status"]!="Resolved" else "green")}</div></div>',unsafe_allow_html=True)
            sla_bar(al["Created"],al["SLA_Hours"])
    with tab_all: show_al(df_alerts)
    with tab_fin: show_al(df_alerts[df_alerts["Category"]=="Financial"])
    with tab_ops: show_al(df_alerts[df_alerts["Category"]=="Operational"])
    with tab_comp: show_al(df_alerts[df_alerts["Category"]=="Compliance"])
    st.subheader("Case Management")
    active=df_alerts[df_alerts["Status"].isin(["Open","Investigating"])]
    if len(active)>0:
        sel=st.selectbox("Select",active["ID"].tolist()); al=df_alerts[df_alerts["ID"]==sel].iloc[0]
        st.info(f"**{al['Title']}** - {al['Description']}")
        bc1,bc2,bc3=st.columns(3)
        if bc1.button("Investigate",key="inv3",use_container_width=True): add_audit(f"Investigating {sel}","Alerts"); add_feed(f"Investigating {sel}","alert")
        if bc2.button("Resolve",key="res3",use_container_width=True): add_audit(f"Resolved {sel}","Alerts"); add_feed(f"Resolved {sel}","alert")
        bc3.button("Dismiss",key="dis3",use_container_width=True)
        show_comments(sel)

elif page_key == "Reports":
    st.title("Reports & Analytics")
    tab_d,tab_p,tab_k=st.tabs(["Daily/Weekly/Monthly","Profitability","KPIs"])
    with tab_d:
        rpt=st.radio("",["Daily","Weekly","Monthly"],horizontal=True)
        if rpt=="Daily":
            d1,d2,d3,d4=st.columns(4); d1.metric("Opening",fmt(opening_balance)); d2.metric("In",fmt(cash_in)); d3.metric("Out",fmt(cash_out)); d4.metric("Net",fmt(net_flow))
            fig=go.Figure(); fig.add_trace(go.Bar(x=["In","Out","Net"],y=[cash_in,cash_out,net_flow],marker_color=["#10b981","#ef4444","#6366f1"])); fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
        elif rpt=="Weekly":
            fig=go.Figure(); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Net"],mode="lines+markers",line=dict(color="#6366f1",width=2.5))); fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
        else:
            fig=px.treemap(df_profit,path=["Client"],values="Profit",color="Profit",color_continuous_scale=["#ef4444","#f59e0b","#10b981"]); fig.update_layout(**PL,height=300); st.plotly_chart(fig,use_container_width=True)
    with tab_p:
        pt=st.tabs(["Client","IB","Campaigns"])
        with pt[0]:
            fig=go.Figure(); fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Revenue"],name="Revenue",marker_color="#10b981")); fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Costs"],name="Costs",marker_color="#ef4444"))
            fig.update_layout(**PL,height=300,barmode="group"); st.plotly_chart(fig,use_container_width=True)
            st.dataframe(df_profit,use_container_width=True,hide_index=True); export_df(df_profit,"Profitability","exp_prof")
        with pt[1]:
            fig=go.Figure(); fig.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Net_Revenue"],name="Revenue",marker_color="#10b981")); fig.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Commission"],name="Commission",marker_color="#8b5cf6"))
            fig.update_layout(**PL,height=300,barmode="group"); st.plotly_chart(fig,use_container_width=True)
        with pt[2]:
            camps=pd.DataFrame([{"Campaign":"Welcome Bonus","Spend":45000,"Revenue":180000,"ROI":300},{"Campaign":"Loyalty","Spend":22000,"Revenue":95000,"ROI":332},{"Campaign":"Referral","Spend":15000,"Revenue":72000,"ROI":380}])
            st.dataframe(camps,use_container_width=True,hide_index=True)
    with tab_k:
        kc1,kc2=st.columns(2)
        with kc1: fig=px.line(df_kpi,x="Month",y="Net_Flow",markers=True,color_discrete_sequence=["#6366f1"]); fig.update_layout(**PL,height=240,title="Net Flow"); st.plotly_chart(fig,use_container_width=True)
        with kc2: fig=px.line(df_kpi,x="Month",y="Recon_Rate",markers=True,color_discrete_sequence=["#10b981"]); fig.add_hline(y=95,line_dash="dash",line_color="#f59e0b"); fig.update_layout(**PL,height=240,title="Recon Rate"); st.plotly_chart(fig,use_container_width=True)
        st.dataframe(df_kpi,use_container_width=True,hide_index=True); export_df(df_kpi,"KPIs","exp_kpi")

elif page_key == "Cash Forecast":
    st.title("Cash Flow Forecast")
    avg_d=df_cash["Deposits"].mean(); avg_w=df_cash["Withdrawals"].mean(); std_d=df_cash["Deposits"].std()
    np.random.seed(42)
    fc=[{"Date":f"May {19+i}","Net":avg_d-avg_w+np.random.normal(0,std_d*0.3)} for i in range(7)]
    fig=go.Figure(); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Net"],mode="lines+markers",name="Actual",line=dict(color="#6366f1",width=2.5)))
    fdf=pd.DataFrame(fc); fig.add_trace(go.Scatter(x=fdf["Date"],y=fdf["Net"],mode="lines+markers",name="Forecast",line=dict(color="#6366f1",width=2,dash="dot")))
    fig.update_layout(**PL,height=320,title="Net Flow - Actual + 7-Day Forecast"); st.plotly_chart(fig,use_container_width=True)
    st.dataframe(fdf,use_container_width=True,hide_index=True,column_config={"Net":st.column_config.NumberColumn(format="$%.0f")})

elif page_key == "Risk Monitor":
    st.title("Risk Early Warning System")
    liq_risk="Red" if buffer_pct<15 else "Yellow" if buffer_pct<20 else "Green"
    ops_risk="Red" if unmatched>3 else "Yellow" if unmatched>1 else "Green"
    settle_risk="Yellow" if len(df_alerts[df_alerts["Title"].str.contains("Delay",case=False)&(df_alerts["Status"]!="Resolved")])>0 else "Green"
    for name,status in [("Liquidity Risk",liq_risk),("Settlement Risk",settle_risk),("Operational Risk",ops_risk)]:
        color={"Green":"#10b981","Yellow":"#f59e0b","Red":"#ef4444"}[status]
        st.markdown(f'<div class="risk-row"><span class="risk-label">{name}</span><span class="risk-val" style="color:{color}">{status}</span></div>',unsafe_allow_html=True)
    st.subheader("Scenario Analysis")
    inc=st.slider("Withdrawal increase %",0,200,50,10); nwd=pending_wd*(1+inc/100); nliq=available_cash-nwd-bonus_liability-commission_liability
    sc1,sc2=st.columns(2); sc1.metric("New Net Liquidity",fmt(nliq)); sc2.metric("New Buffer",f"{nliq/available_cash*100:.1f}%")

elif page_key == "AI Anomaly":
    st.title("AI Anomaly Detection")
    st.caption("Automated pattern analysis on transaction data")
    anomalies=detect_anomalies(df_txn)
    if len(anomalies)>0:
        st.warning(f"{len(anomalies)} anomalies detected")
        st.dataframe(anomalies,use_container_width=True,hide_index=True,column_config={"Amount":st.column_config.NumberColumn(format="$%.0f")})
        fig=px.scatter(anomalies,x="Client",y="Amount",color="Risk",color_discrete_map={"High":"#ef4444","Medium":"#f59e0b"},size="Amount")
        fig.update_layout(**PL,height=280,title="Anomaly Map"); st.plotly_chart(fig,use_container_width=True)
    else: st.success("No anomalies detected")

elif page_key == "Client Scoring":
    st.title("Client Scoring Model")
    scores=client_scores(df_txn,df_profit)
    st.dataframe(scores,use_container_width=True,hide_index=True,column_config={"Volume":st.column_config.NumberColumn(format="$%.0f"),"Risk_Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d")})
    fig=px.bar(scores,x="Client",y="Risk_Score",color=scores["Risk_Score"].apply(lambda x:"Good" if x>70 else "Medium" if x>40 else "Poor"),color_discrete_map={"Good":"#10b981","Medium":"#f59e0b","Poor":"#ef4444"})
    fig.update_layout(**PL,height=260,showlegend=False,title="Client Risk Scores"); st.plotly_chart(fig,use_container_width=True)
    export_df(scores,"Client_Scores","exp_scores")

elif page_key == "Data Quality":
    st.title("Data Quality Monitor")
    all_issues=[]
    for df,name in [(df_txn,"Transactions"),(df_rec,"Reconciliation"),(df_led,"Ledger"),(df_alerts,"Alerts")]:
        all_issues.extend(check_data_quality(df,name))
    if all_issues:
        df_dq=pd.DataFrame(all_issues)
        st.warning(f"{len(all_issues)} data quality issues found")
        st.dataframe(df_dq,use_container_width=True,hide_index=True)
    else:
        st.success("All datasets pass quality checks")
    st.subheader("Dataset Summary")
    summary=pd.DataFrame([{"Dataset":"Transactions","Rows":len(df_txn),"Columns":len(df_txn.columns)},{"Dataset":"Reconciliation","Rows":len(df_rec),"Columns":len(df_rec.columns)},{"Dataset":"Ledger","Rows":len(df_led),"Columns":len(df_led.columns)},{"Dataset":"Alerts","Rows":len(df_alerts),"Columns":len(df_alerts.columns)}])
    st.dataframe(summary,use_container_width=True,hide_index=True)

elif page_key == "Scheduled Jobs":
    st.title("Scheduled Jobs Dashboard")
    df_jobs=pd.DataFrame(scheduled_jobs)
    running=len(df_jobs[df_jobs["Status"]=="Success"]); warnings=len(df_jobs[df_jobs["Status"]=="Warning"])
    c1,c2,c3=st.columns(3); c1.metric("Total Jobs",len(df_jobs)); c2.metric("Healthy",running); c3.metric("Warnings",warnings,delta_color="inverse")
    for _,job in df_jobs.iterrows():
        dot="dot-g" if job["Status"]=="Success" else "dot-y"
        st.markdown(f'<div class="risk-row"><span class="risk-label"><span class="status-dot {dot}"></span>{job["Job"]}</span><span class="sm">{job["Schedule"]} | Last: {job["Last_Run"]} | {job["Duration"]} | {job["Records"]} records</span></div>',unsafe_allow_html=True)

elif page_key == "Custom Dashboard":
    st.title("Custom Dashboard Builder")
    st.caption("Select which metrics to display")
    available_widgets=["Opening Balance","Cash In","Cash Out","Net Flow","Available Cash","Pending WD","Net Liquidity","Buffer %","Recon Rate","Unmatched","Open Alerts","Daily Volume"]
    selected=st.multiselect("Choose metrics",available_widgets,default=available_widgets[:6])
    widget_values={"Opening Balance":fmt(opening_balance),"Cash In":fmt(cash_in),"Cash Out":fmt(cash_out),"Net Flow":fmt(net_flow),"Available Cash":fmt(available_cash),"Pending WD":fmt(pending_wd),"Net Liquidity":fmt(net_liquidity),"Buffer %":f"{buffer_pct:.1f}%","Recon Rate":f"{recon_rate:.1f}%","Unmatched":str(unmatched),"Open Alerts":str(open_alert_count),"Daily Volume":f"{len(df_txn[df_txn['Value_Date']==today])}"}
    if selected:
        cols=st.columns(min(len(selected),4))
        for i,w in enumerate(selected):
            cols[i%len(cols)].metric(w,widget_values.get(w,"N/A"))
    show_chart=st.multiselect("Add charts",["Cash Flow","Net Flow Bar","Recon Pie"])
    if "Cash Flow" in show_chart:
        fig=go.Figure(); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Deposits"],mode="lines",name="In",line=dict(color="#10b981"))); fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Withdrawals"],mode="lines",name="Out",line=dict(color="#ef4444")))
        fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
    if "Net Flow Bar" in show_chart:
        fig=px.bar(df_cash,x="Date",y="Net",color_discrete_sequence=["#6366f1"]); fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)
    if "Recon Pie" in show_chart:
        rp=pd.DataFrame({"S":["Matched","Partial","Exception"],"C":[matched_count,len(df_rec[df_rec["Status"]=="Partial"]),len(df_rec[df_rec["Status"]=="Exception"])]})
        fig=px.pie(rp,values="C",names="S",hole=0.5,color_discrete_map={"Matched":"#10b981","Partial":"#f59e0b","Exception":"#ef4444"}); fig.update_layout(**PL,height=240); st.plotly_chart(fig,use_container_width=True)

elif page_key == "Audit Log":
    st.title("Audit Log - SOX Compliance")
    df_audit=pd.DataFrame(st.session_state.audit_log)
    sf=st.selectbox("Section",["All"]+sorted(df_audit["Section"].unique().tolist())); uf=st.selectbox("User",["All"]+sorted(df_audit["User"].unique().tolist()))
    view=df_audit.copy()
    if sf!="All": view=view[view["Section"]==sf]
    if uf!="All": view=view[view["User"]==uf]
    st.dataframe(view,use_container_width=True,hide_index=True); export_df(view,"Audit_Log","exp_audit")

elif page_key == "Architecture":
    st.title("System Architecture")
    layers=[("1. Connected Systems",["CRM","PSP Gateways","Banks","Trading","Bonus","Commission"]),("2. Data Integration",["API","File Import","Scheduler","Validation"]),("3. Normalization",["Currency","Time","ID Map","Type Map"]),("4. Reconciliation",["ID Match","Amount Match","Time Window","Exceptions"]),("5. Ledger",["Client Funds","Cash","Fees","Commissions"])]
    for title,items in layers:
        st.subheader(title)
        cols=st.columns(len(items) if len(items)<=4 else 3)
        for i,item in enumerate(items):
            cols[i%len(cols)].markdown(f'<div class="card" style="text-align:center;min-height:40px"><div class="sm" style="color:{TXT};font-weight:500">{item}</div></div>',unsafe_allow_html=True)
        st.markdown('<div class="flow-arrow">v</div>',unsafe_allow_html=True)

elif page_key == "Integrations":
    st.title("Integrations")
    for name,proto,health,sync in [("CRM","REST","Online","2m"),("Stripe","REST+WH","Online","30s"),("Adyen","REST","Warning","4h"),("Worldpay","REST","Online","5m"),("JPMorgan","SWIFT","Online","15m"),("HSBC","SWIFT","Online","15m"),("Deutsche Bank","SWIFT","Online","15m"),("Barclays","SWIFT","Online","15m"),("Trading","WebSocket","Online","RT"),("Bonus Engine","REST","Online","18m"),("Commission","REST","Online","10m")]:
        dot="dot-g" if health=="Online" else "dot-y"
        st.markdown(f'<div class="risk-row"><span class="risk-label"><span class="status-dot {dot}"></span>{name}</span><span class="sm">{proto} | {sync}</span></div>',unsafe_allow_html=True)

elif page_key == "Data Import":
    st.title("Data Import Wizard")
    st.caption("Upload a CSV and map columns to system fields")
    uf=st.file_uploader("Upload CSV",type=["csv"])
    if uf:
        df=pd.read_csv(uf); st.success(f"Loaded {len(df)} rows"); st.dataframe(df.head(),use_container_width=True,hide_index=True)
        st.subheader("Column Mapping")
        system_fields=["ID","Type","Source","Amount","CCY","Status","Client","Counterparty","Bank","Timestamp","Value_Date","Description","-- Skip --"]
        mapping={}
        cols=st.columns(3)
        for i,col_name in enumerate(df.columns):
            with cols[i%3]:
                mapping[col_name]=st.selectbox(f"{col_name} ->",system_fields,index=len(system_fields)-1,key=f"map_{col_name}")
        if st.button("Import Data",use_container_width=True,type="primary"):
            mapped={v:k for k,v in mapping.items() if v!="-- Skip --"}
            st.success(f"Mapped {len(mapped)} fields: {', '.join(mapped.keys())}")
            add_audit(f"Imported {len(df)} rows from {uf.name}","Data Import"); add_feed(f"Imported {len(df)} rows","txn")

elif page_key == "Settings":
    st.title("Settings")
    tab1,tab2,tab3=st.tabs(["General","Users","Rules"])
    with tab1:
        st.text_input("Company",value="FinanceOps Corp"); st.selectbox("Base CCY",["USD","EUR","GBP"],key="s_ccy")
        st.number_input("Session Timeout (min)",value=SESSION_TIMEOUT_MIN,step=5); st.number_input("Buffer Threshold %",value=15.0,step=0.5)
        st.button("Save",use_container_width=True,key="sg")
    with tab2:
        st.dataframe(pd.DataFrame([{"User":k,"Name":v["name"],"Role":v["role"]} for k,v in USERS.items()]),use_container_width=True,hide_index=True)
    with tab3:
        st.number_input("WD Spike %",value=200,step=10); st.number_input("Recon Threshold %",value=5.0,step=0.5); st.number_input("Amount Tolerance %",value=1.0,step=0.1)
        st.button("Save",use_container_width=True,key="sr")

elif page_key == "File Upload":
    st.title("File Upload")
    uploaded=st.file_uploader("Drop files",type=["docx","xlsx","xls","csv","pdf","txt"],accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            ext=uf.name.rsplit(".",1)[-1].lower(); st.markdown("---"); st.subheader(uf.name)
            try:
                if ext=="csv": df=pd.read_csv(uf); st.dataframe(df,use_container_width=True,hide_index=True); export_df(df,uf.name,f"exp_{uf.name}")
                elif ext in ["xlsx","xls"]:
                    xls=pd.ExcelFile(uf); sheet=st.selectbox("Sheet",xls.sheet_names,key=f"sh_{uf.name}"); df=pd.read_excel(uf,sheet_name=sheet); st.dataframe(df,use_container_width=True,hide_index=True)
                elif ext=="docx":
                    from docx import Document; doc=Document(uf)
                    for p in doc.paragraphs:
                        if p.text.strip(): st.markdown(p.text)
                elif ext=="pdf":
                    from PyPDF2 import PdfReader
                    for pg in PdfReader(uf).pages:
                        txt=pg.extract_text()
                        if txt: st.text(txt)
                elif ext=="txt": st.text(uf.read().decode("utf-8",errors="replace"))
            except Exception as e: st.error(str(e))
    else: st.info("Upload .docx, .xlsx, .csv, .pdf, or .txt files")
