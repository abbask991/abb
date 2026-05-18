"""
FinanceOps — Financial Control & Intelligence Platform
Core: Finance Automation · Smart Reconciliation · Financial Intelligence · Risk Early Warning
Standards: IFRS · ISO 20022 · ISO 4217 · Basel III · SOX
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

st.set_page_config(page_title="FinanceOps", page_icon="🏛️", layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════
st.markdown("""<style>
.main .block-container{padding-top:1rem}
div[data-testid="stMetric"]{background:#0f1729;border:1px solid #1e2d4a;border-radius:10px;padding:14px 18px}
div[data-testid="stMetric"] label{font-size:11px!important;text-transform:uppercase;letter-spacing:.5px}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:11px;font-weight:600}
.bg-green{background:rgba(16,185,129,.12);color:#10b981}.bg-red{background:rgba(239,68,68,.12);color:#ef4444}
.bg-yellow{background:rgba(245,158,11,.12);color:#f59e0b}.bg-blue{background:rgba(59,130,246,.12);color:#3b82f6}
.bg-purple{background:rgba(139,92,246,.12);color:#8b5cf6}.bg-cyan{background:rgba(6,182,212,.12);color:#06b6d4}
.card{background:#0f1729;border:1px solid #1e2d4a;border-radius:10px;padding:16px 18px;margin-bottom:8px}
.card h4{font-size:14px;font-weight:700;margin-bottom:8px;color:#e2e8f0}
.sm{font-size:12px;color:#94a3b8}
.alert-card{background:#0f1729;border:1px solid #1e2d4a;border-radius:8px;padding:14px 18px;margin-bottom:8px}
.alert-title{font-weight:600;font-size:13px;margin-bottom:3px}
.alert-desc{font-size:12px;color:#94a3b8;line-height:1.5}
.metric-box{background:#0f1729;border:1px solid #1e2d4a;border-radius:8px;padding:12px;text-align:center}
.metric-box .val{font-size:20px;font-weight:700;color:#e2e8f0}
.metric-box .lbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.5px}
.flow-arrow{text-align:center;font-size:20px;color:#6366f1;margin:4px 0}
.risk-row{display:flex;justify-content:space-between;align-items:center;padding:10px 14px;background:#0f1729;border:1px solid #1e2d4a;border-radius:8px;margin-bottom:6px}
.risk-label{font-size:13px;color:#e2e8f0}
.risk-val{font-weight:700;font-size:13px}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
.dot-g{background:#10b981}.dot-y{background:#f59e0b}.dot-r{background:#ef4444}
div[data-testid="stSidebar"]>div:first-child{padding-top:.8rem}
</style>""", unsafe_allow_html=True)

PL = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=10,b=0))

def fmt(n): return f"${n:,.0f}"
def badge(t, c="blue"): return f'<span class="badge bg-{c}">{t}</span>'
def sev_icon(s): return {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🔵"}.get(s,"⚪")
def risk_color(s): return {"Green":"#10b981","Yellow":"#f59e0b","Red":"#ef4444"}.get(s,"#94a3b8")

# ══════════════════════════════════════════════
# MOCK DATA
# ══════════════════════════════════════════════
transactions_data = [
    {"ID":"TXN-001","Type":"Deposit","Source":"CRM","Amount":25000,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Client deposit"},
    {"ID":"TXN-002","Type":"Withdrawal","Source":"CRM","Amount":12500,"CCY":"USD","Status":"Settled","Client":"Globe Ltd","Counterparty":"Adyen","Bank":"HSBC","Timestamp":"2026-05-18 09:15","Value_Date":"2026-05-18","Description":"Client withdrawal"},
    {"ID":"TXN-003","Type":"Deposit","Source":"PSP","Amount":8700,"CCY":"EUR","Status":"Pending","Client":"NovaTech","Counterparty":"Stripe","Bank":"Deutsche Bank","Timestamp":"2026-05-18 09:42","Value_Date":"2026-05-18","Description":"PSP settlement T+1"},
    {"ID":"TXN-004","Type":"Transfer","Source":"Bank","Amount":50000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"JPMorgan","Timestamp":"2026-05-18 10:01","Value_Date":"2026-05-18","Description":"Treasury rebalancing"},
    {"ID":"TXN-005","Type":"Fee","Source":"PSP","Amount":125,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-18 08:23","Value_Date":"2026-05-18","Description":"Processing fee 0.5%"},
    {"ID":"TXN-006","Type":"Commission","Source":"Commission","Amount":375,"CCY":"USD","Status":"Settled","Client":"IB-Alpha","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 10:30","Value_Date":"2026-05-18","Description":"IB commission payout"},
    {"ID":"TXN-007","Type":"Deposit","Source":"CRM","Amount":150000,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Worldpay","Bank":"Barclays","Timestamp":"2026-05-17 14:22","Value_Date":"2026-05-17","Description":"High-value deposit"},
    {"ID":"TXN-008","Type":"Withdrawal","Source":"CRM","Amount":45000,"CCY":"GBP","Status":"Failed","Client":"BritCo","Counterparty":"Adyen","Bank":"Barclays","Timestamp":"2026-05-17 16:45","Value_Date":"2026-05-17","Description":"Failed — insufficient PSP balance"},
    {"ID":"TXN-009","Type":"Deposit","Source":"PSP","Amount":32000,"CCY":"USD","Status":"Settled","Client":"SolarInc","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-17 11:10","Value_Date":"2026-05-17","Description":"PSP batch settlement"},
    {"ID":"TXN-010","Type":"Withdrawal","Source":"CRM","Amount":18200,"CCY":"EUR","Status":"Pending","Client":"EuroTrade","Counterparty":"Adyen","Bank":"Deutsche Bank","Timestamp":"2026-05-18 07:50","Value_Date":"2026-05-18","Description":"Pending withdrawal T+1"},
    {"ID":"TXN-011","Type":"Fee","Source":"PSP","Amount":89,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Worldpay","Bank":"Barclays","Timestamp":"2026-05-17 14:22","Value_Date":"2026-05-17","Description":"Processing fee"},
    {"ID":"TXN-012","Type":"Deposit","Source":"CRM","Amount":5600,"CCY":"USD","Status":"Reversed","Client":"QuickPay","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-16 09:30","Value_Date":"2026-05-16","Description":"Chargeback reversal"},
    {"ID":"TXN-013","Type":"Transfer","Source":"Bank","Amount":200000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"JPMorgan","Timestamp":"2026-05-16 15:00","Value_Date":"2026-05-16","Description":"Liquidity rebalancing"},
    {"ID":"TXN-014","Type":"Commission","Source":"Commission","Amount":1250,"CCY":"USD","Status":"Settled","Client":"IB-Beta","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-16 16:00","Value_Date":"2026-05-16","Description":"Monthly IB commission"},
    {"ID":"TXN-015","Type":"Deposit","Source":"CRM","Amount":72000,"CCY":"USD","Status":"Settled","Client":"TradeCo","Counterparty":"Stripe","Bank":"JPMorgan","Timestamp":"2026-05-15 10:20","Value_Date":"2026-05-15","Description":"Client deposit"},
    {"ID":"TXN-016","Type":"Bonus","Source":"Bonus","Amount":15000,"CCY":"USD","Status":"Settled","Client":"Acme Corp","Counterparty":"Bonus Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 11:00","Value_Date":"2026-05-18","Description":"Welcome bonus credit"},
    {"ID":"TXN-017","Type":"Bonus","Source":"Bonus","Amount":5000,"CCY":"USD","Status":"Settled","Client":"MegaFund","Counterparty":"Bonus Engine","Bank":"Barclays","Timestamp":"2026-05-17 09:00","Value_Date":"2026-05-17","Description":"Loyalty bonus"},
    {"ID":"TXN-018","Type":"Deposit","Source":"PSP","Amount":42000,"CCY":"EUR","Status":"Settled","Client":"EuroTrade","Counterparty":"Adyen","Bank":"Deutsche Bank","Timestamp":"2026-05-18 06:30","Value_Date":"2026-05-18","Description":"PSP settlement"},
    {"ID":"TXN-019","Type":"Transfer","Source":"Bank","Amount":85000,"CCY":"USD","Status":"Settled","Client":"Internal","Counterparty":"Treasury","Bank":"HSBC","Timestamp":"2026-05-18 08:00","Value_Date":"2026-05-18","Description":"Inter-bank transfer"},
    {"ID":"TXN-020","Type":"Commission","Source":"Commission","Amount":2100,"CCY":"USD","Status":"Pending","Client":"IB-Gamma","Counterparty":"IB Engine","Bank":"JPMorgan","Timestamp":"2026-05-18 11:15","Value_Date":"2026-05-18","Description":"Weekly IB accrual"},
]

reconciliation_data = [
    {"Case_ID":"REC-001","TXN_ID":"TXN-001","CRM_Amt":25000,"PSP_Amt":25000,"Bank_Amt":25000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched across CRM↔PSP↔Bank"},
    {"Case_ID":"REC-002","TXN_ID":"TXN-003","CRM_Amt":8700,"PSP_Amt":8700,"Bank_Amt":9000,"CCY":"EUR","Status":"Partial","Score":72,"Case_Status":"Investigating","Method":"ID+Time","Notes":"€300 bank variance — FX rounding suspected"},
    {"Case_ID":"REC-003","TXN_ID":"TXN-008","CRM_Amt":45000,"PSP_Amt":45000,"Bank_Amt":0,"CCY":"GBP","Status":"Exception","Score":0,"Case_Status":"Open","Method":"ID Match","Notes":"No bank record — PSP failed withdrawal"},
    {"Case_ID":"REC-004","TXN_ID":"TXN-007","CRM_Amt":150000,"PSP_Amt":150000,"Bank_Amt":150000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-005","TXN_ID":"TXN-012","CRM_Amt":5600,"PSP_Amt":5600,"Bank_Amt":5600,"CCY":"USD","Status":"Exception","Score":45,"Case_Status":"Investigating","Method":"Amount+Time","Notes":"Chargeback — PSP reversed, bank pending"},
    {"Case_ID":"REC-006","TXN_ID":"TXN-009","CRM_Amt":32000,"PSP_Amt":32000,"Bank_Amt":32000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-007","TXN_ID":"TXN-010","CRM_Amt":18200,"PSP_Amt":18200,"Bank_Amt":18500,"CCY":"EUR","Status":"Partial","Score":68,"Case_Status":"Open","Method":"ID+Time","Notes":"€300 discrepancy — fee or FX delta"},
    {"Case_ID":"REC-008","TXN_ID":"TXN-002","CRM_Amt":12500,"PSP_Amt":12500,"Bank_Amt":12500,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-009","TXN_ID":"TXN-015","CRM_Amt":72000,"PSP_Amt":72000,"Bank_Amt":72000,"CCY":"USD","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount","Notes":"Auto-matched"},
    {"Case_ID":"REC-010","TXN_ID":"TXN-018","CRM_Amt":42000,"PSP_Amt":42000,"Bank_Amt":42000,"CCY":"EUR","Status":"Matched","Score":100,"Case_Status":"Closed","Method":"ID+Amount+FX","Notes":"EUR matched at spot rate"},
]

ledger_data = [
    {"Date":"2026-05-18","Account":"Cash Account","Debit":25000,"Credit":0,"CCY":"USD","Ref":"TXN-001","Narration":"Deposit received — Acme Corp"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":25000,"CCY":"USD","Ref":"TXN-001","Narration":"Client funds credited"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":12500,"Credit":0,"CCY":"USD","Ref":"TXN-002","Narration":"Withdrawal — Globe Ltd"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":12500,"CCY":"USD","Ref":"TXN-002","Narration":"Cash disbursed"},
    {"Date":"2026-05-18","Account":"PSP Account","Debit":8700,"Credit":0,"CCY":"EUR","Ref":"TXN-003","Narration":"PSP receivable — Stripe T+1"},
    {"Date":"2026-05-18","Account":"Client Liability","Debit":0,"Credit":8700,"CCY":"EUR","Ref":"TXN-003","Narration":"Client funds — NovaTech"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":50000,"Credit":0,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer in"},
    {"Date":"2026-05-18","Account":"Cash Account","Debit":0,"Credit":50000,"CCY":"USD","Ref":"TXN-004","Narration":"Treasury transfer out (reserve)"},
    {"Date":"2026-05-18","Account":"Fee Account","Debit":0,"Credit":125,"CCY":"USD","Ref":"TXN-005","Narration":"Fee revenue recognized"},
    {"Date":"2026-05-18","Account":"Commission Account","Debit":375,"Credit":0,"CCY":"USD","Ref":"TXN-006","Narration":"IB commission expense"},
    {"Date":"2026-05-18","Account":"Commission Account","Debit":0,"Credit":375,"CCY":"USD","Ref":"TXN-006","Narration":"Commission payable — IB Alpha"},
    {"Date":"2026-05-17","Account":"Cash Account","Debit":150000,"Credit":0,"CCY":"USD","Ref":"TXN-007","Narration":"Deposit — MegaFund"},
    {"Date":"2026-05-17","Account":"Client Liability","Debit":0,"Credit":150000,"CCY":"USD","Ref":"TXN-007","Narration":"Client funds — MegaFund"},
    {"Date":"2026-05-17","Account":"Cash Account","Debit":32000,"Credit":0,"CCY":"USD","Ref":"TXN-009","Narration":"PSP settlement — SolarInc"},
    {"Date":"2026-05-17","Account":"Client Liability","Debit":0,"Credit":32000,"CCY":"USD","Ref":"TXN-009","Narration":"Client funds — SolarInc"},
]

alerts_data = [
    {"ID":"ALT-001","Title":"Cash Position Imbalance","Category":"Financial","Severity":"Critical","Status":"Open","Description":"GL cash balance deviates from bank by $12,400.","SLA":"4h","Linked":"TXN-004","Created":"2026-05-18 10:15"},
    {"ID":"ALT-002","Title":"Withdrawal Spike Detected","Category":"Financial","Severity":"High","Status":"Investigating","Description":"Withdrawal volume 340% above 30-day average.","SLA":"8h","Linked":"TXN-002, TXN-010","Created":"2026-05-18 09:30"},
    {"ID":"ALT-003","Title":"PSP Settlement Delay — Adyen","Category":"Operational","Severity":"Medium","Status":"Open","Description":"Adyen batch delayed 4h beyond SLA. Pending: €63,200.","SLA":"12h","Linked":"TXN-010","Created":"2026-05-18 08:00"},
    {"ID":"ALT-004","Title":"Reconciliation Exception Rate 20%","Category":"Operational","Severity":"High","Status":"Investigating","Description":"Above 5% threshold — 2 unresolved exceptions.","SLA":"4h","Linked":"REC-003, REC-005","Created":"2026-05-18 10:00"},
    {"ID":"ALT-005","Title":"Chargeback — Provision Required","Category":"Financial","Severity":"Medium","Status":"Resolved","Description":"TXN-012 chargeback booked. Provision updated.","SLA":"24h","Linked":"TXN-012","Created":"2026-05-16 10:30"},
    {"ID":"ALT-006","Title":"Liquidity Buffer Below Threshold","Category":"Financial","Severity":"Critical","Status":"Open","Description":"Buffer at 11.2% — below 15% minimum.","SLA":"2h","Linked":"","Created":"2026-05-18 07:00"},
    {"ID":"ALT-007","Title":"Unexplained Transaction Detected","Category":"Compliance","Severity":"High","Status":"Open","Description":"$35,000 credit with no matching CRM record. Requires AML review.","SLA":"4h","Linked":"","Created":"2026-05-18 11:00"},
]

bank_balances = [
    {"Bank":"JPMorgan Chase","CCY":"USD","Balance":2450000},
    {"Bank":"HSBC","CCY":"USD","Balance":890000},
    {"Bank":"Deutsche Bank","CCY":"EUR","Balance":1250000},
    {"Bank":"Barclays","CCY":"GBP","Balance":675000},
    {"Bank":"Barclays","CCY":"USD","Balance":320000},
]

psp_balances = [
    {"PSP":"Stripe","CCY":"USD","Balance":185000,"Pending_In":32000,"Pending_Out":8500},
    {"PSP":"Stripe","CCY":"EUR","Balance":45000,"Pending_In":8700,"Pending_Out":0},
    {"PSP":"Adyen","CCY":"USD","Balance":92000,"Pending_In":0,"Pending_Out":12500},
    {"PSP":"Adyen","CCY":"EUR","Balance":63200,"Pending_In":0,"Pending_Out":18200},
    {"PSP":"Worldpay","CCY":"USD","Balance":210000,"Pending_In":0,"Pending_Out":0},
    {"PSP":"Worldpay","CCY":"GBP","Balance":55000,"Pending_In":0,"Pending_Out":45000},
]

cash_flow = [
    {"Date":"May 12","Deposits":185000,"Withdrawals":72000,"Net":113000},
    {"Date":"May 13","Deposits":210000,"Withdrawals":95000,"Net":115000},
    {"Date":"May 14","Deposits":145000,"Withdrawals":120000,"Net":25000},
    {"Date":"May 15","Deposits":290000,"Withdrawals":88000,"Net":202000},
    {"Date":"May 16","Deposits":178000,"Withdrawals":156000,"Net":22000},
    {"Date":"May 17","Deposits":320000,"Withdrawals":105000,"Net":215000},
    {"Date":"May 18","Deposits":307700,"Withdrawals":75700,"Net":232000},
]

profitability = [
    {"Client":"Acme Corp","Revenue":45000,"Costs":12000,"Profit":33000},
    {"Client":"Globe Ltd","Revenue":28000,"Costs":9500,"Profit":18500},
    {"Client":"MegaFund","Revenue":82000,"Costs":18000,"Profit":64000},
    {"Client":"NovaTech","Revenue":15000,"Costs":6200,"Profit":8800},
    {"Client":"SolarInc","Revenue":32000,"Costs":8800,"Profit":23200},
    {"Client":"TradeCo","Revenue":51000,"Costs":14500,"Profit":36500},
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
df_cash = pd.DataFrame(cash_flow)
df_profit = pd.DataFrame(profitability)
df_ib = pd.DataFrame(ib_data)
df_kpi = pd.DataFrame(monthly_kpis)

# Computed values
opening_balance = 4250000
today = "2026-05-18"
cash_in = df_txn[(df_txn["Type"]=="Deposit") & (df_txn["Value_Date"]==today) & (df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
cash_out = df_txn[(df_txn["Type"].isin(["Withdrawal","Fee","Commission"])) & (df_txn["Value_Date"]==today) & (df_txn["Status"].isin(["Settled","Pending"]))]["Amount"].sum()
net_flow = cash_in - cash_out
total_bank = df_bank["Balance"].sum()
total_psp = df_psp["Balance"].sum()
available_cash = total_bank + total_psp
pending_wd = df_txn[(df_txn["Type"]=="Withdrawal")&(df_txn["Status"]=="Pending")]["Amount"].sum() + df_psp["Pending_Out"].sum()
bonus_liability = 125000
commission_liability = 48500
net_liquidity = available_cash - pending_wd - bonus_liability - commission_liability
buffer_pct = (net_liquidity / available_cash * 100) if available_cash > 0 else 0
matched = len(df_rec[df_rec["Status"]=="Matched"])
recon_rate = matched / len(df_rec) * 100 if len(df_rec) > 0 else 0
unmatched = len(df_rec[df_rec["Status"].isin(["Partial","Exception"])])
open_alerts = len(df_alerts[df_alerts["Status"].isin(["Open","Investigating"])])

# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏛️ **FinanceOps**")
    st.caption("Financial Control & Intelligence Platform")
    st.markdown("---")
    page = st.radio("", [
        "📊 Dashboard",
        "🏗️ System Architecture",
        "💸 Transactions",
        "🔄 Reconciliation",
        "📒 Ledger",
        "💧 Liquidity",
        f"⚠️ Alerts ({open_alerts})",
        "📈 Reports & Analytics",
        "🛡️ Risk Monitor",
        "🔌 Integrations",
        "⚙️ Settings",
        "📂 File Upload",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("🟢 **System Online**")
    st.caption(datetime.now().strftime("%a, %b %d, %Y — %H:%M"))
    st.markdown("---")
    st.caption("**Users Online**")
    st.caption("👔 CFO · 📊 Finance Mgr · ⚙️ Ops Mgr")


# ══════════════════════════════════════════════
# 1. DASHBOARD — Financial Control Tower
# ══════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Dashboard — Financial Control Tower")
    st.caption("Real-time financial overview for CFO · Finance Manager · Operations Manager")

    # Role selector
    role = st.radio("View as:", ["👔 CFO", "📊 Finance Manager", "⚙️ Operations Manager"], horizontal=True)

    # ── Financial Overview (all roles)
    st.subheader("💵 Financial Overview")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Opening Balance", fmt(opening_balance))
    f2.metric("Cash In Today", fmt(cash_in), f"+{cash_in/opening_balance*100:.1f}%")
    f3.metric("Cash Out Today", fmt(cash_out), f"-{cash_out/opening_balance*100:.1f}%", delta_color="inverse")
    f4.metric("Net Flow", fmt(net_flow), f"{'+'if net_flow>=0 else ''}{net_flow/opening_balance*100:.1f}%")

    # ── Liquidity (CFO + Finance Manager)
    if role in ["👔 CFO", "📊 Finance Manager"]:
        st.subheader("💧 Liquidity")
        l1,l2,l3,l4 = st.columns(4)
        l1.metric("Available Cash", fmt(available_cash))
        l2.metric("Pending Withdrawals", fmt(pending_wd), delta_color="inverse")
        l3.metric("PSP Balances", fmt(total_psp))
        l4.metric("Net Liquidity", fmt(net_liquidity), f"Buffer: {buffer_pct:.1f}%")

        cl,cr = st.columns(2)
        with cl:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Deposits"],mode="lines",name="Cash In",fill="tozeroy",line=dict(color="#10b981",width=2),fillcolor="rgba(16,185,129,.12)"))
            fig.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Withdrawals"],mode="lines",name="Cash Out",fill="tozeroy",line=dict(color="#ef4444",width=2),fillcolor="rgba(239,68,68,.12)"))
            fig.update_layout(**PL,height=240,legend=dict(orientation="h",y=1.1))
            st.plotly_chart(fig,use_container_width=True)
        with cr:
            fig2 = px.bar(df_cash,x="Date",y="Net",color_discrete_sequence=["#6366f1"])
            fig2.update_layout(**PL,height=240,showlegend=False)
            st.plotly_chart(fig2,use_container_width=True)

    # ── Dashboard Charts (all roles)
    st.subheader("📊 Quick Financial Charts")
    dc1,dc2,dc3 = st.columns(3)
    with dc1:
        type_counts = df_txn["Type"].value_counts().reset_index(); type_counts.columns=["Type","Count"]
        fig_mix = px.pie(type_counts,values="Count",names="Type",hole=0.5,color_discrete_sequence=["#10b981","#ef4444","#6366f1","#f59e0b","#8b5cf6","#06b6d4"])
        fig_mix.update_layout(**PL,height=220); fig_mix.update_traces(textinfo="value+percent")
        st.caption("Transaction Mix"); st.plotly_chart(fig_mix,use_container_width=True)
    with dc2:
        status_counts = df_txn["Status"].value_counts().reset_index(); status_counts.columns=["Status","Count"]
        fig_st = px.pie(status_counts,values="Count",names="Status",hole=0.5,color_discrete_map={"Settled":"#10b981","Pending":"#f59e0b","Failed":"#ef4444","Reversed":"#8b5cf6"})
        fig_st.update_layout(**PL,height=220); fig_st.update_traces(textinfo="value+percent")
        st.caption("Status Breakdown"); st.plotly_chart(fig_st,use_container_width=True)
    with dc3:
        psp_bal = df_psp.groupby("PSP")["Balance"].sum().reset_index()
        fig_psp = px.bar(psp_bal,x="PSP",y="Balance",color="PSP",color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig_psp.update_layout(**PL,height=220,showlegend=False)
        st.caption("PSP Balances"); st.plotly_chart(fig_psp,use_container_width=True)

    # Hourly activity
    df_txn["Hour"] = df_txn["Timestamp"].str.extract(r"(\d{2}):\d{2}").astype(int)
    hourly = df_txn.groupby("Hour").size().reset_index(name="Count")
    fig_hourly = px.bar(hourly,x="Hour",y="Count",color_discrete_sequence=["#6366f1"])
    fig_hourly.update_layout(**PL,height=180,showlegend=False,xaxis_title="Hour of Day",yaxis_title="Transactions")
    st.caption("Hourly Transaction Activity"); st.plotly_chart(fig_hourly,use_container_width=True)

    # ── Alerts (all roles)
    st.subheader("🚨 Alerts")
    exceptions_count = len(df_rec[df_rec["Status"]=="Exception"])
    delays = len(df_alerts[df_alerts["Title"].str.contains("Delay|Settlement",case=False)&(df_alerts["Status"]!="Resolved")])
    liq_warn = len(df_alerts[df_alerts["Title"].str.contains("Liquidity|Buffer",case=False)&(df_alerts["Status"]!="Resolved")])
    a1,a2,a3 = st.columns(3)
    a1.metric("Unresolved Exceptions", exceptions_count, delta_color="inverse")
    a2.metric("Settlement Delays", delays, delta_color="inverse")
    a3.metric("Liquidity Warnings", liq_warn, delta_color="inverse")

    for _, al in df_alerts[df_alerts["Status"].isin(["Open","Investigating"])].iterrows():
        st.markdown(f'<div class="alert-card"><div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow")} <span class="sm" style="margin-left:6px">SLA: {al["SLA"]}</span></div></div>', unsafe_allow_html=True)

    # ── KPIs (all roles)
    st.subheader("📈 KPIs")
    k1,k2,k3 = st.columns(3)
    k1.metric("Daily Volume", f"{len(df_txn[df_txn['Value_Date']==today])} transactions")
    k2.metric("Reconciliation Rate", f"{recon_rate:.1f}%", f"{matched}/{len(df_rec)} matched")
    k3.metric("Unmatched Transactions", unmatched, delta_color="inverse")

    # ── Financial Intelligence (CFO)
    if role == "👔 CFO":
        st.subheader("🧠 Financial Intelligence")
        cl2,cr2 = st.columns(2)
        with cl2:
            st.markdown("**Top Profitable Clients**")
            top = df_profit.nlargest(3,"Profit")
            for _,r in top.iterrows():
                st.markdown(f'<div class="risk-row"><span class="risk-label">{r["Client"]}</span><span class="risk-val" style="color:#10b981">{fmt(r["Profit"])}</span></div>',unsafe_allow_html=True)
        with cr2:
            st.markdown("**Top Costly Clients**")
            costly = df_profit.nlargest(3,"Costs")
            for _,r in costly.iterrows():
                st.markdown(f'<div class="risk-row"><span class="risk-label">{r["Client"]}</span><span class="risk-val" style="color:#ef4444">{fmt(r["Costs"])}</span></div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 2. SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════
elif page == "🏗️ System Architecture":
    st.title("🏗️ System Architecture")
    st.caption("End-to-end pipeline — Connected Systems → Ledger → Analytics → Users")

    layers = [
        ("1️⃣ Connected Systems", ["CRM", "PSP Gateways (Stripe, Adyen, Worldpay)", "Bank Accounts (JPMorgan, HSBC, Deutsche, Barclays)", "Trading Platform", "Bonus Engine", "Commission Engine"]),
        ("2️⃣ Data Integration Layer", ["API Connectors (REST, WebSocket, SWIFT)", "File Imports (MT940, CSV, XML)", "Scheduled Data Pull (real-time to hourly)", "Data Validation & Deduplication"]),
        ("3️⃣ Data Normalization", ["Currency Standardization (ISO 4217)", "Time Standardization (UTC / ISO 8601)", "ID Mapping (cross-system)", "Transaction Type Mapping"]),
        ("4️⃣ Reconciliation Engine", ["ID Match (CRM ↔ PSP ↔ Bank)", "Amount Match (±tolerance)", "Time Window Match (±24h)", "Auto Exception Detection & Case Creation"]),
        ("5️⃣ Central Financial Ledger", ["Client Funds", "Company Cash", "Fees", "Commissions", "Adjustments"]),
    ]

    for title, items in layers:
        st.subheader(title)
        cols = st.columns(len(items) if len(items) <= 4 else 3)
        for i, item in enumerate(items):
            cols[i % len(cols)].markdown(f'<div class="card" style="text-align:center;min-height:60px"><div class="sm" style="color:#e2e8f0;font-weight:500">{item}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)

    el,er = st.columns(2)
    with el:
        st.subheader("6a. Liquidity Engine")
        for label in ["Available Cash","Pending Withdrawals","PSP Holdbacks","Net Liquidity"]:
            st.markdown(f'<div class="risk-row"><span class="risk-label">{label}</span><span class="risk-val" style="color:#6366f1">Active</span></div>',unsafe_allow_html=True)
    with er:
        st.subheader("6b. Alert & Exception Engine")
        for label in ["Reconciliation Errors","PSP Delays","Cash Imbalance","Threshold Breach"]:
            st.markdown(f'<div class="risk-row"><span class="risk-label">{label}</span><span class="risk-val" style="color:#f59e0b">Monitoring</span></div>',unsafe_allow_html=True)

    st.markdown('<div class="flow-arrow">▼</div>', unsafe_allow_html=True)
    st.subheader("7️⃣ Reporting & Analytics → 8️⃣ End Users")
    u1,u2,u3,u4,u5 = st.columns(5)
    for col, (icon, role) in zip([u1,u2,u3,u4,u5], [("👔","CFO"),("📊","Finance Mgr"),("⚙️","Operations"),("🔄","Recon Team"),("🏢","Management")]):
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:24px">{icon}</div><h4 style="font-size:12px">{role}</h4></div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 3. TRANSACTIONS
# ══════════════════════════════════════════════
elif page == "💸 Transactions":
    st.title("💸 Transactions")
    st.caption("Source: CRM · PSP · Bank · Commission · Bonus")

    s1,s2,s3,s4,s5 = st.columns(5)
    s1.metric("CRM",len(df_txn[df_txn["Source"]=="CRM"]))
    s2.metric("PSP",len(df_txn[df_txn["Source"]=="PSP"]))
    s3.metric("Bank",len(df_txn[df_txn["Source"]=="Bank"]))
    s4.metric("Commission",len(df_txn[df_txn["Source"]=="Commission"]))
    s5.metric("Bonus",len(df_txn[df_txn["Source"]=="Bonus"]))

    tabs = st.tabs(["All","CRM","PSP","Bank","Commission","Bonus"])
    sources = ["All","CRM","PSP","Bank","Commission","Bonus"]
    for idx,tab in enumerate(tabs):
        with tab:
            base = df_txn if sources[idx]=="All" else df_txn[df_txn["Source"]==sources[idx]]
            fc1,fc2,fc3,fc4,fc5,fc6 = st.columns(6)
            with fc1: search=st.text_input("Search",key=f"s{idx}",placeholder="ID, client...")
            with fc2: tf=st.selectbox("Type",["All"]+sorted(base["Type"].unique().tolist()),key=f"t{idx}")
            with fc3: sf=st.selectbox("Status",["All"]+sorted(base["Status"].unique().tolist()),key=f"st{idx}")
            with fc4: pf=st.selectbox("PSP/Counterparty",["All"]+sorted(base["Counterparty"].unique().tolist()),key=f"p{idx}")
            with fc5: bf=st.selectbox("Bank",["All"]+sorted(base["Bank"].unique().tolist()),key=f"b{idx}")
            with fc6: cf=st.selectbox("Currency",["All"]+sorted(base["CCY"].unique().tolist()),key=f"c{idx}")

            f = base.copy()
            if search: f=f[f.apply(lambda r:search.lower() in r["ID"].lower() or search.lower() in r["Client"].lower(),axis=1)]
            if tf!="All": f=f[f["Type"]==tf]
            if sf!="All": f=f[f["Status"]==sf]
            if pf!="All": f=f[f["Counterparty"]==pf]
            if bf!="All": f=f[f["Bank"]==bf]
            if cf!="All": f=f[f["CCY"]==cf]

            st.caption(f"{len(f)} of {len(base)} transactions")
            st.dataframe(f[["ID","Type","Source","Client","Amount","CCY","Counterparty","Bank","Status","Timestamp"]],use_container_width=True,hide_index=True,column_config={"Amount":st.column_config.NumberColumn(format="%.2f")})

    st.markdown("---")

    # Transaction Analytics Charts
    st.subheader("📊 Transaction Analytics")
    tc1,tc2 = st.columns(2)
    with tc1:
        src_vol = df_txn.groupby("Source")["Amount"].sum().reset_index().sort_values("Amount",ascending=True)
        fig_sv = px.bar(src_vol,y="Source",x="Amount",orientation="h",color="Source",color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#8b5cf6","#06b6d4"])
        fig_sv.update_layout(**PL,height=240,showlegend=False,title="Volume by Source")
        st.plotly_chart(fig_sv,use_container_width=True)
    with tc2:
        src_cnt = df_txn["Source"].value_counts().reset_index(); src_cnt.columns=["Source","Count"]
        fig_sc = px.pie(src_cnt,values="Count",names="Source",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#8b5cf6","#06b6d4"])
        fig_sc.update_layout(**PL,height=240); fig_sc.update_traces(textinfo="value+percent")
        fig_sc.update_layout(title="Transactions by Source")
        st.plotly_chart(fig_sc,use_container_width=True)

    tc3,tc4 = st.columns(2)
    with tc3:
        daily_vol = df_txn.groupby("Value_Date").size().reset_index(name="Count").sort_values("Value_Date")
        fig_dv = px.line(daily_vol,x="Value_Date",y="Count",markers=True,color_discrete_sequence=["#6366f1"])
        fig_dv.update_layout(**PL,height=220,title="Daily Transaction Volume")
        st.plotly_chart(fig_dv,use_container_width=True)
    with tc4:
        fig_hist = px.histogram(df_txn,x="Amount",nbins=15,color_discrete_sequence=["#8b5cf6"])
        fig_hist.update_layout(**PL,height=220,title="Amount Distribution")
        st.plotly_chart(fig_hist,use_container_width=True)

    tc5,tc6 = st.columns(2)
    with tc5:
        client_vol = df_txn.groupby("Client")["Amount"].sum().nlargest(8).reset_index()
        fig_cv = px.bar(client_vol,x="Client",y="Amount",color_discrete_sequence=["#10b981"])
        fig_cv.update_layout(**PL,height=220,showlegend=False,title="Top Clients by Volume")
        st.plotly_chart(fig_cv,use_container_width=True)
    with tc6:
        ccy_vol = df_txn.groupby("CCY")["Amount"].sum().reset_index()
        fig_ccv = px.pie(ccy_vol,values="Amount",names="CCY",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444"])
        fig_ccv.update_layout(**PL,height=220); fig_ccv.update_traces(textinfo="value+percent")
        fig_ccv.update_layout(title="Volume by Currency")
        st.plotly_chart(fig_ccv,use_container_width=True)

    st.markdown("---")
    st.subheader("Transaction Details")
    sel=st.selectbox("Select transaction",df_txn["ID"].tolist())
    if sel:
        tx=df_txn[df_txn["ID"]==sel].iloc[0]
        c1,c2,c3=st.columns(3)
        c1.markdown(f"**ID:** {tx['ID']}\n\n**Type:** {tx['Type']}\n\n**Source:** {tx['Source']}\n\n**Amount:** {tx['Amount']:,.2f} {tx['CCY']}")
        c2.markdown(f"**Status:** {tx['Status']}\n\n**Client:** {tx['Client']}\n\n**Counterparty:** {tx['Counterparty']}\n\n**Bank:** {tx['Bank']}")
        c3.markdown(f"**Timestamp:** {tx['Timestamp']}\n\n**Value Date:** {tx['Value_Date']}\n\n**Description:** {tx['Description']}")


# ══════════════════════════════════════════════
# 4. RECONCILIATION — Smart Engine
# ══════════════════════════════════════════════
elif page == "🔄 Reconciliation":
    st.title("🔄 Smart Reconciliation Engine")
    st.caption("3-way match: CRM ↔ PSP ↔ Bank — Auto case creation for exceptions")

    m=len(df_rec[df_rec["Status"]=="Matched"])
    p=len(df_rec[df_rec["Status"]=="Partial"])
    e=len(df_rec[df_rec["Status"]=="Exception"])
    c1,c2,c3,c4=st.columns(4)
    c1.metric("✅ Matched",m,f"{m/len(df_rec)*100:.0f}%")
    c2.metric("⚠️ Partial",p,f"{p/len(df_rec)*100:.0f}%")
    c3.metric("❌ Exceptions",e,delta_color="inverse")
    c4.metric("Recon Rate",f"{m/len(df_rec)*100:.1f}%")

    # Recon Charts
    rc1,rc2,rc3 = st.columns(3)
    with rc1:
        rec_pie = pd.DataFrame({"Status":["Matched","Partial","Exception"],"Count":[m,p,e]})
        fig_rp = px.pie(rec_pie,values="Count",names="Status",hole=0.55,color_discrete_map={"Matched":"#10b981","Partial":"#f59e0b","Exception":"#ef4444"})
        fig_rp.update_layout(**PL,height=220); fig_rp.update_traces(textinfo="value+percent")
        st.caption("Match Distribution"); st.plotly_chart(fig_rp,use_container_width=True)
    with rc2:
        fig_score = px.histogram(df_rec,x="Score",nbins=10,color_discrete_sequence=["#6366f1"])
        fig_score.update_layout(**PL,height=220,title="Score Distribution")
        st.plotly_chart(fig_score,use_container_width=True)
    with rc3:
        method_cnt = df_rec["Method"].value_counts().reset_index(); method_cnt.columns=["Method","Count"]
        fig_mc = px.bar(method_cnt,x="Method",y="Count",color_discrete_sequence=["#8b5cf6"])
        fig_mc.update_layout(**PL,height=220,showlegend=False,title="Match Methods Used")
        st.plotly_chart(fig_mc,use_container_width=True)

    # Variance chart
    variance_data = df_rec[df_rec["Status"].isin(["Partial","Exception"])].copy()
    if len(variance_data) > 0:
        variance_data["Variance"] = variance_data["Bank_Amt"] - variance_data["CRM_Amt"]
        fig_var = px.bar(variance_data,x="Case_ID",y="Variance",color=variance_data["Variance"].apply(lambda x:"Surplus" if x>0 else "Deficit" if x<0 else "Zero"),color_discrete_map={"Surplus":"#10b981","Deficit":"#ef4444","Zero":"#94a3b8"})
        fig_var.update_layout(**PL,height=200,showlegend=True,title="Variance Analysis (Bank − CRM)")
        st.plotly_chart(fig_var,use_container_width=True)

    tab1,tab2,tab3=st.tabs(["Matched","Partial","Exceptions"])
    for tab,status in [(tab1,"Matched"),(tab2,"Partial"),(tab3,"Exception")]:
        with tab:
            data=df_rec[df_rec["Status"]==status]
            st.dataframe(data[["Case_ID","TXN_ID","CRM_Amt","PSP_Amt","Bank_Amt","CCY","Score","Method","Case_Status","Notes"]],use_container_width=True,hide_index=True,
                column_config={"Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%d%%"),"CRM_Amt":st.column_config.NumberColumn(format="%.0f"),"PSP_Amt":st.column_config.NumberColumn(format="%.0f"),"Bank_Amt":st.column_config.NumberColumn(format="%.0f")})

    st.markdown("---")
    st.subheader("Case Actions")
    open_cases=df_rec[df_rec["Case_Status"].isin(["Open","Investigating"])]
    if len(open_cases)>0:
        sel=st.selectbox("Select case",open_cases["Case_ID"].tolist())
        cs=df_rec[df_rec["Case_ID"]==sel].iloc[0]
        st.info(f"**{cs['Case_ID']}** — {cs['Notes']} | CRM: {cs['CRM_Amt']:,.0f} | PSP: {cs['PSP_Amt']:,.0f} | Bank: {cs['Bank_Amt']:,.0f}")
        bc1,bc2,bc3=st.columns(3)
        bc1.button("🔍 Investigate",key="inv",use_container_width=True)
        bc2.button("✅ Approve",key="app",use_container_width=True)
        bc3.button("❌ Reject",key="rej",use_container_width=True)


# ══════════════════════════════════════════════
# 5. LEDGER — Single Source of Truth
# ══════════════════════════════════════════════
elif page == "📒 Ledger":
    st.title("📒 Ledger — Single Source of Truth")
    st.caption("Date · Account · Debit · Credit · Reference")

    total_dr=df_led["Debit"].sum()
    total_cr=df_led["Credit"].sum()
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Total Debits",fmt(total_dr))
    c2.metric("Total Credits",fmt(total_cr))
    c3.metric("Balance Check","✅ Balanced" if abs(total_dr-total_cr)<0.01 else "❌ Imbalanced")
    c4.metric("Entries",len(df_led))

    # Ledger Charts
    lc1,lc2 = st.columns(2)
    with lc1:
        acct_dr = df_led.groupby("Account")["Debit"].sum().reset_index()
        acct_cr = df_led.groupby("Account")["Credit"].sum().reset_index()
        fig_drcr = go.Figure()
        fig_drcr.add_trace(go.Bar(x=acct_dr["Account"],y=acct_dr["Debit"],name="Debit",marker_color="#06b6d4"))
        fig_drcr.add_trace(go.Bar(x=acct_cr["Account"],y=acct_cr["Credit"],name="Credit",marker_color="#8b5cf6"))
        fig_drcr.update_layout(**PL,height=260,barmode="group",title="Debit vs Credit by Account")
        st.plotly_chart(fig_drcr,use_container_width=True)
    with lc2:
        daily_entries = df_led.groupby("Date").size().reset_index(name="Entries")
        fig_de = px.bar(daily_entries,x="Date",y="Entries",color_discrete_sequence=["#6366f1"])
        fig_de.update_layout(**PL,height=260,showlegend=False,title="Daily Journal Volume")
        st.plotly_chart(fig_de,use_container_width=True)

    lc3,lc4 = st.columns(2)
    with lc3:
        acct_counts = df_led["Account"].value_counts().reset_index(); acct_counts.columns=["Account","Entries"]
        fig_ac = px.pie(acct_counts,values="Entries",names="Account",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6"])
        fig_ac.update_layout(**PL,height=240); fig_ac.update_traces(textinfo="value+percent")
        fig_ac.update_layout(title="Entries by Account")
        st.plotly_chart(fig_ac,use_container_width=True)
    with lc4:
        net_by_acct = df_led.groupby("Account").agg(Net=("Debit","sum")).reset_index()
        net_by_acct["Net"] = df_led.groupby("Account")["Debit"].sum().values - df_led.groupby("Account")["Credit"].sum().values
        fig_nb = px.bar(net_by_acct,x="Account",y="Net",color=net_by_acct["Net"].apply(lambda x:"Positive" if x>=0 else "Negative"),color_discrete_map={"Positive":"#10b981","Negative":"#ef4444"})
        fig_nb.update_layout(**PL,height=240,showlegend=False,title="Net Balance by Account")
        st.plotly_chart(fig_nb,use_container_width=True)

    acct_filter=st.selectbox("Filter by Account",["All","Client Liability","Cash Account","PSP Account","Fee Account","Commission Account"])
    view = df_led if acct_filter=="All" else df_led[df_led["Account"]==acct_filter]

    st.dataframe(view[["Date","Account","Debit","Credit","CCY","Ref","Narration"]],use_container_width=True,hide_index=True,column_config={"Debit":st.column_config.NumberColumn(format="$%.2f"),"Credit":st.column_config.NumberColumn(format="$%.2f")})

    st.markdown("---")
    st.subheader("Account Balances")
    tb=df_led.groupby("Account").agg(Debits=("Debit","sum"),Credits=("Credit","sum"),Entries=("Ref","count")).reset_index()
    tb["Net"]=tb["Debits"]-tb["Credits"]
    st.dataframe(tb,use_container_width=True,hide_index=True,column_config={"Debits":st.column_config.NumberColumn(format="$%.0f"),"Credits":st.column_config.NumberColumn(format="$%.0f"),"Net":st.column_config.NumberColumn(format="$%.0f")})

    st.subheader("Audit Trail")
    st.dataframe(df_led.sort_values("Date",ascending=False).head(10)[["Date","Account","Debit","Credit","Ref","Narration"]],use_container_width=True,hide_index=True)


# ══════════════════════════════════════════════
# 6. LIQUIDITY
# ══════════════════════════════════════════════
elif page == "💧 Liquidity":
    st.title("💧 Liquidity Intelligence")
    st.caption("Net Available Liquidity = Banks + PSP − Pending − Liabilities")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Available Cash",fmt(available_cash))
    c2.metric("Pending Withdrawals",fmt(pending_wd),delta_color="inverse")
    c3.metric("Liabilities",fmt(bonus_liability+commission_liability),delta_color="inverse")
    c4.metric("Net Liquidity",fmt(net_liquidity),f"Buffer: {buffer_pct:.1f}%")

    # Liquidity Charts
    liq1,liq2,liq3 = st.columns(3)
    with liq1:
        bank_by_ccy = df_bank.groupby("CCY")["Balance"].sum().reset_index()
        fig_bc = px.pie(bank_by_ccy,values="Balance",names="CCY",hole=0.55,color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig_bc.update_layout(**PL,height=220); fig_bc.update_traces(textinfo="value+percent")
        st.caption("Bank by Currency"); st.plotly_chart(fig_bc,use_container_width=True)
    with liq2:
        psp_comp = df_psp.groupby("PSP").agg(Balance=("Balance","sum"),Pending=("Pending_Out","sum")).reset_index()
        fig_pc = go.Figure()
        fig_pc.add_trace(go.Bar(x=psp_comp["PSP"],y=psp_comp["Balance"],name="Balance",marker_color="#10b981"))
        fig_pc.add_trace(go.Bar(x=psp_comp["PSP"],y=psp_comp["Pending"],name="Pending Out",marker_color="#ef4444"))
        fig_pc.update_layout(**PL,height=220,barmode="group")
        st.caption("PSP Balance vs Pending"); st.plotly_chart(fig_pc,use_container_width=True)
    with liq3:
        fig_gauge = go.Figure(go.Indicator(mode="gauge+number",value=buffer_pct,title={"text":"Buffer %"},gauge={"axis":{"range":[0,30]},"bar":{"color":"#ef4444" if buffer_pct<15 else "#10b981"},"steps":[{"range":[0,15],"color":"rgba(239,68,68,.15)"},{"range":[15,30],"color":"rgba(16,185,129,.15)"}],"threshold":{"line":{"color":"#f59e0b","width":3},"thickness":0.8,"value":15}}))
        fig_gauge.update_layout(**PL,height=220)
        st.caption("Liquidity Buffer Gauge"); st.plotly_chart(fig_gauge,use_container_width=True)

    # Cash flow waterfall
    waterfall_items = ["Bank Balances","PSP Balances","Pending Out","Bonus","Commissions","Net Liquidity"]
    waterfall_vals = [total_bank,total_psp,-pending_wd,-bonus_liability,-commission_liability,net_liquidity]
    fig_wf = go.Figure(go.Waterfall(x=waterfall_items,y=waterfall_vals,measure=["relative","relative","relative","relative","relative","total"],connector={"line":{"color":"#334155"}},increasing={"marker":{"color":"#10b981"}},decreasing={"marker":{"color":"#ef4444"}},totals={"marker":{"color":"#6366f1"}}))
    fig_wf.update_layout(**PL,height=280,title="Liquidity Waterfall")
    st.plotly_chart(fig_wf,use_container_width=True)

    cl,cr=st.columns(2)
    with cl:
        st.subheader("🏦 Bank Balances")
        st.dataframe(df_bank,use_container_width=True,hide_index=True,column_config={"Balance":st.column_config.NumberColumn(format="$%d")})
    with cr:
        st.subheader("💳 PSP Balances")
        st.dataframe(df_psp,use_container_width=True,hide_index=True,column_config={"Balance":st.column_config.NumberColumn(format="$%d"),"Pending_In":st.column_config.NumberColumn(format="$%d"),"Pending_Out":st.column_config.NumberColumn(format="$%d")})

    st.subheader("Liabilities")
    li1,li2=st.columns(2)
    li1.metric("🎁 Bonus Exposure",fmt(bonus_liability))
    li2.metric("🤝 Commission Liabilities",fmt(commission_liability))

    st.subheader("Liquidity Calculation")
    calc=pd.DataFrame({"Item":["Bank Balances","PSP Balances","Pending Outflows","Bonus Liability","Commission Liability","Net Liquidity"],"Amount":[total_bank,total_psp,-pending_wd,-bonus_liability,-commission_liability,net_liquidity]})
    fig=px.bar(calc,y="Item",x="Amount",orientation="h",color=calc["Amount"].apply(lambda x:"Positive" if x>=0 else "Negative"),color_discrete_map={"Positive":"#10b981","Negative":"#ef4444"})
    fig.update_layout(**PL,height=280,showlegend=False)
    st.plotly_chart(fig,use_container_width=True)


# ══════════════════════════════════════════════
# 7. ALERTS & EXCEPTIONS
# ══════════════════════════════════════════════
elif "Alerts" in page:
    st.title("⚠️ Alerts & Exceptions")
    st.caption("Financial · Operational · Compliance — Case Management")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("🔴 Critical",len(df_alerts[(df_alerts["Severity"]=="Critical")&(df_alerts["Status"]!="Resolved")]))
    c2.metric("🟡 Open",len(df_alerts[df_alerts["Status"]=="Open"]))
    c3.metric("🔵 Investigating",len(df_alerts[df_alerts["Status"]=="Investigating"]))
    c4.metric("🟢 Resolved",len(df_alerts[df_alerts["Status"]=="Resolved"]))

    # Alert Charts
    ac1,ac2,ac3 = st.columns(3)
    with ac1:
        sev_cnt = df_alerts["Severity"].value_counts().reset_index(); sev_cnt.columns=["Severity","Count"]
        fig_sev = px.pie(sev_cnt,values="Count",names="Severity",hole=0.55,color_discrete_map={"Critical":"#ef4444","High":"#f59e0b","Medium":"#f59e0b","Low":"#3b82f6"})
        fig_sev.update_layout(**PL,height=200); fig_sev.update_traces(textinfo="value+percent")
        st.caption("By Severity"); st.plotly_chart(fig_sev,use_container_width=True)
    with ac2:
        cat_cnt = df_alerts["Category"].value_counts().reset_index(); cat_cnt.columns=["Category","Count"]
        fig_cat = px.bar(cat_cnt,x="Category",y="Count",color="Category",color_discrete_sequence=["#6366f1","#10b981","#f59e0b"])
        fig_cat.update_layout(**PL,height=200,showlegend=False)
        st.caption("By Category"); st.plotly_chart(fig_cat,use_container_width=True)
    with ac3:
        stat_cnt = df_alerts["Status"].value_counts().reset_index(); stat_cnt.columns=["Status","Count"]
        fig_stat = px.pie(stat_cnt,values="Count",names="Status",hole=0.55,color_discrete_map={"Open":"#f59e0b","Investigating":"#3b82f6","Resolved":"#10b981"})
        fig_stat.update_layout(**PL,height=200); fig_stat.update_traces(textinfo="value+percent")
        st.caption("By Status"); st.plotly_chart(fig_stat,use_container_width=True)

    tab_all,tab_fin,tab_ops,tab_comp=st.tabs(["All","Financial","Operational","Compliance"])

    sev_f=st.selectbox("Severity",["All","Critical","High","Medium","Low"])
    stat_f=st.selectbox("Status",["All","Open","Investigating","Resolved"],key="as")

    def show_alerts(data):
        d=data.copy()
        if sev_f!="All":d=d[d["Severity"]==sev_f]
        if stat_f!="All":d=d[d["Status"]==stat_f]
        if len(d)==0: st.success("No alerts"); return
        st.dataframe(d[["ID","Title","Category","Severity","Status","SLA","Created"]],use_container_width=True,hide_index=True)
        for _,al in d.iterrows():
            st.markdown(f'<div class="alert-card"><div class="alert-title">{sev_icon(al["Severity"])} {al["Title"]}</div><div class="alert-desc">{al["Description"]}</div><div style="margin-top:6px">{badge(al["Severity"],"red" if al["Severity"] in ["Critical","High"] else "yellow")} {badge(al["Category"],"blue")} {badge(al["Status"],"yellow" if al["Status"]!="Resolved" else "green")}</div></div>',unsafe_allow_html=True)

    with tab_all: show_alerts(df_alerts)
    with tab_fin: show_alerts(df_alerts[df_alerts["Category"]=="Financial"])
    with tab_ops: show_alerts(df_alerts[df_alerts["Category"]=="Operational"])
    with tab_comp: show_alerts(df_alerts[df_alerts["Category"]=="Compliance"])

    st.markdown("---")
    st.subheader("Case Management")
    active=df_alerts[df_alerts["Status"].isin(["Open","Investigating"])]
    if len(active)>0:
        sel=st.selectbox("Select alert",active["ID"].tolist())
        al=df_alerts[df_alerts["ID"]==sel].iloc[0]
        st.info(f"**{al['Title']}** — {al['Description']}")
        st.markdown(f"**Linked:** {al['Linked']} | **SLA:** {al['SLA']}")
        bc1,bc2,bc3,bc4=st.columns(4)
        bc1.button("📋 Open Case",use_container_width=True,key="oc")
        bc2.button("🔍 Investigate",use_container_width=True,key="inv2")
        bc3.button("✅ Resolve",use_container_width=True,key="res")
        bc4.button("❌ Dismiss",use_container_width=True,key="dis")


# ══════════════════════════════════════════════
# 8. REPORTS & ANALYTICS
# ══════════════════════════════════════════════
elif page == "📈 Reports & Analytics":
    st.title("📈 Reports & Analytics")
    st.caption("Daily · Weekly · Monthly reports + Profitability Analysis + KPIs")

    tab_daily,tab_weekly,tab_monthly,tab_profit,tab_kpis=st.tabs(["Daily Report","Weekly Report","Monthly Report","Profitability","KPIs"])

    with tab_daily:
        st.subheader("📅 Daily Finance Report — May 18, 2026")
        d1,d2,d3,d4=st.columns(4)
        d1.metric("Opening Balance",fmt(opening_balance))
        d2.metric("Cash In",fmt(cash_in))
        d3.metric("Cash Out",fmt(cash_out))
        d4.metric("Net Flow",fmt(net_flow))
        fig=go.Figure()
        fig.add_trace(go.Bar(x=["Cash In","Cash Out","Net Flow"],y=[cash_in,cash_out,net_flow],marker_color=["#10b981","#ef4444","#6366f1"]))
        fig.update_layout(**PL,height=260)
        st.plotly_chart(fig,use_container_width=True)

    with tab_weekly:
        st.subheader("📋 Weekly Control Report — May 12–18, 2026")
        w1,w2,w3=st.columns(3)
        w1.metric("Recon Rate",f"{recon_rate:.1f}%")
        w2.metric("Exceptions",len(df_rec[df_rec["Status"]=="Exception"]))
        w3.metric("PSP Performance",f"{len(df_psp)} active")
        fig_w=go.Figure()
        fig_w.add_trace(go.Scatter(x=df_cash["Date"],y=df_cash["Net"],mode="lines+markers",line=dict(color="#6366f1",width=2.5)))
        fig_w.update_layout(**PL,height=240)
        st.plotly_chart(fig_w,use_container_width=True)

    with tab_monthly:
        st.subheader("📊 Monthly Financial Report — May 2026")
        total_rev=df_profit["Revenue"].sum();total_cost=df_profit["Costs"].sum();total_prof=df_profit["Profit"].sum()
        m1,m2,m3=st.columns(3)
        m1.metric("Revenue",fmt(total_rev))
        m2.metric("Costs",fmt(total_cost))
        m3.metric("Profit",fmt(total_prof),f"Margin: {total_prof/total_rev*100:.1f}%")

        # Monthly charts
        mr1,mr2 = st.columns(2)
        with mr1:
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Bar(x=df_kpi["Month"],y=df_kpi["Net_Flow"],name="Net Flow",marker_color="#10b981"))
            fig_rev.add_trace(go.Bar(x=df_kpi["Month"],y=df_kpi["Op_Costs"],name="Op Costs",marker_color="#ef4444"))
            fig_rev.update_layout(**PL,height=260,barmode="group",title="Revenue vs Costs Trend")
            st.plotly_chart(fig_rev,use_container_width=True)
        with mr2:
            margin_data = df_kpi.copy()
            margin_data["Margin"] = ((margin_data["Net_Flow"] - margin_data["Op_Costs"]) / margin_data["Net_Flow"] * 100)
            fig_margin = px.line(margin_data,x="Month",y="Margin",markers=True,color_discrete_sequence=["#8b5cf6"])
            fig_margin.update_layout(**PL,height=260,title="Profit Margin Trend (%)")
            st.plotly_chart(fig_margin,use_container_width=True)

        # Profitability treemap
        fig_tree = px.treemap(df_profit,path=["Client"],values="Profit",color="Profit",color_continuous_scale=["#ef4444","#f59e0b","#10b981"])
        fig_tree.update_layout(**PL,height=280,title="Client Profitability Map")
        st.plotly_chart(fig_tree,use_container_width=True)

    with tab_profit:
        st.subheader("💰 Profitability Analysis")
        pt1,pt2,pt3=st.tabs(["Client Profitability","IB Profitability","Campaign Profitability"])
        with pt1:
            cl,cr=st.columns(2)
            with cl:
                fig=go.Figure()
                fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Revenue"],name="Revenue",marker_color="#10b981"))
                fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Costs"],name="Costs",marker_color="#ef4444"))
                fig.add_trace(go.Bar(x=df_profit["Client"],y=df_profit["Profit"],name="Profit",marker_color="#6366f1"))
                fig.update_layout(**PL,height=300,barmode="group")
                st.plotly_chart(fig,use_container_width=True)
            with cr:
                df_p=df_profit.copy();df_p["Margin"]=(df_p["Profit"]/df_p["Revenue"]*100).round(1).astype(str)+"%"
                st.dataframe(df_p.sort_values("Profit",ascending=False),use_container_width=True,hide_index=True,column_config={"Revenue":st.column_config.NumberColumn(format="$%d"),"Costs":st.column_config.NumberColumn(format="$%d"),"Profit":st.column_config.NumberColumn(format="$%d")})
        with pt2:
            cl2,cr2=st.columns(2)
            with cl2:
                fig_ib=go.Figure()
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Net_Revenue"],name="Net Revenue",marker_color="#10b981"))
                fig_ib.add_trace(go.Bar(x=df_ib["Partner"],y=df_ib["Commission"],name="Commission",marker_color="#8b5cf6"))
                fig_ib.update_layout(**PL,height=300,barmode="group")
                st.plotly_chart(fig_ib,use_container_width=True)
            with cr2:
                st.dataframe(df_ib.sort_values("Net_Revenue",ascending=False),use_container_width=True,hide_index=True,column_config={"Volume":st.column_config.NumberColumn(format="$%d"),"Commission":st.column_config.NumberColumn(format="$%d"),"Net_Revenue":st.column_config.NumberColumn(format="$%d")})
        with pt3:
            campaigns=pd.DataFrame([{"Campaign":"Welcome Bonus","Spend":45000,"Revenue":180000,"ROI":300},{"Campaign":"Loyalty Program","Spend":22000,"Revenue":95000,"ROI":332},{"Campaign":"Referral Bonus","Spend":15000,"Revenue":72000,"ROI":380},{"Campaign":"VIP Cashback","Spend":35000,"Revenue":110000,"ROI":214}])
            cl3,cr3=st.columns(2)
            with cl3:
                fig_c=go.Figure()
                fig_c.add_trace(go.Bar(x=campaigns["Campaign"],y=campaigns["Spend"],name="Spend",marker_color="#ef4444"))
                fig_c.add_trace(go.Bar(x=campaigns["Campaign"],y=campaigns["Revenue"],name="Revenue",marker_color="#10b981"))
                fig_c.update_layout(**PL,height=300,barmode="group")
                st.plotly_chart(fig_c,use_container_width=True)
            with cr3:
                st.dataframe(campaigns.sort_values("ROI",ascending=False),use_container_width=True,hide_index=True,column_config={"Spend":st.column_config.NumberColumn(format="$%d"),"Revenue":st.column_config.NumberColumn(format="$%d"),"ROI":st.column_config.ProgressColumn(min_value=0,max_value=400,format="%d%%")})

    with tab_kpis:
        st.subheader("📈 Financial KPIs")
        cl,cr=st.columns(2)
        with cl:
            fig_nf=px.line(df_kpi,x="Month",y="Net_Flow",markers=True,color_discrete_sequence=["#6366f1"])
            fig_nf.update_layout(**PL,height=260,title="Net Flow Trend")
            st.plotly_chart(fig_nf,use_container_width=True)
        with cr:
            fig_rr=px.line(df_kpi,x="Month",y="Recon_Rate",markers=True,color_discrete_sequence=["#10b981"])
            fig_rr.add_hline(y=95,line_dash="dash",line_color="#f59e0b",annotation_text="95% Target")
            fig_rr.update_layout(**PL,height=260,title="Reconciliation Rate")
            st.plotly_chart(fig_rr,use_container_width=True)
        cl2,cr2=st.columns(2)
        with cl2:
            fig_oc=px.bar(df_kpi,x="Month",y="Op_Costs",color_discrete_sequence=["#ef4444"])
            fig_oc.update_layout(**PL,height=260,showlegend=False,title="Operating Costs")
            st.plotly_chart(fig_oc,use_container_width=True)
        with cr2:
            fig_ex=px.line(df_kpi,x="Month",y="Exceptions",markers=True,color_discrete_sequence=["#f59e0b"])
            fig_ex.update_layout(**PL,height=260,title="Exception Trends")
            st.plotly_chart(fig_ex,use_container_width=True)


# ══════════════════════════════════════════════
# 9. RISK EARLY WARNING SYSTEM
# ══════════════════════════════════════════════
elif page == "🛡️ Risk Monitor":
    st.title("🛡️ Risk Early Warning System")
    st.caption("Proactive risk detection — Liquidity · Settlement · Operational · Concentration")

    # Risk indicators
    liq_risk = "Red" if buffer_pct < 15 else "Yellow" if buffer_pct < 20 else "Green"
    settle_risk = "Yellow" if len(df_alerts[df_alerts["Title"].str.contains("Delay|Settlement",case=False)&(df_alerts["Status"]!="Resolved")]) > 0 else "Green"
    ops_risk = "Red" if unmatched > 3 else "Yellow" if unmatched > 1 else "Green"
    psp_volumes = df_txn.groupby("Counterparty").size()
    max_psp_pct = (psp_volumes.max() / psp_volumes.sum() * 100) if len(psp_volumes) > 0 else 0
    conc_risk = "Red" if max_psp_pct > 50 else "Yellow" if max_psp_pct > 35 else "Green"

    st.subheader("Risk Dashboard")
    risks = [
        ("Liquidity Risk", liq_risk, f"Buffer: {buffer_pct:.1f}%", "Cash position vs obligations"),
        ("Settlement Risk", settle_risk, f"{len(df_alerts[df_alerts['Title'].str.contains('Delay',case=False)])} delays", "PSP settlement timing"),
        ("Operational Risk", ops_risk, f"{unmatched} unmatched", "Reconciliation exceptions"),
        ("Concentration Risk", conc_risk, f"Top PSP: {max_psp_pct:.0f}%", "Dependency on single PSP/IB"),
    ]

    r1,r2,r3,r4=st.columns(4)
    for col,(name,status,detail,desc) in zip([r1,r2,r3,r4],risks):
        color=risk_color(status)
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:32px;margin-bottom:4px">{"🟢" if status=="Green" else "🟡" if status=="Yellow" else "🔴"}</div><h4 style="font-size:13px">{name}</h4><div style="font-size:18px;font-weight:700;color:{color}">{status}</div><div class="sm">{detail}</div></div>',unsafe_allow_html=True)

    st.markdown("---")

    # Detailed risk indicators
    st.subheader("Liquidity Risk Indicators")
    for label,val,threshold,status in [
        ("Net Liquidity",fmt(net_liquidity),"Min: $500K","🟢" if net_liquidity>500000 else "🔴"),
        ("Liquidity Buffer",f"{buffer_pct:.1f}%","Min: 15%","🟢" if buffer_pct>=15 else "🔴"),
        ("Withdrawal Pressure",fmt(pending_wd),f"{pending_wd/available_cash*100:.1f}% of cash","🟢" if pending_wd/available_cash<0.1 else "🟡"),
    ]:
        st.markdown(f'<div class="risk-row"><span class="risk-label">{status} {label}</span><span class="risk-val">{val} <span class="sm">({threshold})</span></span></div>',unsafe_allow_html=True)

    st.subheader("Operational Risk Indicators")
    for label,val,threshold,status in [
        ("Recon Exception Rate",f"{(1-recon_rate/100)*100:.1f}%","Max: 5%","🟢" if (1-recon_rate/100)*100<=5 else "🔴"),
        ("PSP Delays",f"{len(df_alerts[df_alerts['Title'].str.contains('Delay',case=False)])} active","Max: 0","🟢" if len(df_alerts[df_alerts['Title'].str.contains('Delay',case=False)])==0 else "🟡"),
        ("Failed Transactions",f"{len(df_txn[df_txn['Status']=='Failed'])}","Max: 0","🟢" if len(df_txn[df_txn['Status']=='Failed'])==0 else "🔴"),
    ]:
        st.markdown(f'<div class="risk-row"><span class="risk-label">{status} {label}</span><span class="risk-val">{val} <span class="sm">({threshold})</span></span></div>',unsafe_allow_html=True)

    # Risk Radar Chart
    st.subheader("Risk Radar")
    risk_scores = {"Liquidity": 85 if liq_risk=="Green" else 50 if liq_risk=="Yellow" else 20,
                   "Settlement": 80 if settle_risk=="Green" else 50 if settle_risk=="Yellow" else 20,
                   "Operational": 80 if ops_risk=="Green" else 50 if ops_risk=="Yellow" else 20,
                   "Concentration": 70 if conc_risk=="Green" else 45 if conc_risk=="Yellow" else 20,
                   "Compliance": 60}
    categories = list(risk_scores.keys())
    values = list(risk_scores.values())
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=values+[values[0]],theta=categories+[categories[0]],fill="toself",fillcolor="rgba(99,102,241,.15)",line=dict(color="#6366f1",width=2),name="Current"))
    fig_radar.add_trace(go.Scatterpolar(r=[80]*6,theta=categories+[categories[0]],line=dict(color="#10b981",width=1,dash="dash"),name="Target"))
    fig_radar.update_layout(**PL,height=320,polar=dict(bgcolor="rgba(0,0,0,0)",radialaxis=dict(visible=True,range=[0,100],gridcolor="#1e2d4a"),angularaxis=dict(gridcolor="#1e2d4a")))
    st.plotly_chart(fig_radar,use_container_width=True)

    st.subheader("Concentration Risk")
    cr1,cr2 = st.columns(2)
    with cr1:
        st.markdown("**PSP Volume Distribution**")
        psp_vol=df_txn.groupby("Counterparty")["Amount"].sum().reset_index().sort_values("Amount",ascending=False)
        fig_conc=px.pie(psp_vol,values="Amount",names="Counterparty",hole=0.5,color_discrete_sequence=["#6366f1","#10b981","#f59e0b","#ef4444","#8b5cf6"])
        fig_conc.update_layout(**PL,height=260)
        st.plotly_chart(fig_conc,use_container_width=True)
    with cr2:
        st.markdown("**Client Volume Distribution**")
        client_vol=df_txn.groupby("Client")["Amount"].sum().nlargest(6).reset_index()
        fig_cv=px.pie(client_vol,values="Amount",names="Client",hole=0.5,color_discrete_sequence=["#10b981","#6366f1","#f59e0b","#8b5cf6","#06b6d4","#ef4444"])
        fig_cv.update_layout(**PL,height=260)
        st.plotly_chart(fig_cv,use_container_width=True)

    st.subheader("Scenario Analysis")
    st.markdown("**What if withdrawals increase by:**")
    increase=st.slider("Withdrawal increase %",0,200,50,10)
    new_wd=pending_wd*(1+increase/100)
    new_liq=available_cash-new_wd-bonus_liability-commission_liability
    new_buf=new_liq/available_cash*100 if available_cash>0 else 0
    sc1,sc2,sc3=st.columns(3)
    sc1.metric("New Pending WD",fmt(new_wd),f"+{increase}%",delta_color="inverse")
    sc2.metric("New Net Liquidity",fmt(new_liq))
    sc3.metric("New Buffer",f"{new_buf:.1f}%","🔴 BREACH" if new_buf<15 else "✅ OK",delta_color="inverse" if new_buf<15 else "normal")


# ══════════════════════════════════════════════
# 10. INTEGRATIONS
# ══════════════════════════════════════════════
elif page == "🔌 Integrations":
    st.title("🔌 Integrations")
    st.caption("Connected systems and data feeds")

    integrations = [
        {"System":"CRM","Protocol":"REST API","Status":"Connected","Last_Sync":"2 min ago","Records":"8,420","Health":"Online"},
        {"System":"Stripe (PSP)","Protocol":"REST + Webhook","Status":"Connected","Last_Sync":"30 sec ago","Records":"12,105","Health":"Online"},
        {"System":"Adyen (PSP)","Protocol":"REST API","Status":"Delayed","Last_Sync":"4h ago","Records":"5,890","Health":"Warning"},
        {"System":"Worldpay (PSP)","Protocol":"REST API","Status":"Connected","Last_Sync":"5 min ago","Records":"3,200","Health":"Online"},
        {"System":"JPMorgan (Bank)","Protocol":"SWIFT / SFTP","Status":"Connected","Last_Sync":"15 min ago","Records":"2,100","Health":"Online"},
        {"System":"HSBC (Bank)","Protocol":"SWIFT / SFTP","Status":"Connected","Last_Sync":"15 min ago","Records":"980","Health":"Online"},
        {"System":"Deutsche Bank","Protocol":"SWIFT / SFTP","Status":"Connected","Last_Sync":"15 min ago","Records":"1,450","Health":"Online"},
        {"System":"Barclays (Bank)","Protocol":"SWIFT / SFTP","Status":"Connected","Last_Sync":"15 min ago","Records":"890","Health":"Online"},
        {"System":"Trading Platform","Protocol":"WebSocket","Status":"Connected","Last_Sync":"Real-time","Records":"45,890","Health":"Online"},
        {"System":"Bonus Engine","Protocol":"REST API","Status":"Connected","Last_Sync":"18 min ago","Records":"1,240","Health":"Online"},
        {"System":"Commission Engine","Protocol":"REST API","Status":"Connected","Last_Sync":"10 min ago","Records":"892","Health":"Online"},
    ]
    df_int=pd.DataFrame(integrations)
    st.dataframe(df_int,use_container_width=True,hide_index=True)

    for intg in integrations:
        dot="dot-g" if intg["Health"]=="Online" else "dot-y" if intg["Health"]=="Warning" else "dot-r"
        st.markdown(f'<div class="risk-row"><span class="risk-label"><span class="status-dot {dot}"></span>{intg["System"]}</span><span class="sm">{intg["Protocol"]} · Last: {intg["Last_Sync"]} · {intg["Records"]} records</span></div>',unsafe_allow_html=True)


# ══════════════════════════════════════════════
# 11. SETTINGS
# ══════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.caption("System configuration and administration")

    tab_gen,tab_users,tab_alerts_cfg,tab_recon_cfg=st.tabs(["General","Users & Roles","Alert Rules","Reconciliation"])

    with tab_gen:
        st.subheader("General Settings")
        st.text_input("Company Name",value="FinanceOps Corp")
        st.selectbox("Base Currency",["USD","EUR","GBP"],index=0)
        st.selectbox("Timezone",["UTC","EST","CET","GMT"],index=0)
        st.number_input("Liquidity Buffer Threshold %",value=15.0,step=0.5)
        st.button("💾 Save Settings",use_container_width=True)

    with tab_users:
        st.subheader("Users & Roles")
        users=pd.DataFrame([
            {"User":"Ahmed K.","Role":"CFO","Access":"Full","Status":"Active"},
            {"User":"Sarah M.","Role":"Finance Manager","Access":"Finance + Reports","Status":"Active"},
            {"User":"Omar R.","Role":"Operations Manager","Access":"Transactions + Alerts","Status":"Active"},
            {"User":"Lina T.","Role":"Recon Analyst","Access":"Reconciliation + Ledger","Status":"Active"},
            {"User":"David P.","Role":"Compliance","Access":"Alerts + Reports","Status":"Active"},
        ])
        st.dataframe(users,use_container_width=True,hide_index=True)

    with tab_alerts_cfg:
        st.subheader("Alert Rules")
        st.number_input("Withdrawal Spike Threshold %",value=200,step=10)
        st.number_input("Cash Imbalance Tolerance $",value=5000,step=500)
        st.number_input("Recon Exception Rate Threshold %",value=5.0,step=0.5)
        st.number_input("PSP Delay SLA (hours)",value=4,step=1)
        st.button("💾 Save Alert Rules",use_container_width=True)

    with tab_recon_cfg:
        st.subheader("Reconciliation Rules")
        st.number_input("Amount Match Tolerance %",value=1.0,step=0.1)
        st.number_input("Time Window (hours)",value=24,step=1)
        st.checkbox("Auto-create cases for exceptions",value=True)
        st.checkbox("Auto-match by ID + Amount",value=True)
        st.checkbox("Enable FX tolerance matching",value=True)
        st.button("💾 Save Recon Rules",use_container_width=True)


# ══════════════════════════════════════════════
# 12. FILE UPLOAD
# ══════════════════════════════════════════════
elif page == "📂 File Upload":
    st.title("📂 File Upload")
    st.caption("Upload Word (.docx), Excel (.xlsx), CSV, PDF, or Text files")

    uploaded=st.file_uploader("Drop files here",type=["docx","xlsx","xls","csv","pdf","txt"],accept_multiple_files=True)

    if uploaded:
        for uf in uploaded:
            ext=uf.name.rsplit(".",1)[-1].lower()
            sz=f"{uf.size/1024:.1f} KB" if uf.size<1048576 else f"{uf.size/1048576:.1f} MB"
            st.markdown("---")
            st.subheader(f"📄 {uf.name}")
            st.caption(f".{ext} · {sz}")
            try:
                if ext=="csv":
                    df=pd.read_csv(uf)
                    st.success(f"{len(df)} rows × {len(df.columns)} columns")
                    t1,t2=st.tabs(["Data","Statistics"])
                    with t1: st.dataframe(df,use_container_width=True,hide_index=True)
                    with t2: st.dataframe(df.describe(),use_container_width=True)
                    st.download_button("📥 Download CSV",df.to_csv(index=False).encode(),"data.csv","text/csv",key=f"dc_{uf.name}")
                elif ext in ["xlsx","xls"]:
                    xls=pd.ExcelFile(uf)
                    sheet=st.selectbox("Sheet",xls.sheet_names,key=f"sh_{uf.name}")
                    df=pd.read_excel(uf,sheet_name=sheet)
                    st.success(f"{len(df)} rows × {len(df.columns)} columns")
                    st.dataframe(df,use_container_width=True,hide_index=True)
                    st.download_button("📥 Download CSV",df.to_csv(index=False).encode(),"data.csv","text/csv",key=f"dx_{uf.name}")
                elif ext=="docx":
                    from docx import Document
                    doc=Document(uf)
                    paras=[p.text for p in doc.paragraphs if p.text.strip()]
                    tables=doc.tables
                    st.success(f"{len(paras)} paragraphs, {len(tables)} tables")
                    t1,t2=st.tabs(["Text",f"Tables ({len(tables)})"])
                    with t1:
                        for p in doc.paragraphs:
                            if p.text.strip():
                                if p.style and "Heading" in (p.style.name or ""):
                                    st.markdown(f"### {p.text}")
                                else:
                                    st.markdown(p.text)
                    with t2:
                        for ti,table in enumerate(tables):
                            rows=[[c.text.strip() for c in r.cells] for r in table.rows]
                            if len(rows)>1:
                                df_t=pd.DataFrame(rows[1:],columns=rows[0])
                                st.markdown(f"**Table {ti+1}**")
                                st.dataframe(df_t,use_container_width=True,hide_index=True)
                                st.download_button(f"📥 Table {ti+1}",df_t.to_csv(index=False).encode(),f"table_{ti+1}.csv","text/csv",key=f"dt_{uf.name}_{ti}")
                elif ext=="pdf":
                    from PyPDF2 import PdfReader
                    reader=PdfReader(uf)
                    st.success(f"{len(reader.pages)} pages")
                    for i,pg in enumerate(reader.pages):
                        txt=pg.extract_text()
                        if txt:
                            st.markdown(f"**Page {i+1}**")
                            st.text(txt)
                elif ext=="txt":
                    st.text(uf.read().decode("utf-8",errors="replace"))
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown('<div class="card" style="text-align:center;padding:40px"><div style="font-size:48px;margin-bottom:12px">📂</div><h4>Upload files to get started</h4><div class="sm">Supported: .docx · .xlsx · .csv · .pdf · .txt</div></div>',unsafe_allow_html=True)
