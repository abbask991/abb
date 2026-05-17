import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import json

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DETECTION_PARAMS = {
    "same_symbol":        {"label": "Same Symbol",           "points": 10, "icon": "1f4ca", "desc": "Both accounts trade the same instrument"},
    "opposite_direction": {"label": "Opposite Direction",    "points": 20, "icon": "1f500", "desc": "One buys while the other sells (hedging)"},
    "open_time_match":    {"label": "Open Time Match",       "points": 15, "icon": "23f1",  "desc": "Positions opened at near-identical time"},
    "close_time_match":   {"label": "Close Time Match",      "points": 15, "icon": "23f2",  "desc": "Positions closed at near-identical time"},
    "same_lot_size":      {"label": "Same Lot Size",         "points": 10, "icon": "1f4e6", "desc": "Matching or highly similar trade volume"},
    "shared_cid":         {"label": "Shared Device (CID)",   "points": 20, "icon": "1f4f1", "desc": "Accounts accessed from the same device"},
    "shared_ip":          {"label": "Shared IP Address",     "points": 20, "icon": "1f310", "desc": "Accounts connected from the same IP"},
    "bonus_similarity":   {"label": "Bonus Similarity",      "points": 10, "icon": "1f4b0", "desc": "Bonus values within \u00b130% range"},
    "similar_exposure":   {"label": "Similar Exposure",      "points": 15, "icon": "1f4ca", "desc": "Total market exposure is highly similar"},
}

MAX_SCORE = sum(p["points"] for p in DETECTION_PARAMS.values())

RISK_LEVELS = [
    (0,  20, "Low",      "#22c55e", "Monitor activity. No immediate action required."),
    (21, 40, "Medium",   "#eab308", "Review account behavior. Monitor withdrawals. Flag for observation."),
    (41, 60, "High",     "#f97316", "Temporary withdrawal review. Manual investigation. Compliance review required."),
    (61, MAX_SCORE, "Critical", "#ef4444",
     "Immediate fraud investigation. Bonus cancellation consideration. Potential account suspension. Escalate to Risk & Compliance Department."),
]

SAMPLE_TRADES_A = pd.DataFrame({
    "Trade ID": ["T1001", "T1002", "T1003", "T1004", "T1005"],
    "Symbol": ["EURUSD", "XAUUSD", "GBPJPY", "EURUSD", "USDJPY"],
    "Direction": ["BUY", "BUY", "SELL", "SELL", "BUY"],
    "Lot Size": [1.00, 0.50, 2.00, 1.00, 0.30],
    "Open Time": pd.to_datetime([
        "2026-05-15 09:30:00", "2026-05-15 10:00:00",
        "2026-05-15 11:15:00", "2026-05-15 14:00:00",
        "2026-05-15 15:30:00",
    ]),
    "Close Time": pd.to_datetime([
        "2026-05-15 10:00:00", "2026-05-15 11:00:00",
        "2026-05-15 12:00:00", "2026-05-15 15:00:00",
        "2026-05-15 16:30:00",
    ]),
    "Open Price": [1.0850, 2350.00, 198.50, 1.0870, 156.30],
    "Close Price": [1.0870, 2360.00, 197.80, 1.0840, 156.50],
})

SAMPLE_TRADES_B = pd.DataFrame({
    "Trade ID": ["T2001", "T2002", "T2003", "T2004", "T2005"],
    "Symbol": ["EURUSD", "XAUUSD", "USDJPY", "EURUSD", "GBPJPY"],
    "Direction": ["SELL", "SELL", "BUY", "BUY", "SELL"],
    "Lot Size": [1.00, 0.49, 1.50, 1.00, 0.30],
    "Open Time": pd.to_datetime([
        "2026-05-15 09:30:02", "2026-05-15 10:00:05",
        "2026-05-15 11:20:00", "2026-05-15 14:00:01",
        "2026-05-15 15:35:00",
    ]),
    "Close Time": pd.to_datetime([
        "2026-05-15 10:00:03", "2026-05-15 11:00:10",
        "2026-05-15 12:05:00", "2026-05-15 15:00:02",
        "2026-05-15 16:35:00",
    ]),
    "Open Price": [1.0850, 2350.50, 156.30, 1.0870, 198.60],
    "Close Price": [1.0830, 2340.00, 156.50, 1.0900, 198.00],
})

CONTRACT_SIZES = {
    "EURUSD": 100_000, "GBPUSD": 100_000, "USDJPY": 100_000,
    "GBPJPY": 100_000, "XAUUSD": 100, "XAGUSD": 5_000,
}
DEFAULT_CONTRACT_SIZE = 100_000

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_risk_level(score: int):
    for low, high, label, color, action in RISK_LEVELS:
        if low <= score <= high:
            return label, color, action
    return "Unknown", "#6b7280", ""


def time_diff_seconds(t1, t2):
    if pd.isna(t1) or pd.isna(t2):
        return float("inf")
    return abs((t1 - t2).total_seconds())


def lot_similarity(lot_a, lot_b, tolerance=0.05):
    if lot_a == 0 and lot_b == 0:
        return True
    max_lot = max(abs(lot_a), abs(lot_b))
    if max_lot == 0:
        return True
    return abs(lot_a - lot_b) / max_lot <= tolerance


def bonus_within_range(bonus_a, bonus_b, pct=0.30):
    if bonus_a == 0 and bonus_b == 0:
        return True
    max_bonus = max(abs(bonus_a), abs(bonus_b))
    if max_bonus == 0:
        return True
    return abs(bonus_a - bonus_b) / max_bonus <= pct


def compute_exposure(trades_df):
    total = 0.0
    for _, row in trades_df.iterrows():
        cs = CONTRACT_SIZES.get(row["Symbol"], DEFAULT_CONTRACT_SIZE)
        total += row["Lot Size"] * cs * row.get("Open Price", 1.0)
    return total


def exposure_similar(exp_a, exp_b, tolerance=0.20):
    max_exp = max(abs(exp_a), abs(exp_b))
    if max_exp == 0:
        return True
    return abs(exp_a - exp_b) / max_exp <= tolerance


def match_trades(trades_a, trades_b, time_threshold_sec):
    """Find best-match trade pairs between two accounts by symbol and open time."""
    pairs = []
    used_b = set()
    for _, ta in trades_a.iterrows():
        best_match = None
        best_dt = float("inf")
        for idx_b, tb in trades_b.iterrows():
            if idx_b in used_b:
                continue
            if ta["Symbol"] != tb["Symbol"]:
                continue
            dt = time_diff_seconds(ta["Open Time"], tb["Open Time"])
            if dt < best_dt:
                best_dt = dt
                best_match = idx_b
        if best_match is not None:
            used_b.add(best_match)
            pairs.append((ta, trades_b.loc[best_match], best_dt))
    return pairs


def analyze_pair(trades_a, trades_b, account_info_a, account_info_b,
                 time_threshold_sec, lot_tolerance):
    """Run all detection checks and return scored results."""
    results = {k: {"matched": False, "details": []} for k in DETECTION_PARAMS}

    pairs = match_trades(trades_a, trades_b, time_threshold_sec)

    for ta, tb, dt in pairs:
        pair_label = f"{ta['Trade ID']} \u2194 {tb['Trade ID']}"

        if ta["Symbol"] == tb["Symbol"]:
            results["same_symbol"]["matched"] = True
            results["same_symbol"]["details"].append(
                f"{pair_label}: both trade **{ta['Symbol']}**")

            if ta["Direction"] != tb["Direction"]:
                results["opposite_direction"]["matched"] = True
                results["opposite_direction"]["details"].append(
                    f"{pair_label}: **{ta['Direction']}** vs **{tb['Direction']}** on {ta['Symbol']}")

            if dt <= time_threshold_sec:
                results["open_time_match"]["matched"] = True
                results["open_time_match"]["details"].append(
                    f"{pair_label}: open times differ by **{dt:.1f}s**")

            close_dt = time_diff_seconds(ta["Close Time"], tb["Close Time"])
            if close_dt <= time_threshold_sec:
                results["close_time_match"]["matched"] = True
                results["close_time_match"]["details"].append(
                    f"{pair_label}: close times differ by **{close_dt:.1f}s**")

            if lot_similarity(ta["Lot Size"], tb["Lot Size"], lot_tolerance):
                results["same_lot_size"]["matched"] = True
                results["same_lot_size"]["details"].append(
                    f"{pair_label}: **{ta['Lot Size']}** vs **{tb['Lot Size']}** lots")

    cid_a = set(account_info_a.get("cids", []))
    cid_b = set(account_info_b.get("cids", []))
    shared_cids = cid_a & cid_b
    if shared_cids:
        results["shared_cid"]["matched"] = True
        results["shared_cid"]["details"].append(
            f"Shared devices: **{', '.join(shared_cids)}**")

    ip_a = set(account_info_a.get("ips", []))
    ip_b = set(account_info_b.get("ips", []))
    shared_ips = ip_a & ip_b
    if shared_ips:
        results["shared_ip"]["matched"] = True
        results["shared_ip"]["details"].append(
            f"Shared IPs: **{', '.join(shared_ips)}**")

    bonus_a = account_info_a.get("bonus", 0)
    bonus_b = account_info_b.get("bonus", 0)
    if bonus_a > 0 and bonus_b > 0 and bonus_within_range(bonus_a, bonus_b):
        results["bonus_similarity"]["matched"] = True
        results["bonus_similarity"]["details"].append(
            f"**${bonus_a:.2f}** vs **${bonus_b:.2f}** (within \u00b130%)")

    exp_a = compute_exposure(trades_a)
    exp_b = compute_exposure(trades_b)
    if exposure_similar(exp_a, exp_b):
        results["similar_exposure"]["matched"] = True
        results["similar_exposure"]["details"].append(
            f"Exposure A: **${exp_a:,.2f}** vs B: **${exp_b:,.2f}**")

    total_score = sum(
        DETECTION_PARAMS[k]["points"] for k, v in results.items() if v["matched"]
    )
    return results, total_score, exp_a, exp_b


def build_position_detail_rows(trades_a, trades_b, account_info_a, account_info_b,
                                time_threshold_sec, lot_tolerance):
    """Build per-position detection table rows with all flag columns."""
    pairs = match_trades(trades_a, trades_b, time_threshold_sec)
    rows = []
    for ta, tb, dt in pairs:
        if ta["Symbol"] != tb["Symbol"]:
            continue
        close_dt = time_diff_seconds(ta["Close Time"], tb["Close Time"])
        is_opposite = ta["Direction"] != tb["Direction"]
        is_open_match = dt <= time_threshold_sec
        is_close_match = close_dt <= time_threshold_sec
        is_lot_match = lot_similarity(ta["Lot Size"], tb["Lot Size"], lot_tolerance)

        flags_hit = sum([True, is_opposite, is_open_match, is_close_match, is_lot_match])
        pair_score = (
            DETECTION_PARAMS["same_symbol"]["points"]
            + (DETECTION_PARAMS["opposite_direction"]["points"] if is_opposite else 0)
            + (DETECTION_PARAMS["open_time_match"]["points"] if is_open_match else 0)
            + (DETECTION_PARAMS["close_time_match"]["points"] if is_close_match else 0)
            + (DETECTION_PARAMS["same_lot_size"]["points"] if is_lot_match else 0)
        )

        rows.append({
            "Trade A": ta["Trade ID"],
            "Trade B": tb["Trade ID"],
            "Symbol": ta["Symbol"],
            "Dir A": ta["Direction"],
            "Dir B": tb["Direction"],
            "Lot A": ta["Lot Size"],
            "Lot B": tb["Lot Size"],
            "Open A": ta["Open Time"],
            "Open B": tb["Open Time"],
            "Open \u0394 (s)": round(dt, 1),
            "Close A": ta["Close Time"],
            "Close B": tb["Close Time"],
            "Close \u0394 (s)": round(close_dt, 1),
            "Same Symbol": "\u2705",
            "Opposite Dir": "\u2705" if is_opposite else "\u274c",
            "Open Time Match": "\u2705" if is_open_match else "\u274c",
            "Close Time Match": "\u2705" if is_close_match else "\u274c",
            "Same Lot": "\u2705" if is_lot_match else "\u274c",
            "Pair Score": pair_score,
        })
    return rows


def build_account_detail_rows(account_info_a, account_info_b, exp_a, exp_b):
    """Build the account-level detection summary."""
    cid_a = set(account_info_a.get("cids", []))
    cid_b = set(account_info_b.get("cids", []))
    ip_a = set(account_info_a.get("ips", []))
    ip_b = set(account_info_b.get("ips", []))
    shared_cids = cid_a & cid_b
    shared_ips = ip_a & ip_b
    bonus_a = account_info_a.get("bonus", 0)
    bonus_b = account_info_b.get("bonus", 0)
    bonus_match = bonus_a > 0 and bonus_b > 0 and bonus_within_range(bonus_a, bonus_b)
    exp_match = exposure_similar(exp_a, exp_b)

    rows = [
        {
            "Check": "Shared IP Address",
            "Account A": ", ".join(ip_a),
            "Account B": ", ".join(ip_b),
            "Overlap / Value": ", ".join(shared_ips) if shared_ips else "None",
            "Flagged": "\u2705" if shared_ips else "\u274c",
            "Points": DETECTION_PARAMS["shared_ip"]["points"] if shared_ips else 0,
        },
        {
            "Check": "Shared Device (CID)",
            "Account A": ", ".join(cid_a),
            "Account B": ", ".join(cid_b),
            "Overlap / Value": ", ".join(shared_cids) if shared_cids else "None",
            "Flagged": "\u2705" if shared_cids else "\u274c",
            "Points": DETECTION_PARAMS["shared_cid"]["points"] if shared_cids else 0,
        },
        {
            "Check": "Bonus Similarity (\u00b130%)",
            "Account A": f"${bonus_a:,.2f}",
            "Account B": f"${bonus_b:,.2f}",
            "Overlap / Value": f"\u0394 {abs(bonus_a - bonus_b) / max(bonus_a, bonus_b, 1) * 100:.1f}%" if max(bonus_a, bonus_b) > 0 else "N/A",
            "Flagged": "\u2705" if bonus_match else "\u274c",
            "Points": DETECTION_PARAMS["bonus_similarity"]["points"] if bonus_match else 0,
        },
        {
            "Check": "Similar Exposure (\u00b120%)",
            "Account A": f"${exp_a:,.0f}",
            "Account B": f"${exp_b:,.0f}",
            "Overlap / Value": f"\u0394 {abs(exp_a - exp_b) / max(exp_a, exp_b, 1) * 100:.1f}%" if max(exp_a, exp_b) > 0 else "N/A",
            "Flagged": "\u2705" if exp_match else "\u274c",
            "Points": DETECTION_PARAMS["similar_exposure"]["points"] if exp_match else 0,
        },
    ]
    return rows


def parse_csv_trades(uploaded_file):
    df = pd.read_csv(uploaded_file)
    required = {"Trade ID", "Symbol", "Direction", "Lot Size", "Open Time", "Close Time"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {', '.join(missing)}")
        return None
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df["Close Time"] = pd.to_datetime(df["Close Time"])
    df["Lot Size"] = pd.to_numeric(df["Lot Size"], errors="coerce").fillna(0)
    if "Open Price" not in df.columns:
        df["Open Price"] = 1.0
    else:
        df["Open Price"] = pd.to_numeric(df["Open Price"], errors="coerce").fillna(1.0)
    if "Close Price" not in df.columns:
        df["Close Price"] = 1.0
    else:
        df["Close Price"] = pd.to_numeric(df["Close Price"], errors="coerce").fillna(1.0)
    return df


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bonus Abuse Detection",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    div[data-testid="stMetric"] {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=48)
    st.title("Configuration")

    st.subheader("Data Source")
    data_mode = st.radio(
        "Choose input method",
        ["Sample Data", "Upload CSV", "Manual Entry"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Detection Thresholds")
    time_threshold = st.select_slider(
        "Time Match Window",
        options=[5, 10, 30, 60, 120, 300],
        value=30,
        format_func=lambda x: f"{x}s" if x < 60 else f"{x // 60}m",
    )
    lot_tolerance = st.slider(
        "Lot Size Tolerance",
        min_value=1, max_value=20, value=5,
        format="%d%%",
    ) / 100.0

    st.divider()
    st.subheader("Account A")
    ips_a_str = st.text_input("IP Addresses", "192.168.1.10, 10.0.0.5", key="ip_a")
    cids_a_str = st.text_input("Device IDs", "DEV-001, DEV-003", key="cid_a")
    bonus_a = st.number_input("Bonus Amount ($)", min_value=0.0, value=200.0, step=10.0, key="bonus_a")

    st.divider()
    st.subheader("Account B")
    ips_b_str = st.text_input("IP Addresses", "192.168.1.10, 10.0.0.8", key="ip_b")
    cids_b_str = st.text_input("Device IDs", "DEV-001, DEV-004", key="cid_b")
    bonus_b = st.number_input("Bonus Amount ($)", min_value=0.0, value=170.0, step=10.0, key="bonus_b")

account_info_a = {
    "ips": [x.strip() for x in ips_a_str.split(",") if x.strip()],
    "cids": [x.strip() for x in cids_a_str.split(",") if x.strip()],
    "bonus": bonus_a,
}
account_info_b = {
    "ips": [x.strip() for x in ips_b_str.split(",") if x.strip()],
    "cids": [x.strip() for x in cids_b_str.split(",") if x.strip()],
    "bonus": bonus_b,
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("Bonus Abuse Detection System")
    st.caption("Identify suspicious trading behavior and potential bonus exploitation between accounts")
with col_badge:
    st.markdown("")
    st.markdown(f"**Engine v1.0** &nbsp;|&nbsp; Max Score: **{MAX_SCORE}**")

# ---------------------------------------------------------------------------
# Load trade data
# ---------------------------------------------------------------------------

trades_a = None
trades_b = None

if data_mode == "Sample Data":
    trades_a = SAMPLE_TRADES_A.copy()
    trades_b = SAMPLE_TRADES_B.copy()

elif data_mode == "Upload CSV":
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        file_a = st.file_uploader("Account A Trades", type=["csv"], key="csv_a")
    with col_u2:
        file_b = st.file_uploader("Account B Trades", type=["csv"], key="csv_b")

    st.info(
        "Required columns: **Trade ID**, **Symbol**, **Direction**, "
        "**Lot Size**, **Open Time**, **Close Time**. "
        "Optional: **Open Price**, **Close Price**.",
        icon="\u2139\ufe0f",
    )
    if file_a:
        trades_a = parse_csv_trades(file_a)
    if file_b:
        trades_b = parse_csv_trades(file_b)

elif data_mode == "Manual Entry":
    tab_a, tab_b = st.tabs(["\U0001f4cb Account A Trades", "\U0001f4cb Account B Trades"])
    with tab_a:
        trades_a = st.data_editor(
            SAMPLE_TRADES_A.copy(), num_rows="dynamic",
            key="editor_a", use_container_width=True,
        )
    with tab_b:
        trades_b = st.data_editor(
            SAMPLE_TRADES_B.copy(), num_rows="dynamic",
            key="editor_b", use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

if trades_a is not None and trades_b is not None and not trades_a.empty and not trades_b.empty:
    results, total_score, exp_a, exp_b = analyze_pair(
        trades_a, trades_b, account_info_a, account_info_b,
        time_threshold, lot_tolerance,
    )
    risk_label, risk_color, risk_action = get_risk_level(total_score)
    matched_count = sum(1 for v in results.values() if v["matched"])
    score_pct = int(round(total_score / MAX_SCORE * 100))

    # ── Top Metrics ──────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score", f"{total_score} / {MAX_SCORE}", f"{score_pct}%")
    m2.metric("Risk Level", risk_label)
    m3.metric("Flags Triggered", f"{matched_count} / {len(DETECTION_PARAMS)}")
    m4.metric("Trades Analyzed", f"{len(trades_a)} + {len(trades_b)}")

    # ── Main Content Tabs ────────────────────────────────────────────────
    tab_overview, tab_details, tab_exposure, tab_trades, tab_workflow = st.tabs([
        "\U0001f3af Overview",
        "\U0001f50d Detection Details",
        "\U0001f4b9 Exposure Analysis",
        "\U0001f4c4 Trade Data",
        "\U0001f4cb Workflow & Export",
    ])

    # ── TAB: Overview ────────────────────────────────────────────────────
    with tab_overview:
        col_gauge, col_radar = st.columns(2)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=total_score,
                number=dict(font=dict(size=52, color=risk_color)),
                delta=dict(reference=40, increasing=dict(color="#ef4444"), decreasing=dict(color="#22c55e")),
                gauge=dict(
                    axis=dict(range=[0, MAX_SCORE], tickwidth=2, tickcolor="#334155",
                              tickfont=dict(color="#94a3b8")),
                    bgcolor="#1e293b",
                    bar=dict(color=risk_color, thickness=0.3),
                    steps=[
                        dict(range=[0, 20], color="rgba(34,197,94,0.15)"),
                        dict(range=[21, 40], color="rgba(234,179,8,0.15)"),
                        dict(range=[41, 60], color="rgba(249,115,22,0.15)"),
                        dict(range=[61, MAX_SCORE], color="rgba(239,68,68,0.15)"),
                    ],
                    threshold=dict(
                        line=dict(color=risk_color, width=4),
                        thickness=0.85, value=total_score,
                    ),
                ),
                title=dict(text="RISK SCORE", font=dict(size=16, color="#94a3b8")),
            ))
            fig_gauge.update_layout(
                height=350,
                margin=dict(l=30, r=30, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e2e8f0"),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_radar:
            labels = [DETECTION_PARAMS[k]["label"] for k in DETECTION_PARAMS]
            values = [DETECTION_PARAMS[k]["points"] if results[k]["matched"] else 0 for k in DETECTION_PARAMS]
            max_vals = [DETECTION_PARAMS[k]["points"] for k in DETECTION_PARAMS]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=max_vals + [max_vals[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.06)",
                line=dict(color="rgba(99,102,241,0.25)", width=1, dash="dot"),
                name="Maximum",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=f"rgba({int(risk_color[1:3],16)},{int(risk_color[3:5],16)},{int(risk_color[5:7],16)},0.2)",
                line=dict(color=risk_color, width=2.5),
                name="Detected",
                marker=dict(size=6, color=risk_color),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 25], showticklabels=False,
                                    gridcolor="rgba(99,102,241,0.1)"),
                    angularaxis=dict(tickfont=dict(size=10, color="#94a3b8"),
                                     gridcolor="rgba(99,102,241,0.1)"),
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, font=dict(color="#94a3b8")),
                margin=dict(l=60, r=60, t=40, b=40),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # Score breakdown bar
        bar_labels = [DETECTION_PARAMS[k]["label"] for k in DETECTION_PARAMS]
        bar_scored = [DETECTION_PARAMS[k]["points"] if results[k]["matched"] else 0 for k in DETECTION_PARAMS]
        bar_max = [DETECTION_PARAMS[k]["points"] for k in DETECTION_PARAMS]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bar_labels, y=bar_max, name="Max Points",
            marker_color="rgba(99,102,241,0.15)",
            marker_line=dict(color="rgba(99,102,241,0.3)", width=1),
        ))
        fig_bar.add_trace(go.Bar(
            x=bar_labels, y=bar_scored, name="Points Scored",
            marker_color=[risk_color if s > 0 else "rgba(99,102,241,0.08)" for s in bar_scored],
            text=[f"+{s}" if s > 0 else "" for s in bar_scored],
            textposition="outside",
            textfont=dict(color=risk_color, size=12, family="monospace"),
        ))
        fig_bar.update_layout(
            barmode="overlay", height=300,
            margin=dict(l=40, r=20, t=30, b=80),
            legend=dict(orientation="h", y=1.08, font=dict(color="#94a3b8")),
            yaxis=dict(title="Points", gridcolor="rgba(99,102,241,0.08)",
                       tickfont=dict(color="#94a3b8")),
            xaxis=dict(tickfont=dict(color="#94a3b8", size=9), tickangle=-35),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Risk action callout
        if total_score <= 20:
            st.success(f"**{risk_label} Risk** \u2014 {risk_action}", icon="\u2705")
        elif total_score <= 40:
            st.warning(f"**{risk_label} Risk** \u2014 {risk_action}", icon="\u26a0\ufe0f")
        elif total_score <= 60:
            st.warning(f"**{risk_label} Risk** \u2014 {risk_action}", icon="\U0001f6a8")
        else:
            st.error(f"**{risk_label} Risk** \u2014 {risk_action}", icon="\U0001f6a8")

    # ── TAB: Detection Details ───────────────────────────────────────────
    with tab_details:

        # --- Position-Level Detection ---
        st.subheader("Position-Level Detection")
        st.caption("Each row is a matched trade pair between accounts. All trade-level flags are shown per position.")

        pos_rows = build_position_detail_rows(
            trades_a, trades_b, account_info_a, account_info_b,
            time_threshold, lot_tolerance,
        )
        if pos_rows:
            pos_df = pd.DataFrame(pos_rows)
            st.dataframe(
                pos_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Lot A": st.column_config.NumberColumn(format="%.2f"),
                    "Lot B": st.column_config.NumberColumn(format="%.2f"),
                    "Open A": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "Open B": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "Close A": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "Close B": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm:ss"),
                    "Pair Score": st.column_config.NumberColumn(format="%d pts"),
                },
            )
            total_position_score = sum(r["Pair Score"] for r in pos_rows)
            st.markdown(f"**Total Position-Level Score: {total_position_score} pts**")
        else:
            st.info("No matching positions found between the two accounts.")

        st.divider()

        # --- Account-Level Detection ---
        st.subheader("Account-Level Detection")
        st.caption("Checks that compare account metadata: IP addresses, device IDs, bonus amounts, and total exposure.")

        acct_rows = build_account_detail_rows(account_info_a, account_info_b, exp_a, exp_b)
        acct_df = pd.DataFrame(acct_rows)
        st.dataframe(
            acct_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Points": st.column_config.NumberColumn(format="%d pts"),
            },
        )
        total_account_score = sum(r["Points"] for r in acct_rows)
        st.markdown(f"**Total Account-Level Score: {total_account_score} pts**")

        st.divider()

        # --- Combined Summary Table ---
        st.subheader("Full Detection Summary")
        summary_rows = []
        for key, cfg in DETECTION_PARAMS.items():
            matched = results[key]["matched"]
            summary_rows.append({
                "Parameter": cfg["label"],
                "Max Points": cfg["points"],
                "Flagged": "\u2705" if matched else "\u274c",
                "Points Scored": cfg["points"] if matched else 0,
                "Evidence": "; ".join(
                    d.replace("**", "") for d in results[key]["details"]
                ) if results[key]["details"] else "\u2014",
            })
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Max Points": st.column_config.NumberColumn(format="%d"),
                "Points Scored": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.markdown(f"**Grand Total: {total_score} / {MAX_SCORE} pts**")

    # ── TAB: Exposure Analysis ───────────────────────────────────────────
    with tab_exposure:
        st.subheader("Total Market Exposure")

        ex1, ex2 = st.columns(2)
        ex1.metric("Account A Exposure", f"${exp_a:,.0f}")
        ex2.metric("Account B Exposure", f"${exp_b:,.0f}")

        col_bar_exp, col_pie_exp = st.columns(2)
        with col_bar_exp:
            fig_exp = go.Figure(go.Bar(
                x=["Account A", "Account B"],
                y=[exp_a, exp_b],
                marker=dict(
                    color=["rgba(99,102,241,0.7)", "rgba(168,85,247,0.7)"],
                    line=dict(color=["#6366f1", "#a855f7"], width=2),
                ),
                text=[f"${exp_a:,.0f}", f"${exp_b:,.0f}"],
                textposition="outside",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_exp.update_layout(
                height=350, margin=dict(l=40, r=20, t=30, b=40),
                yaxis=dict(title="Exposure ($)", gridcolor="rgba(99,102,241,0.08)",
                           tickfont=dict(color="#94a3b8")),
                xaxis=dict(tickfont=dict(color="#e2e8f0", size=13)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_exp, use_container_width=True)

        with col_pie_exp:
            if exp_a + exp_b > 0:
                fig_pie = go.Figure(go.Pie(
                    labels=["Account A", "Account B"],
                    values=[exp_a, exp_b],
                    hole=0.55,
                    marker=dict(
                        colors=["rgba(99,102,241,0.8)", "rgba(168,85,247,0.8)"],
                        line=dict(color="#0f172a", width=3),
                    ),
                    textinfo="percent+label",
                    textfont=dict(color="#e2e8f0", size=12),
                    hoverinfo="label+value+percent",
                ))
                fig_pie.update_layout(
                    height=350, margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    annotations=[dict(
                        text=f"${(exp_a+exp_b):,.0f}",
                        x=0.5, y=0.5, font_size=14,
                        font_color="#94a3b8", showarrow=False,
                    )],
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        if exposure_similar(exp_a, exp_b):
            st.warning("Exposures are within 20% of each other \u2014 flagged as similar.", icon="\u26a0\ufe0f")
        else:
            st.info("Exposure difference exceeds 20% threshold \u2014 not flagged.", icon="\u2139\ufe0f")

    # ── TAB: Trade Data ──────────────────────────────────────────────────
    with tab_trades:
        td1, td2 = st.columns(2)
        with td1:
            st.subheader("Account A")
            st.dataframe(
                trades_a, use_container_width=True, hide_index=True,
                column_config={
                    "Direction": st.column_config.TextColumn(width="small"),
                    "Lot Size": st.column_config.NumberColumn(format="%.2f"),
                },
            )
        with td2:
            st.subheader("Account B")
            st.dataframe(
                trades_b, use_container_width=True, hide_index=True,
                column_config={
                    "Direction": st.column_config.TextColumn(width="small"),
                    "Lot Size": st.column_config.NumberColumn(format="%.2f"),
                },
            )

        st.divider()
        st.subheader("Matched Trade Pairs")
        pairs = match_trades(trades_a, trades_b, time_threshold)
        if pairs:
            pair_rows = []
            for ta, tb, dt in pairs:
                if ta["Symbol"] == tb["Symbol"]:
                    pair_rows.append({
                        "Account A Trade": ta["Trade ID"],
                        "Account B Trade": tb["Trade ID"],
                        "Symbol": ta["Symbol"],
                        "Dir A": ta["Direction"],
                        "Dir B": tb["Direction"],
                        "Lot A": ta["Lot Size"],
                        "Lot B": tb["Lot Size"],
                        "Time Diff (s)": round(dt, 1),
                        "Hedged": "Yes" if ta["Direction"] != tb["Direction"] else "No",
                    })
            if pair_rows:
                st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No symbol-matched trade pairs found.")
        else:
            st.info("No trade pairs matched between accounts.")

    # ── TAB: Workflow & Export ────────────────────────────────────────────
    with tab_workflow:
        st.subheader("Investigation Workflow")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown("##### Step 1")
            st.markdown("**Detection**")
            st.caption("Engine identifies suspicious linked accounts automatically.")
        with w2:
            st.markdown("##### Step 2")
            st.markdown("**Scoring**")
            st.caption("Risk score calculated based on matched parameters.")
        with w3:
            st.markdown("##### Step 3")
            st.markdown("**Analyst Review**")
            st.caption("Review trading history, device/IP logs, bonus and exposure data.")
        with w4:
            st.markdown("##### Step 4")
            st.markdown("**Decision**")
            st.caption("Clear account / Remove bonus / Restrict withdrawals / Suspend / Escalate.")

        st.divider()
        st.subheader("Risk Classification Reference")
        ref_df = pd.DataFrame({
            "Score Range": ["0 \u2013 20", "21 \u2013 40", "41 \u2013 60", f"61 \u2013 {MAX_SCORE}"],
            "Risk Level": ["Low", "Medium", "High", "Critical"],
            "Recommended Action": [a for _, _, _, _, a in RISK_LEVELS],
        })
        st.dataframe(ref_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Export Report")

        report = {
            "generated_at": datetime.now().isoformat(),
            "risk_score": total_score,
            "max_score": MAX_SCORE,
            "risk_level": risk_label,
            "flags_triggered": matched_count,
            "time_threshold_sec": time_threshold,
            "lot_tolerance_pct": lot_tolerance * 100,
            "account_a": {
                "trades_count": len(trades_a),
                "exposure": round(exp_a, 2),
                "bonus": bonus_a,
                "ips": account_info_a["ips"],
                "cids": account_info_a["cids"],
            },
            "account_b": {
                "trades_count": len(trades_b),
                "exposure": round(exp_b, 2),
                "bonus": bonus_b,
                "ips": account_info_b["ips"],
                "cids": account_info_b["cids"],
            },
            "detection_results": {
                DETECTION_PARAMS[k]["label"]: {
                    "matched": v["matched"],
                    "points": DETECTION_PARAMS[k]["points"] if v["matched"] else 0,
                    "details": v["details"],
                }
                for k, v in results.items()
            },
            "recommended_action": risk_action,
        }
        report_json = json.dumps(report, indent=2, default=str)

        col_dl1, col_dl2 = st.columns([1, 3])
        with col_dl1:
            st.download_button(
                label="\U0001f4e5 Download JSON",
                data=report_json,
                file_name=f"abuse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        with col_dl2:
            csv_data = pd.DataFrame([
                {
                    "Parameter": DETECTION_PARAMS[k]["label"],
                    "Matched": v["matched"],
                    "Points": DETECTION_PARAMS[k]["points"] if v["matched"] else 0,
                    "Details": "; ".join(v["details"]),
                }
                for k, v in results.items()
            ])
            st.download_button(
                label="\U0001f4e5 Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"abuse_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

else:
    st.divider()
    st.info(
        "Configure account data in the sidebar and provide trades for both accounts to begin analysis.",
        icon="\U0001f6e1\ufe0f",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Bonus Abuse Detection System \u2022 Risk & Compliance Department \u2022 "
    "Always perform manual review before permanent enforcement actions."
)
