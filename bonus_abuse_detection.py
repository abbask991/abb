import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import hashlib
import json
import io

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DETECTION_PARAMS = {
    "same_symbol": {"label": "Same Symbol", "points": 10},
    "opposite_direction": {"label": "Opposite Direction", "points": 20},
    "open_time_match": {"label": "Open Time Match", "points": 15},
    "close_time_match": {"label": "Close Time Match", "points": 15},
    "same_lot_size": {"label": "Same Lot Size", "points": 10},
    "shared_cid": {"label": "Shared CID (Device)", "points": 20},
    "shared_ip": {"label": "Shared IP", "points": 20},
    "bonus_similarity": {"label": "Bonus \u00b130% Similarity", "points": 10},
    "similar_exposure": {"label": "Similar Total Exposure", "points": 15},
}

MAX_SCORE = sum(p["points"] for p in DETECTION_PARAMS.values())

RISK_LEVELS = [
    (0, 20, "Low Risk", "#22c55e", "Monitor activity. No immediate action required."),
    (21, 40, "Medium Risk", "#f59e0b", "Review account behavior. Monitor withdrawals. Flag for observation."),
    (41, 60, "High Risk", "#f97316", "Temporary withdrawal review. Manual investigation. Compliance review required."),
    (61, MAX_SCORE, "Critical Risk", "#ef4444", "Immediate fraud investigation. Bonus cancellation consideration. Potential account suspension. Escalate to Risk & Compliance."),
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
                f"{pair_label}: both trade {ta['Symbol']}")

            if ta["Direction"] != tb["Direction"]:
                results["opposite_direction"]["matched"] = True
                results["opposite_direction"]["details"].append(
                    f"{pair_label}: {ta['Direction']} vs {tb['Direction']} on {ta['Symbol']}")

            if dt <= time_threshold_sec:
                results["open_time_match"]["matched"] = True
                results["open_time_match"]["details"].append(
                    f"{pair_label}: open times differ by {dt:.1f}s")

            close_dt = time_diff_seconds(ta["Close Time"], tb["Close Time"])
            if close_dt <= time_threshold_sec:
                results["close_time_match"]["matched"] = True
                results["close_time_match"]["details"].append(
                    f"{pair_label}: close times differ by {close_dt:.1f}s")

            if lot_similarity(ta["Lot Size"], tb["Lot Size"], lot_tolerance):
                results["same_lot_size"]["matched"] = True
                results["same_lot_size"]["details"].append(
                    f"{pair_label}: {ta['Lot Size']} vs {tb['Lot Size']} lots")

    cid_a = set(account_info_a.get("cids", []))
    cid_b = set(account_info_b.get("cids", []))
    shared_cids = cid_a & cid_b
    if shared_cids:
        results["shared_cid"]["matched"] = True
        results["shared_cid"]["details"].append(
            f"Shared devices: {', '.join(shared_cids)}")

    ip_a = set(account_info_a.get("ips", []))
    ip_b = set(account_info_b.get("ips", []))
    shared_ips = ip_a & ip_b
    if shared_ips:
        results["shared_ip"]["matched"] = True
        results["shared_ip"]["details"].append(
            f"Shared IPs: {', '.join(shared_ips)}")

    bonus_a = account_info_a.get("bonus", 0)
    bonus_b = account_info_b.get("bonus", 0)
    if bonus_a > 0 and bonus_b > 0 and bonus_within_range(bonus_a, bonus_b):
        results["bonus_similarity"]["matched"] = True
        results["bonus_similarity"]["details"].append(
            f"${bonus_a:.2f} vs ${bonus_b:.2f} (within \u00b130%)")

    exp_a = compute_exposure(trades_a)
    exp_b = compute_exposure(trades_b)
    if exposure_similar(exp_a, exp_b):
        results["similar_exposure"]["matched"] = True
        results["similar_exposure"]["details"].append(
            f"Exposure A: ${exp_a:,.2f} vs B: ${exp_b:,.2f}")

    total_score = sum(
        DETECTION_PARAMS[k]["points"] for k, v in results.items() if v["matched"]
    )
    return results, total_score, exp_a, exp_b


def build_radar_data(results):
    labels = []
    values = []
    for key, cfg in DETECTION_PARAMS.items():
        labels.append(cfg["label"])
        values.append(cfg["points"] if results[key]["matched"] else 0)
    return labels, values


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
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bonus Abuse Detection System",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.25rem;
        border-left: 4px solid #3b82f6;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        color: #fff;
    }
    .param-row {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .param-matched {
        background: #fef2f2;
        border-left: 4px solid #ef4444;
    }
    .param-clear {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
    }
    .detail-text {
        font-size: 0.85rem;
        color: #64748b;
        margin-left: 1rem;
    }
    .score-display {
        font-size: 4rem;
        font-weight: 800;
        text-align: center;
        line-height: 1.1;
    }
    .workflow-step {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .workflow-step h4 {
        margin: 0 0 0.25rem 0;
        color: #1e293b;
    }
    .workflow-step p {
        margin: 0;
        color: #64748b;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    data_mode = st.radio(
        "Data Source",
        ["Sample Data", "Upload CSV", "Manual Entry"],
        help="Choose how to provide trade data for analysis.",
    )

    st.markdown("### Detection Thresholds")
    time_threshold = st.selectbox(
        "Time Match Threshold",
        [5, 10, 30, 60, 120, 300],
        index=2,
        format_func=lambda x: f"{x} seconds",
        help="Maximum time difference (seconds) to consider trades synchronized.",
    )
    lot_tolerance = st.slider(
        "Lot Size Tolerance (%)",
        min_value=1, max_value=20, value=5,
        help="Percentage tolerance for lot size comparison.",
    ) / 100.0

    st.markdown("### Account Metadata")
    st.markdown("**Account A**")
    ips_a_str = st.text_input("IPs (comma-sep)", "192.168.1.10, 10.0.0.5", key="ip_a")
    cids_a_str = st.text_input("Device IDs (comma-sep)", "DEV-001, DEV-003", key="cid_a")
    bonus_a = st.number_input("Bonus ($)", min_value=0.0, value=200.0, step=10.0, key="bonus_a")

    st.markdown("**Account B**")
    ips_b_str = st.text_input("IPs (comma-sep)", "192.168.1.10, 10.0.0.8", key="ip_b")
    cids_b_str = st.text_input("Device IDs (comma-sep)", "DEV-001, DEV-004", key="cid_b")
    bonus_b = st.number_input("Bonus ($)", min_value=0.0, value=170.0, step=10.0, key="bonus_b")

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

st.markdown('<p class="main-header">Bonus Abuse Detection System</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Identify suspicious trading behavior and potential bonus exploitation between trading accounts</p>',
    unsafe_allow_html=True,
)

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
        st.markdown("#### Account A Trades CSV")
        file_a = st.file_uploader("Upload Account A", type=["csv"], key="csv_a")
    with col_u2:
        st.markdown("#### Account B Trades CSV")
        file_b = st.file_uploader("Upload Account B", type=["csv"], key="csv_b")

    st.info(
        "CSV must contain columns: **Trade ID**, **Symbol**, **Direction**, "
        "**Lot Size**, **Open Time**, **Close Time**. "
        "Optional: **Open Price**, **Close Price**."
    )
    if file_a:
        trades_a = parse_csv_trades(file_a)
    if file_b:
        trades_b = parse_csv_trades(file_b)

elif data_mode == "Manual Entry":
    st.markdown("#### Enter Trade Data")
    tab_a, tab_b = st.tabs(["Account A Trades", "Account B Trades"])

    with tab_a:
        st.markdown("Edit the table below to define Account A trades.")
        trades_a = st.data_editor(
            SAMPLE_TRADES_A.copy(),
            num_rows="dynamic",
            key="editor_a",
            use_container_width=True,
        )
    with tab_b:
        st.markdown("Edit the table below to define Account B trades.")
        trades_b = st.data_editor(
            SAMPLE_TRADES_B.copy(),
            num_rows="dynamic",
            key="editor_b",
            use_container_width=True,
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

    # --- Top metrics row ---
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f'<div class="metric-card"><h4 style="margin:0;color:#64748b">Risk Score</h4>'
            f'<p style="font-size:2.5rem;font-weight:800;margin:0;color:{risk_color}">{total_score}</p>'
            f'<p style="margin:0;color:#94a3b8">out of {MAX_SCORE}</p></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-card"><h4 style="margin:0;color:#64748b">Risk Level</h4>'
            f'<p><span class="risk-badge" style="background:{risk_color}">{risk_label}</span></p></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-card"><h4 style="margin:0;color:#64748b">Flags Triggered</h4>'
            f'<p style="font-size:2.5rem;font-weight:800;margin:0;color:#1e293b">{matched_count}</p>'
            f'<p style="margin:0;color:#94a3b8">of {len(DETECTION_PARAMS)} checks</p></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-card"><h4 style="margin:0;color:#64748b">Trade Pairs</h4>'
            f'<p style="font-size:2.5rem;font-weight:800;margin:0;color:#1e293b">'
            f'{len(trades_a)} / {len(trades_b)}</p>'
            f'<p style="margin:0;color:#94a3b8">Account A / B</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # --- Detailed results + charts ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Detection Results")
        for key, cfg in DETECTION_PARAMS.items():
            matched = results[key]["matched"]
            css_class = "param-matched" if matched else "param-clear"
            icon = "\u274c" if matched else "\u2705"
            pts = f"+{cfg['points']} pts" if matched else "0 pts"
            st.markdown(
                f'<div class="param-row {css_class}">'
                f'<span style="font-size:1.1rem;min-width:1.5rem">{icon}</span>'
                f'<span style="flex:1;font-weight:600;margin-left:0.5rem">{cfg["label"]}</span>'
                f'<span style="font-weight:700;color:{risk_color if matched else "#22c55e"}">{pts}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if matched and results[key]["details"]:
                for d in results[key]["details"]:
                    st.markdown(f'<p class="detail-text">\u2022 {d}</p>', unsafe_allow_html=True)

    with col_right:
        st.markdown("### Risk Radar")
        labels, values = build_radar_data(results)
        max_vals = [DETECTION_PARAMS[k]["points"] for k in DETECTION_PARAMS]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=max_vals + [max_vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="rgba(59,130,246,0.3)", width=1),
            name="Maximum",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            fillcolor=f"rgba({int(risk_color[1:3],16)},{int(risk_color[3:5],16)},{int(risk_color[5:7],16)},0.25)",
            line=dict(color=risk_color, width=2),
            name="Detected",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 25], showticklabels=False),
                angularaxis=dict(tickfont=dict(size=10)),
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=40, r=40, t=20, b=40),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Score gauge
        st.markdown("### Score Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_score,
            gauge=dict(
                axis=dict(range=[0, MAX_SCORE], tickwidth=1),
                bar=dict(color=risk_color),
                steps=[
                    dict(range=[0, 20], color="#dcfce7"),
                    dict(range=[21, 40], color="#fef9c3"),
                    dict(range=[41, 60], color="#ffedd5"),
                    dict(range=[61, MAX_SCORE], color="#fee2e2"),
                ],
                threshold=dict(line=dict(color="#1e293b", width=3), thickness=0.8, value=total_score),
            ),
            title=dict(text="Risk Score"),
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # --- Score breakdown bar chart ---
    st.markdown("### Score Breakdown")
    bar_labels = [DETECTION_PARAMS[k]["label"] for k in DETECTION_PARAMS]
    bar_scored = [DETECTION_PARAMS[k]["points"] if results[k]["matched"] else 0 for k in DETECTION_PARAMS]
    bar_max = [DETECTION_PARAMS[k]["points"] for k in DETECTION_PARAMS]
    bar_colors = [risk_color if results[k]["matched"] else "#e2e8f0" for k in DETECTION_PARAMS]

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=bar_labels, y=bar_max, name="Max Points",
        marker_color="#e2e8f0", text=bar_max, textposition="outside",
    ))
    fig_bar.add_trace(go.Bar(
        x=bar_labels, y=bar_scored, name="Scored",
        marker_color=bar_colors, text=bar_scored, textposition="outside",
    ))
    fig_bar.update_layout(
        barmode="overlay", height=350,
        margin=dict(l=40, r=20, t=20, b=80),
        legend=dict(orientation="h", y=1.08),
        yaxis_title="Points",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Exposure comparison ---
    st.markdown("### Exposure Comparison")
    ec1, ec2 = st.columns(2)
    with ec1:
        fig_exp = go.Figure(go.Bar(
            x=["Account A", "Account B"],
            y=[exp_a, exp_b],
            marker_color=["#3b82f6", "#8b5cf6"],
            text=[f"${exp_a:,.0f}", f"${exp_b:,.0f}"],
            textposition="outside",
        ))
        fig_exp.update_layout(
            height=300, margin=dict(l=40, r=20, t=20, b=40),
            yaxis_title="Total Exposure ($)",
        )
        st.plotly_chart(fig_exp, use_container_width=True)
    with ec2:
        if exp_a + exp_b > 0:
            fig_pie = go.Figure(go.Pie(
                labels=["Account A", "Account B"],
                values=[exp_a, exp_b],
                hole=0.5,
                marker=dict(colors=["#3b82f6", "#8b5cf6"]),
                textinfo="percent+label",
            ))
            fig_pie.update_layout(
                height=300, margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- Trade Data Viewer ---
    st.markdown("### Trade Data")
    td1, td2 = st.columns(2)
    with td1:
        st.markdown("**Account A**")
        st.dataframe(trades_a, use_container_width=True, hide_index=True)
    with td2:
        st.markdown("**Account B**")
        st.dataframe(trades_b, use_container_width=True, hide_index=True)

    # --- Recommended Actions ---
    st.markdown("### Recommended Actions")
    st.markdown(
        f'<div style="background:{risk_color}15;border-left:4px solid {risk_color};'
        f'border-radius:8px;padding:1rem 1.5rem">'
        f'<h4 style="margin:0 0 0.5rem 0;color:{risk_color}">{risk_label} \u2014 Action Plan</h4>'
        f'<p style="margin:0;color:#334155">{risk_action}</p></div>',
        unsafe_allow_html=True,
    )

    # --- Investigation Workflow ---
    st.markdown("### Investigation Workflow")
    wf1, wf2, wf3, wf4 = st.columns(4)
    steps = [
        ("Step 1", "Detection", "Engine identifies suspicious linked accounts"),
        ("Step 2", "Scoring", "Risk score calculated automatically"),
        ("Step 3", "Analyst Review", "Review trading, device, IP, bonus history"),
        ("Step 4", "Decision", "Clear / Remove bonus / Restrict / Suspend / Escalate"),
    ]
    for col, (step, title, desc) in zip([wf1, wf2, wf3, wf4], steps):
        with col:
            st.markdown(
                f'<div class="workflow-step">'
                f'<p style="color:#3b82f6;font-weight:700;margin:0">{step}</p>'
                f'<h4>{title}</h4>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    # --- Export ---
    st.markdown("### Export Report")
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

    st.download_button(
        label="Download JSON Report",
        data=report_json,
        file_name=f"abuse_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
    )

else:
    st.info("Please provide trade data for both accounts to begin the analysis.")

# --- Footer ---
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#94a3b8;font-size:0.85rem">'
    'Bonus Abuse Detection System &bull; Risk & Compliance Department &bull; '
    'Always perform manual review before permanent enforcement actions.</p>',
    unsafe_allow_html=True,
)
