"""
IntegrateGlobals - Social Intelligence & Digital Discourse Analysis Platform
National platform for monitoring, analyzing, and transforming digital data
into early warning indicators and strategic insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
from collections import Counter
import json
import io
import base64

from data_generator import (
    generate_posts, generate_accounts, generate_alerts,
    generate_narratives, generate_network_data,
    ARABIC_TOPICS, ARABIC_HASHTAGS, PLATFORMS, REGIONS, LANGUAGES,
    NARRATIVE_THEMES, SENTIMENT_LABELS, STANCE_LABELS
)

# ---------------------------------------------------------------------------
# Page & Theme Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IntegrateGlobals | Social Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#6366f1",
    "secondary": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "text": "#e2e8f0",
    "positive": "#10b981",
    "negative": "#ef4444",
    "neutral": "#6b7280",
}

SEVERITY_COLORS = {
    "حرج": "#ef4444",
    "عالي": "#f59e0b",
    "متوسط": "#3b82f6",
    "منخفض": "#10b981",
}


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap');

    * { font-family: 'Tajawal', sans-serif !important; }

    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #e2e8f0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #6366f1;
        margin: 0.3rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .metric-delta {
        font-size: 0.8rem;
        padding: 2px 8px;
        border-radius: 12px;
        display: inline-block;
    }
    .metric-delta.positive { background: rgba(16,185,129,0.15); color: #10b981; }
    .metric-delta.negative { background: rgba(239,68,68,0.15); color: #ef4444; }
    .metric-delta.neutral  { background: rgba(107,114,128,0.15); color: #6b7280; }

    .section-header {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366f1;
    }

    .alert-card {
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid;
    }
    .alert-critical { background: rgba(239,68,68,0.1); border-color: #ef4444; }
    .alert-high     { background: rgba(245,158,11,0.1); border-color: #f59e0b; }
    .alert-medium   { background: rgba(59,130,246,0.1); border-color: #3b82f6; }
    .alert-low      { background: rgba(16,185,129,0.1); border-color: #10b981; }

    .platform-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .platform-header h1 {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .platform-header p {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    .narrative-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-low    { background: rgba(16,185,129,0.2); color: #10b981; }
    .risk-medium { background: rgba(59,130,246,0.2); color: #3b82f6; }
    .risk-high   { background: rgba(245,158,11,0.2); color: #f59e0b; }
    .risk-critical { background: rgba(239,68,68,0.2); color: #ef4444; }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem 0 1rem 0;
        font-size: 0.8rem;
        border-top: 1px solid #334155;
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    posts = generate_posts(n=5000, days=90, seed=42)
    accounts = generate_accounts(n=200, seed=42)
    alerts = generate_alerts(posts, seed=42)
    narratives = generate_narratives(posts, seed=42)
    network = generate_network_data(accounts, seed=42)
    return posts, accounts, alerts, narratives, network


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def metric_card(label, value, delta=None, delta_type="neutral"):
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-delta {delta_type}">{delta}</div>'
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def severity_to_css(severity):
    mapping = {"حرج": "critical", "عالي": "high", "متوسط": "medium", "منخفض": "low"}
    return mapping.get(severity, "medium")


def format_number(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(int(n))


def get_filtered_data(posts, date_range, platforms, sentiments, regions):
    filtered = posts.copy()
    if date_range:
        start, end = date_range
        filtered = filtered[
            (filtered["date"] >= start) & (filtered["date"] <= end)
        ]
    if platforms:
        filtered = filtered[filtered["platform"].isin(platforms)]
    if sentiments:
        filtered = filtered[filtered["sentiment"].isin(sentiments)]
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    return filtered


# ---------------------------------------------------------------------------
# PAGES
# ---------------------------------------------------------------------------

def render_dashboard(posts, accounts, alerts, narratives):
    """Main Overview Dashboard - Core functions 1-10"""
    st.markdown('<div class="platform-header"><h1>IntegrateGlobals - Social Intelligence Platform</h1><p>National Platform for Digital Discourse Analysis and Strategic Intelligence</p></div>', unsafe_allow_html=True)

    total_posts = len(posts)
    total_accounts = len(accounts)
    total_engagement = posts["engagement"].sum()
    avg_sentiment = posts["sentiment_score"].mean()
    active_alerts = len(alerts[alerts["status"] == "جديد"])
    positive_pct = (posts["sentiment"] == "إيجابي").mean() * 100
    negative_pct = (posts["sentiment"] == "سلبي").mean() * 100

    week_ago = posts["date"].max() - timedelta(days=7)
    recent = posts[posts["date"] > week_ago]
    prev = posts[(posts["date"] <= week_ago) & (posts["date"] > week_ago - timedelta(days=7))]
    vol_change = ((len(recent) - len(prev)) / max(len(prev), 1)) * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(metric_card("Total Posts", format_number(total_posts), f"{vol_change:+.1f}%", "positive" if vol_change > 0 else "negative"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Active Accounts", format_number(total_accounts)), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Total Engagement", format_number(total_engagement)), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Avg Sentiment", f"{avg_sentiment:.2f}", f"{positive_pct:.0f}% Positive", "positive" if avg_sentiment > 0 else "negative"), unsafe_allow_html=True)
    with c5:
        st.markdown(metric_card("Active Alerts", str(active_alerts), "Require Attention", "negative" if active_alerts > 5 else "neutral"), unsafe_allow_html=True)
    with c6:
        bot_pct = accounts["is_bot_suspect"].mean() * 100
        st.markdown(metric_card("Bot Suspects", f"{bot_pct:.1f}%", f"{accounts['is_bot_suspect'].sum()} Accounts", "negative"), unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">Post Volume Over Time</div>', unsafe_allow_html=True)
        daily = posts.groupby("date").agg(
            count=("post_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            total_engagement=("engagement", "sum")
        ).reset_index()
        daily["ma7"] = daily["count"].rolling(7, min_periods=1).mean()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=daily["date"], y=daily["count"], name="Daily Posts",
                   marker_color="rgba(99,102,241,0.4)"),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=daily["date"], y=daily["ma7"], name="7-Day Average",
                       line=dict(color="#8b5cf6", width=2)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=daily["date"], y=daily["avg_sentiment"], name="Avg Sentiment",
                       line=dict(color="#10b981", width=1.5, dash="dot")),
            secondary_y=True
        )
        fig.update_layout(
            height=380,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=30, l=50, r=50),
            legend=dict(orientation="h", y=1.12),
        )
        fig.update_yaxes(title_text="Post Count", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Recent Alerts</div>', unsafe_allow_html=True)
        recent_alerts = alerts.head(6)
        for _, alert in recent_alerts.iterrows():
            css_class = severity_to_css(alert["severity"])
            st.markdown(f"""
            <div class="alert-card alert-{css_class}">
                <strong>{alert['type']}</strong><br/>
                <small>{alert['topic']} | {alert['severity']} | {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-header">Sentiment Distribution</div>', unsafe_allow_html=True)
        sent_counts = posts["sentiment"].value_counts()
        fig_sent = go.Figure(go.Pie(
            labels=sent_counts.index,
            values=sent_counts.values,
            hole=0.55,
            marker_colors=[COLORS["positive"], COLORS["negative"], COLORS["neutral"]],
            textinfo="label+percent",
        ))
        fig_sent.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Platform Distribution</div>', unsafe_allow_html=True)
        plat_counts = posts["platform"].value_counts()
        fig_plat = go.Figure(go.Bar(
            x=plat_counts.values,
            y=plat_counts.index,
            orientation="h",
            marker_color=COLORS["primary"],
        ))
        fig_plat.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=100, r=20),
        )
        st.plotly_chart(fig_plat, use_container_width=True)

    with col3:
        st.markdown('<div class="section-header">Geographic Distribution</div>', unsafe_allow_html=True)
        region_counts = posts["region"].value_counts().head(10)
        fig_region = go.Figure(go.Bar(
            x=region_counts.values,
            y=region_counts.index,
            orientation="h",
            marker_color=COLORS["secondary"],
        ))
        fig_region.update_layout(
            height=300,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=100, r=20),
        )
        st.plotly_chart(fig_region, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Top Hashtags</div>', unsafe_allow_html=True)
        all_tags = []
        for tags in posts["hashtags"]:
            if isinstance(tags, list):
                all_tags.extend(tags)
        tag_counts = pd.Series(all_tags).value_counts().head(15)
        fig_tags = go.Figure(go.Bar(
            x=tag_counts.values,
            y=tag_counts.index,
            orientation="h",
            marker=dict(color=tag_counts.values, colorscale="Viridis"),
        ))
        fig_tags.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=180, r=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_tags, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Most Active Accounts</div>', unsafe_allow_html=True)
        top_accounts = posts.groupby("username").agg(
            posts=("post_id", "count"),
            engagement=("engagement", "sum"),
            avg_sentiment=("sentiment_score", "mean"),
        ).sort_values("engagement", ascending=False).head(15).reset_index()

        fig_acct = go.Figure(go.Bar(
            x=top_accounts["engagement"],
            y=top_accounts["username"],
            orientation="h",
            marker=dict(color=top_accounts["avg_sentiment"], colorscale="RdYlGn", cmin=-1, cmax=1),
            text=top_accounts["posts"].apply(lambda x: f"{x} posts"),
            textposition="inside",
        ))
        fig_acct.update_layout(
            height=400,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20, l=100, r=20),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_acct, use_container_width=True)

    st.markdown('<div class="section-header">Activity Heatmap by Hour and Day</div>', unsafe_allow_html=True)
    posts["day_of_week"] = pd.to_datetime(posts["timestamp"]).dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heatmap_data = posts.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(day_order)

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h}:00" for h in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale="Viridis",
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=30, l=100, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


def render_keyword_monitoring(posts):
    """Keyword & Hashtag Monitoring - Functions 1-5"""
    st.markdown('<div class="section-header">Keyword & Hashtag Monitoring</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        keyword = st.text_input("Search Keywords", placeholder="Enter keyword to monitor...")
    with col_f2:
        selected_hashtags = st.multiselect("Filter by Hashtags", ARABIC_HASHTAGS)
    with col_f3:
        selected_platforms = st.multiselect("Filter by Platform", PLATFORMS, key="kw_platforms")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        date_start = st.date_input("Start Date", posts["date"].min(), key="kw_start")
    with col_d2:
        date_end = st.date_input("End Date", posts["date"].max(), key="kw_end")

    filtered = posts.copy()
    if keyword:
        filtered = filtered[filtered["text"].str.contains(keyword, case=False, na=False)]
    if selected_hashtags:
        filtered = filtered[filtered["hashtags"].apply(
            lambda x: any(h in x for h in selected_hashtags) if isinstance(x, list) else False
        )]
    if selected_platforms:
        filtered = filtered[filtered["platform"].isin(selected_platforms)]
    filtered = filtered[(filtered["date"] >= date_start) & (filtered["date"] <= date_end)]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Matched Posts", format_number(len(filtered))), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Total Engagement", format_number(filtered["engagement"].sum())), unsafe_allow_html=True)
    with c3:
        avg_s = filtered["sentiment_score"].mean() if len(filtered) > 0 else 0
        st.markdown(metric_card("Avg Sentiment", f"{avg_s:.2f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Unique Accounts", format_number(filtered["username"].nunique())), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Volume Timeline", "Posts Feed", "Analytics"])

    with tab1:
        if len(filtered) > 0:
            daily = filtered.groupby("date").size().reset_index(name="count")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["date"], y=daily["count"],
                fill="tozeroy", fillcolor="rgba(99,102,241,0.2)",
                line=dict(color=COLORS["primary"], width=2),
                name="Daily Posts"
            ))
            fig.update_layout(
                height=350,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No posts match the current filters.")

    with tab2:
        if len(filtered) > 0:
            display_cols = ["timestamp", "username", "platform", "text", "sentiment", "engagement", "hashtags_str"]
            st.dataframe(
                filtered[display_cols].sort_values("timestamp", ascending=False).head(100),
                use_container_width=True,
                height=500,
            )
        else:
            st.info("No posts to display.")

    with tab3:
        if len(filtered) > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                sent_dist = filtered["sentiment"].value_counts()
                fig_s = go.Figure(go.Pie(
                    labels=sent_dist.index, values=sent_dist.values,
                    hole=0.5,
                    marker_colors=[COLORS["positive"], COLORS["negative"], COLORS["neutral"]],
                ))
                fig_s.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Sentiment Breakdown")
                st.plotly_chart(fig_s, use_container_width=True)
            with col_b:
                plat_dist = filtered["platform"].value_counts()
                fig_p = go.Figure(go.Bar(x=plat_dist.index, y=plat_dist.values, marker_color=COLORS["primary"]))
                fig_p.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Platform Distribution")
                st.plotly_chart(fig_p, use_container_width=True)


def render_sentiment_analysis(posts):
    """Sentiment & Stance Analysis - Functions 6-7, 27"""
    st.markdown('<div class="section-header">Sentiment & Stance Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Overview", "Sentiment Over Time", "Stance Analysis", "Topic Sentiment"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        for col, sentiment, color in [(c1, "إيجابي", COLORS["positive"]),
                                       (c2, "سلبي", COLORS["negative"]),
                                       (c3, "محايد", COLORS["neutral"])]:
            count = (posts["sentiment"] == sentiment).sum()
            pct = count / len(posts) * 100
            with col:
                st.markdown(metric_card(sentiment, f"{pct:.1f}%", f"{format_number(count)} posts"), unsafe_allow_html=True)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=posts["sentiment_score"], nbinsx=50,
            marker_color=COLORS["primary"],
            opacity=0.7,
        ))
        fig_dist.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig_dist.update_layout(
            height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Sentiment Score Distribution",
            xaxis_title="Score (-1 to +1)", yaxis_title="Count",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab2:
        daily_sent = posts.groupby(["date", "sentiment"]).size().unstack(fill_value=0).reset_index()
        fig_ts = go.Figure()
        for sent, color in [("إيجابي", COLORS["positive"]), ("سلبي", COLORS["negative"]), ("محايد", COLORS["neutral"])]:
            if sent in daily_sent.columns:
                fig_ts.add_trace(go.Scatter(
                    x=daily_sent["date"], y=daily_sent[sent],
                    name=sent, stackgroup="one",
                    line=dict(color=color),
                ))
        fig_ts.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Sentiment Trends Over Time",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown('<div class="section-header">Sentiment Shift Detection</div>', unsafe_allow_html=True)
        daily_avg = posts.groupby("date")["sentiment_score"].mean().reset_index()
        daily_avg["shift"] = daily_avg["sentiment_score"].diff()
        daily_avg["is_shift"] = daily_avg["shift"].abs() > daily_avg["shift"].std() * 1.5

        fig_shift = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig_shift.add_trace(go.Scatter(
            x=daily_avg["date"], y=daily_avg["sentiment_score"],
            line=dict(color=COLORS["primary"]), name="Avg Sentiment"
        ), row=1, col=1)
        shifts = daily_avg[daily_avg["is_shift"]]
        fig_shift.add_trace(go.Scatter(
            x=shifts["date"], y=shifts["sentiment_score"],
            mode="markers", marker=dict(color=COLORS["danger"], size=10, symbol="triangle-up"),
            name="Sentiment Shift"
        ), row=1, col=1)
        fig_shift.add_trace(go.Bar(
            x=daily_avg["date"], y=daily_avg["shift"],
            marker_color=daily_avg["shift"].apply(lambda x: COLORS["positive"] if x > 0 else COLORS["negative"]),
            name="Daily Change"
        ), row=2, col=1)
        fig_shift.update_layout(
            height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_shift, use_container_width=True)

    with tab3:
        st.markdown("**Stance Analysis** - Classifying positions as supportive, opposing, sarcastic, or neutral")
        stance_counts = posts["stance"].value_counts()
        col_a, col_b = st.columns(2)
        with col_a:
            fig_stance = go.Figure(go.Pie(
                labels=stance_counts.index, values=stance_counts.values,
                hole=0.5, marker_colors=["#10b981", "#ef4444", "#f59e0b", "#6b7280"],
            ))
            fig_stance.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_stance, use_container_width=True)
        with col_b:
            stance_by_platform = posts.groupby(["platform", "stance"]).size().unstack(fill_value=0)
            fig_sb = go.Figure()
            for stance in STANCE_LABELS:
                if stance in stance_by_platform.columns:
                    fig_sb.add_trace(go.Bar(name=stance, x=stance_by_platform.index, y=stance_by_platform[stance]))
            fig_sb.update_layout(
                barmode="stack", height=350, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", title="Stance by Platform",
            )
            st.plotly_chart(fig_sb, use_container_width=True)

    with tab4:
        topic_sent = posts.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
        topic_sent["total"] = topic_sent.sum(axis=1)
        topic_sent = topic_sent.sort_values("total", ascending=True).tail(15)
        fig_ts = go.Figure()
        for sent, color in [("إيجابي", COLORS["positive"]), ("محايد", COLORS["neutral"]), ("سلبي", COLORS["negative"])]:
            if sent in topic_sent.columns:
                fig_ts.add_trace(go.Bar(name=sent, y=topic_sent.index, x=topic_sent[sent], orientation="h", marker_color=color))
        fig_ts.update_layout(
            barmode="stack", height=500, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", title="Sentiment by Topic",
            margin=dict(l=200),
        )
        st.plotly_chart(fig_ts, use_container_width=True)


def render_trend_analysis(posts):
    """Trend Detection & Analysis - Functions 11-17"""
    st.markdown('<div class="section-header">Trend Detection & Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Trending Topics", "Velocity Tracking", "Emerging Keywords", "Period Comparison"])

    with tab1:
        recent_window = st.slider("Recent Window (days)", 1, 30, 7, key="trend_window")
        cutoff = posts["date"].max() - timedelta(days=recent_window)
        recent = posts[posts["date"] >= cutoff]
        prev = posts[(posts["date"] < cutoff) & (posts["date"] >= cutoff - timedelta(days=recent_window))]

        topic_recent = recent["topic"].value_counts()
        topic_prev = prev["topic"].value_counts()

        trend_df = pd.DataFrame({
            "topic": ARABIC_TOPICS,
        })
        trend_df["recent_count"] = trend_df["topic"].map(topic_recent).fillna(0)
        trend_df["prev_count"] = trend_df["topic"].map(topic_prev).fillna(0)
        trend_df["change_pct"] = ((trend_df["recent_count"] - trend_df["prev_count"]) / trend_df["prev_count"].replace(0, 1) * 100)
        trend_df["is_trending"] = trend_df["change_pct"] > 20
        trend_df = trend_df.sort_values("change_pct", ascending=False)

        fig_trend = go.Figure()
        colors = trend_df["change_pct"].apply(lambda x: COLORS["positive"] if x > 0 else COLORS["negative"])
        fig_trend.add_trace(go.Bar(
            y=trend_df["topic"], x=trend_df["change_pct"],
            orientation="h", marker_color=colors,
            text=trend_df["change_pct"].apply(lambda x: f"{x:+.0f}%"),
            textposition="outside",
        ))
        fig_trend.update_layout(
            height=600, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Topic Trend Changes",
            xaxis_title="Change %",
            margin=dict(l=200, r=80),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.markdown("**Spread Velocity** - Measuring how fast topics gain traction")
        hourly = posts.groupby([pd.Grouper(key="timestamp", freq="6h"), "topic"]).size().reset_index(name="count")
        top_topics = posts["topic"].value_counts().head(5).index

        fig_vel = go.Figure()
        for topic in top_topics:
            topic_data = hourly[hourly["topic"] == topic]
            fig_vel.add_trace(go.Scatter(
                x=topic_data["timestamp"], y=topic_data["count"],
                name=topic, mode="lines",
            ))
        fig_vel.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Topic Velocity (6-hour windows)",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_vel, use_container_width=True)

        velocity_data = []
        for topic in ARABIC_TOPICS:
            topic_posts = posts[posts["topic"] == topic].sort_values("timestamp")
            if len(topic_posts) < 2:
                continue
            time_span = (topic_posts["timestamp"].max() - topic_posts["timestamp"].min()).total_seconds() / 3600
            velocity = len(topic_posts) / max(time_span, 1)
            velocity_data.append({"topic": topic, "velocity": velocity, "total_posts": len(topic_posts)})

        vel_df = pd.DataFrame(velocity_data).sort_values("velocity", ascending=False)
        fig_vb = go.Figure(go.Bar(
            x=vel_df["topic"], y=vel_df["velocity"],
            marker_color=COLORS["secondary"],
            text=vel_df["velocity"].apply(lambda x: f"{x:.1f}/hr"),
            textposition="outside",
        ))
        fig_vb.update_layout(
            height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Topic Velocity Score (posts/hour)",
            margin=dict(t=40, b=100),
            xaxis_tickangle=45,
        )
        st.plotly_chart(fig_vb, use_container_width=True)

    with tab3:
        st.markdown("**Emerging Keywords & Hashtags** - Detecting rising trends early")
        all_tags = []
        for _, row in posts.iterrows():
            if isinstance(row["hashtags"], list):
                for tag in row["hashtags"]:
                    all_tags.append({"tag": tag, "date": row["date"]})
        tags_df = pd.DataFrame(all_tags)

        if len(tags_df) > 0:
            mid_point = tags_df["date"].median()
            early = tags_df[tags_df["date"] <= mid_point]["tag"].value_counts()
            late = tags_df[tags_df["date"] > mid_point]["tag"].value_counts()

            emergence = pd.DataFrame({"early": early, "late": late}).fillna(0)
            emergence["growth"] = ((emergence["late"] - emergence["early"]) / emergence["early"].replace(0, 1)) * 100
            emergence = emergence.sort_values("growth", ascending=False).head(15)

            fig_em = go.Figure(go.Bar(
                x=emergence["growth"], y=emergence.index,
                orientation="h",
                marker_color=emergence["growth"].apply(lambda x: COLORS["positive"] if x > 0 else COLORS["negative"]),
                text=emergence["growth"].apply(lambda x: f"{x:+.0f}%"),
                textposition="outside",
            ))
            fig_em.update_layout(
                height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Hashtag Growth Rate",
                margin=dict(l=200, r=80),
            )
            st.plotly_chart(fig_em, use_container_width=True)

    with tab4:
        st.markdown("**Period Comparison** - Compare metrics across time periods")
        col1, col2 = st.columns(2)
        with col1:
            period1_start = st.date_input("Period 1 Start", posts["date"].min(), key="p1s")
            period1_end = st.date_input("Period 1 End", posts["date"].min() + timedelta(days=30), key="p1e")
        with col2:
            period2_start = st.date_input("Period 2 Start", posts["date"].max() - timedelta(days=30), key="p2s")
            period2_end = st.date_input("Period 2 End", posts["date"].max(), key="p2e")

        p1 = posts[(posts["date"] >= period1_start) & (posts["date"] <= period1_end)]
        p2 = posts[(posts["date"] >= period2_start) & (posts["date"] <= period2_end)]

        metrics = {
            "Total Posts": (len(p1), len(p2)),
            "Avg Sentiment": (p1["sentiment_score"].mean(), p2["sentiment_score"].mean()),
            "Total Engagement": (p1["engagement"].sum(), p2["engagement"].sum()),
            "Unique Accounts": (p1["username"].nunique(), p2["username"].nunique()),
            "Avg Reach": (p1["reach"].mean(), p2["reach"].mean()),
        }

        comp_df = pd.DataFrame([
            {"Metric": k, "Period 1": v[0], "Period 2": v[1],
             "Change": ((v[1] - v[0]) / max(abs(v[0]), 1)) * 100}
            for k, v in metrics.items()
        ])
        st.dataframe(comp_df.style.format({
            "Period 1": "{:.0f}", "Period 2": "{:.0f}", "Change": "{:+.1f}%"
        }), use_container_width=True)

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Period 1", x=comp_df["Metric"], y=comp_df["Period 1"], marker_color=COLORS["primary"]))
        fig_comp.add_trace(go.Bar(name="Period 2", x=comp_df["Metric"], y=comp_df["Period 2"], marker_color=COLORS["secondary"]))
        fig_comp.update_layout(
            barmode="group", height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_comp, use_container_width=True)


def render_narrative_analysis(posts, narratives):
    """Narrative Analysis & Clustering - Functions 23-30"""
    st.markdown('<div class="section-header">Narrative Analysis & Clustering</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Active Narratives", "Narrative Evolution", "Post Clustering", "Signal vs Noise"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Active Narratives", str(len(narratives))), unsafe_allow_html=True)
        with c2:
            emerging = (narratives["evolution_phase"] == "ناشئة").sum()
            st.markdown(metric_card("Emerging", str(emerging)), unsafe_allow_html=True)
        with c3:
            peak = (narratives["evolution_phase"] == "في الذروة").sum()
            st.markdown(metric_card("At Peak", str(peak)), unsafe_allow_html=True)
        with c4:
            high_risk = narratives["risk_level"].isin(["عالي", "حرج"]).sum()
            st.markdown(metric_card("High Risk", str(high_risk), delta_type="negative"), unsafe_allow_html=True)

        for _, narr in narratives.iterrows():
            risk_class = {"منخفض": "low", "متوسط": "medium", "عالي": "high", "حرج": "critical"}.get(narr["risk_level"], "medium")
            st.markdown(f"""
            <div class="narrative-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <strong style="font-size: 1.1rem; color: #e2e8f0;">{narr['theme']}</strong>
                    <span class="risk-badge risk-{risk_class}">{narr['risk_level']}</span>
                </div>
                <div style="margin-top: 8px; display: flex; gap: 20px; color: #94a3b8; font-size: 0.85rem;">
                    <span>Phase: {narr['evolution_phase']}</span>
                    <span>Posts: {narr['post_count']}</span>
                    <span>Impact: {narr['impact_score']:.0f}</span>
                    <span>Velocity: {narr['spread_velocity']:.0f}</span>
                    <span>Sentiment: {narr['avg_sentiment']:.2f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        selected_narrative = st.selectbox("Select Narrative", narratives["theme"].tolist())
        narr_posts = posts[posts["narrative"] == selected_narrative]

        if len(narr_posts) > 0:
            daily_narr = narr_posts.groupby("date").agg(
                count=("post_id", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                total_engagement=("engagement", "sum"),
            ).reset_index()

            fig_ev = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ev.add_trace(go.Bar(
                x=daily_narr["date"], y=daily_narr["count"],
                name="Post Volume", marker_color="rgba(99,102,241,0.5)"
            ), secondary_y=False)
            fig_ev.add_trace(go.Scatter(
                x=daily_narr["date"], y=daily_narr["avg_sentiment"],
                name="Sentiment Trend", line=dict(color=COLORS["success"], width=2)
            ), secondary_y=True)
            fig_ev.update_layout(
                height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title=f"Narrative Evolution: {selected_narrative}",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_ev, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                sent_dist = narr_posts["sentiment"].value_counts()
                fig_ns = go.Figure(go.Pie(
                    labels=sent_dist.index, values=sent_dist.values,
                    hole=0.5, marker_colors=[COLORS["positive"], COLORS["negative"], COLORS["neutral"]],
                ))
                fig_ns.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Narrative Sentiment")
                st.plotly_chart(fig_ns, use_container_width=True)

            with col_b:
                stance_dist = narr_posts["stance"].value_counts()
                fig_nst = go.Figure(go.Pie(
                    labels=stance_dist.index, values=stance_dist.values,
                    hole=0.5,
                ))
                fig_nst.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Narrative Stance")
                st.plotly_chart(fig_nst, use_container_width=True)

    with tab3:
        st.markdown("**Post Clustering** - Grouping similar posts into thematic clusters")
        cluster_data = posts.groupby("topic").agg(
            count=("post_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            avg_engagement=("engagement", "mean"),
            avg_impact=("impact_score", "mean"),
        ).reset_index()

        fig_cluster = go.Figure(go.Scatter(
            x=cluster_data["avg_sentiment"],
            y=cluster_data["avg_engagement"],
            mode="markers+text",
            marker=dict(
                size=cluster_data["count"] / cluster_data["count"].max() * 50 + 10,
                color=cluster_data["avg_impact"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Impact"),
            ),
            text=cluster_data["topic"],
            textposition="top center",
            textfont=dict(size=9),
        ))
        fig_cluster.update_layout(
            height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Topic Clusters (Size = Volume, Color = Impact)",
            xaxis_title="Average Sentiment", yaxis_title="Average Engagement",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tab4:
        st.markdown("**Signal vs Noise Detection** - Separating meaningful content from noise")
        posts_copy = posts.copy()
        signal_threshold = st.slider("Signal Threshold (Impact Score)", 0, 100, 50, key="signal_thresh")
        posts_copy["classification"] = posts_copy["impact_score"].apply(
            lambda x: "Signal" if x >= signal_threshold else "Noise"
        )

        signal_count = (posts_copy["classification"] == "Signal").sum()
        noise_count = (posts_copy["classification"] == "Noise").sum()
        ratio = signal_count / max(noise_count, 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Signal Posts", format_number(signal_count)), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Noise Posts", format_number(noise_count)), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Signal/Noise Ratio", f"{ratio:.2f}"), unsafe_allow_html=True)

        daily_sn = posts_copy.groupby(["date", "classification"]).size().unstack(fill_value=0).reset_index()
        fig_sn = go.Figure()
        if "Signal" in daily_sn.columns:
            fig_sn.add_trace(go.Scatter(x=daily_sn["date"], y=daily_sn["Signal"], name="Signal", fill="tozeroy",
                                        fillcolor="rgba(99,102,241,0.3)", line=dict(color=COLORS["primary"])))
        if "Noise" in daily_sn.columns:
            fig_sn.add_trace(go.Scatter(x=daily_sn["date"], y=daily_sn["Noise"], name="Noise", fill="tozeroy",
                                        fillcolor="rgba(107,114,128,0.2)", line=dict(color=COLORS["neutral"])))
        fig_sn.update_layout(
            height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Signal vs Noise Over Time", margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_sn, use_container_width=True)


def render_network_analysis(posts, accounts, network):
    """Network Analysis & Influencer Detection - Functions 31-36"""
    st.markdown('<div class="section-header">Network Analysis & Influencer Detection</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Influencer Ranking", "Network Graph", "Engagement Analysis", "Content Spread"])

    with tab1:
        st.markdown("**Influencer Detection** - Identifying and ranking influential accounts")
        top_influencers = accounts.sort_values("influence_score", ascending=False).head(20)

        fig_inf = go.Figure(go.Bar(
            x=top_influencers["influence_score"],
            y=top_influencers["username"],
            orientation="h",
            marker=dict(
                color=top_influencers["influence_score"],
                colorscale="Viridis",
            ),
            text=top_influencers.apply(lambda x: f"Followers: {format_number(x['followers'])} | {x['platform']}", axis=1),
            textposition="inside",
        ))
        fig_inf.update_layout(
            height=600, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Top 20 Influencers by Influence Score",
            margin=dict(l=100, r=20), yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_inf, use_container_width=True)

        st.dataframe(
            top_influencers[["username", "display_name", "platform", "region", "followers",
                           "following", "influence_score", "is_verified", "account_type", "activity_level"]],
            use_container_width=True, height=400,
        )

    with tab2:
        st.markdown("**Network Graph** - Relationship mapping between accounts")
        G = nx.Graph()
        sample_edges = network.head(150)
        for _, edge in sample_edges.iterrows():
            G.add_edge(edge["source_name"], edge["target_name"], weight=edge["weight"],
                      interaction=edge["interaction_type"])

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.5, color="#475569"),
            hoverinfo="none",
        )

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_degrees = [G.degree(n) for n in G.nodes()]

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(
                size=[max(8, d * 3) for d in node_degrees],
                color=node_degrees,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Connections"),
            ),
            text=list(G.nodes()),
            textposition="top center",
            textfont=dict(size=8),
            hovertext=[f"{n}: {G.degree(n)} connections" for n in G.nodes()],
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            height=600, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Account Interaction Network",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(metric_card("Total Nodes", str(G.number_of_nodes())), unsafe_allow_html=True)
        with col_b:
            st.markdown(metric_card("Total Edges", str(G.number_of_edges())), unsafe_allow_html=True)

    with tab3:
        st.markdown("**Engagement Analysis** - Understanding interaction patterns")
        eng_by_acct = posts.groupby("username").agg(
            total_likes=("likes", "sum"),
            total_retweets=("retweets", "sum"),
            total_replies=("replies", "sum"),
            total_engagement=("engagement", "sum"),
            post_count=("post_id", "count"),
        ).reset_index()
        eng_by_acct["eng_per_post"] = eng_by_acct["total_engagement"] / eng_by_acct["post_count"]
        eng_by_acct = eng_by_acct.sort_values("total_engagement", ascending=False).head(20)

        fig_eng = go.Figure()
        fig_eng.add_trace(go.Bar(name="Likes", x=eng_by_acct["username"], y=eng_by_acct["total_likes"], marker_color=COLORS["positive"]))
        fig_eng.add_trace(go.Bar(name="Retweets", x=eng_by_acct["username"], y=eng_by_acct["total_retweets"], marker_color=COLORS["primary"]))
        fig_eng.add_trace(go.Bar(name="Replies", x=eng_by_acct["username"], y=eng_by_acct["total_replies"], marker_color=COLORS["warning"]))
        fig_eng.update_layout(
            barmode="stack", height=400, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", title="Engagement Breakdown by Account",
            margin=dict(t=40, b=80), xaxis_tickangle=45,
        )
        st.plotly_chart(fig_eng, use_container_width=True)

    with tab4:
        st.markdown("**Content Spread Analysis** - How content propagates across accounts")
        interaction_types = network["interaction_type"].value_counts()
        fig_it = go.Figure(go.Pie(
            labels=interaction_types.index, values=interaction_types.values,
            hole=0.5, marker_colors=[COLORS["primary"], COLORS["secondary"], COLORS["success"], COLORS["warning"]],
        ))
        fig_it.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Interaction Types")
        st.plotly_chart(fig_it, use_container_width=True)

        spread_by_platform = posts.groupby("platform").agg(
            avg_reach=("reach", "mean"),
            avg_virality=("virality_score", "mean"),
            total_retweets=("retweets", "sum"),
        ).reset_index()

        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=spread_by_platform["avg_reach"],
            y=spread_by_platform["avg_virality"],
            mode="markers+text",
            marker=dict(size=spread_by_platform["total_retweets"] / spread_by_platform["total_retweets"].max() * 50 + 15,
                       color=COLORS["primary"]),
            text=spread_by_platform["platform"],
            textposition="top center",
        ))
        fig_sp.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Platform Spread (Size = Reshares)",
            xaxis_title="Average Reach", yaxis_title="Average Virality",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_sp, use_container_width=True)


def render_alerts(alerts, posts):
    """Alert & Early Warning System - Functions 18-22, 41"""
    st.markdown('<div class="section-header">Alert & Early Warning System</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    new_alerts = alerts[alerts["status"] == "جديد"]
    with c1:
        st.markdown(metric_card("Total Alerts", str(len(alerts))), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("New Alerts", str(len(new_alerts)), delta_type="negative" if len(new_alerts) > 5 else "neutral"), unsafe_allow_html=True)
    with c3:
        critical = alerts[alerts["severity"] == "حرج"]
        st.markdown(metric_card("Critical", str(len(critical)), delta_type="negative"), unsafe_allow_html=True)
    with c4:
        resolved = alerts[alerts["status"] == "تمت المعالجة"]
        st.markdown(metric_card("Resolved", str(len(resolved)), delta_type="positive"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Alert Feed", "Alert Analytics", "Alert Configuration"])

    with tab1:
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            severity_filter = st.multiselect("Severity", ["حرج", "عالي", "متوسط", "منخفض"], key="alert_sev")
        with col_f2:
            status_filter = st.multiselect("Status", ["جديد", "قيد المراجعة", "تمت المعالجة"], key="alert_stat")
        with col_f3:
            type_filter = st.multiselect("Type", alerts["type"].unique().tolist(), key="alert_type")

        filtered_alerts = alerts.copy()
        if severity_filter:
            filtered_alerts = filtered_alerts[filtered_alerts["severity"].isin(severity_filter)]
        if status_filter:
            filtered_alerts = filtered_alerts[filtered_alerts["status"].isin(status_filter)]
        if type_filter:
            filtered_alerts = filtered_alerts[filtered_alerts["type"].isin(type_filter)]

        for _, alert in filtered_alerts.iterrows():
            css_class = severity_to_css(alert["severity"])
            status_emoji = {"جديد": "[NEW]", "قيد المراجعة": "[REVIEWING]", "تمت المعالجة": "[RESOLVED]"}.get(alert["status"], "")
            st.markdown(f"""
            <div class="alert-card alert-{css_class}">
                <div style="display: flex; justify-content: space-between;">
                    <strong>{alert['type']}</strong>
                    <span style="color: #94a3b8;">{status_emoji} {alert['status']}</span>
                </div>
                <div style="margin-top: 5px; color: #94a3b8; font-size: 0.85rem;">
                    {alert['description']}<br/>
                    Topic: {alert['topic']} | Severity: {alert['severity']} | Assigned: {alert['assigned_to']}
                    <br/>Time: {alert['timestamp'].strftime('%Y-%m-%d %H:%M')}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            sev_counts = alerts["severity"].value_counts()
            fig_sev = go.Figure(go.Pie(
                labels=sev_counts.index, values=sev_counts.values,
                hole=0.5,
                marker_colors=[SEVERITY_COLORS.get(s, "#6b7280") for s in sev_counts.index],
            ))
            fig_sev.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", title="Alerts by Severity")
            st.plotly_chart(fig_sev, use_container_width=True)

        with col_b:
            type_counts = alerts["type"].value_counts()
            fig_type = go.Figure(go.Bar(
                x=type_counts.values, y=type_counts.index,
                orientation="h", marker_color=COLORS["primary"],
            ))
            fig_type.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                                  title="Alerts by Type", margin=dict(l=200))
            st.plotly_chart(fig_type, use_container_width=True)

        alerts["alert_date"] = pd.to_datetime(alerts["timestamp"]).dt.date
        daily_alerts = alerts.groupby("alert_date").size().reset_index(name="count")
        fig_da = go.Figure(go.Scatter(
            x=daily_alerts["alert_date"], y=daily_alerts["count"],
            fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
            line=dict(color=COLORS["danger"]),
        ))
        fig_da.update_layout(
            height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Alert Frequency Over Time", margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_da, use_container_width=True)

    with tab3:
        st.markdown("**Alert Configuration** - Configure alert rules and notification channels")

        with st.expander("Volume Spike Alerts", expanded=True):
            st.slider("Threshold: % increase over baseline", 50, 500, 200, key="vol_thresh")
            st.number_input("Minimum posts in window", 10, 1000, 50, key="vol_min")
            st.selectbox("Time window", ["1 hour", "6 hours", "24 hours"], key="vol_window")

        with st.expander("Sentiment Shift Alerts"):
            st.slider("Threshold: Sentiment change magnitude", 0.1, 1.0, 0.3, key="sent_thresh")
            st.selectbox("Monitoring period", ["6 hours", "12 hours", "24 hours", "48 hours"], key="sent_period")

        with st.expander("Notification Channels"):
            st.checkbox("Email Notifications", value=True, key="notif_email")
            st.text_input("Email Address", placeholder="analyst@domain.com", key="notif_email_addr")
            st.checkbox("Telegram Notifications", value=True, key="notif_tg")
            st.text_input("Telegram Bot Token", type="password", key="notif_tg_token")
            st.text_input("Telegram Chat ID", key="notif_tg_chat")
            st.checkbox("SMS Notifications", key="notif_sms")
            st.checkbox("Dashboard Push Notifications", value=True, key="notif_push")


def render_security(posts, accounts, network):
    """Security & Advanced Detection - Functions 37-41"""
    st.markdown('<div class="section-header">Security & Advanced Detection</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Bot Detection", "Coordinated Campaigns", "Synchronized Activity", "Actor Tracking"])

    with tab1:
        st.markdown("**Bot Detection** - Identifying suspected automated accounts")
        bots = accounts[accounts["is_bot_suspect"]]
        humans = accounts[~accounts["is_bot_suspect"]]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Total Accounts", str(len(accounts))), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Bot Suspects", str(len(bots)), delta_type="negative"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Detection Rate", f"{len(bots)/len(accounts)*100:.1f}%"), unsafe_allow_html=True)
        with c4:
            bot_posts = posts[posts["is_bot_content"]].shape[0]
            st.markdown(metric_card("Bot Posts", format_number(bot_posts)), unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_bot = go.Figure()
            fig_bot.add_trace(go.Scatter(
                x=humans["followers"], y=humans["following"],
                mode="markers", name="Human Accounts",
                marker=dict(color=COLORS["positive"], size=5, opacity=0.5),
            ))
            fig_bot.add_trace(go.Scatter(
                x=bots["followers"], y=bots["following"],
                mode="markers", name="Bot Suspects",
                marker=dict(color=COLORS["danger"], size=8, symbol="x"),
            ))
            fig_bot.update_layout(
                height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Followers vs Following (Bot Detection)",
                xaxis_title="Followers", yaxis_title="Following",
                xaxis_type="log", yaxis_type="log",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_bot, use_container_width=True)

        with col_b:
            bot_by_platform = bots["platform"].value_counts()
            fig_bp = go.Figure(go.Bar(
                x=bot_by_platform.index, y=bot_by_platform.values,
                marker_color=COLORS["danger"],
            ))
            fig_bp.update_layout(
                height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Bot Suspects by Platform",
                margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_bp, use_container_width=True)

        if len(bots) > 0:
            st.dataframe(
                bots[["username", "display_name", "platform", "region", "followers",
                     "following", "influence_score", "created_date", "activity_level"]],
                use_container_width=True,
            )

    with tab2:
        st.markdown("**Coordinated Campaign Detection** - Identifying organized inauthentic behavior")
        coord_posts = posts[posts["is_coordinated"]]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Coordinated Posts", str(len(coord_posts))), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("% of Total", f"{len(coord_posts)/len(posts)*100:.1f}%"), unsafe_allow_html=True)
        with c3:
            coord_accounts = coord_posts["username"].nunique()
            st.markdown(metric_card("Involved Accounts", str(coord_accounts)), unsafe_allow_html=True)

        if len(coord_posts) > 0:
            coord_daily = coord_posts.groupby("date").size().reset_index(name="count")
            fig_cd = go.Figure(go.Scatter(
                x=coord_daily["date"], y=coord_daily["count"],
                fill="tozeroy", fillcolor="rgba(239,68,68,0.2)",
                line=dict(color=COLORS["danger"]),
            ))
            fig_cd.update_layout(
                height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Coordinated Activity Over Time", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_cd, use_container_width=True)

            coord_topics = coord_posts["topic"].value_counts().head(10)
            fig_ct = go.Figure(go.Bar(
                x=coord_topics.values, y=coord_topics.index,
                orientation="h", marker_color=COLORS["warning"],
            ))
            fig_ct.update_layout(
                height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Targeted Topics in Coordinated Campaigns",
                margin=dict(l=200, t=40),
            )
            st.plotly_chart(fig_ct, use_container_width=True)

    with tab3:
        st.markdown("**Synchronized Publishing Analysis** - Detecting timing patterns")
        posts_copy = posts.copy()
        posts_copy["minute_bucket"] = pd.to_datetime(posts_copy["timestamp"]).dt.floor("5min")
        sync_groups = posts_copy.groupby("minute_bucket").size().reset_index(name="count")
        sync_groups["is_suspicious"] = sync_groups["count"] > sync_groups["count"].quantile(0.95)

        fig_sync = go.Figure()
        normal = sync_groups[~sync_groups["is_suspicious"]]
        suspicious = sync_groups[sync_groups["is_suspicious"]]
        fig_sync.add_trace(go.Scatter(
            x=normal["minute_bucket"], y=normal["count"],
            mode="markers", name="Normal",
            marker=dict(color=COLORS["primary"], size=3, opacity=0.5),
        ))
        fig_sync.add_trace(go.Scatter(
            x=suspicious["minute_bucket"], y=suspicious["count"],
            mode="markers", name="Suspicious Spikes",
            marker=dict(color=COLORS["danger"], size=8, symbol="diamond"),
        ))
        fig_sync.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Publishing Synchronization Analysis (5-min buckets)",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_sync, use_container_width=True)

    with tab4:
        st.markdown("**Actor Tracking** - Monitoring key actors and their activity patterns")
        actor_activity = posts.groupby("username").agg(
            post_count=("post_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            platforms=("platform", "nunique"),
            first_post=("timestamp", "min"),
            last_post=("timestamp", "max"),
            avg_credibility=("credibility_score", "mean"),
        ).reset_index()
        actor_activity["activity_span_days"] = (
            actor_activity["last_post"] - actor_activity["first_post"]
        ).dt.days
        actor_activity["posts_per_day"] = actor_activity["post_count"] / actor_activity["activity_span_days"].replace(0, 1)
        actor_activity = actor_activity.sort_values("post_count", ascending=False).head(30)

        fig_actor = go.Figure(go.Scatter(
            x=actor_activity["posts_per_day"],
            y=actor_activity["avg_credibility"],
            mode="markers+text",
            marker=dict(
                size=actor_activity["post_count"] / actor_activity["post_count"].max() * 40 + 8,
                color=actor_activity["avg_sentiment"],
                colorscale="RdYlGn", cmin=-1, cmax=1,
                showscale=True, colorbar=dict(title="Sentiment"),
            ),
            text=actor_activity["username"],
            textposition="top center",
            textfont=dict(size=8),
        ))
        fig_actor.update_layout(
            height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Actor Map (X=Activity Rate, Y=Credibility, Size=Volume, Color=Sentiment)",
            xaxis_title="Posts per Day", yaxis_title="Credibility Score",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_actor, use_container_width=True)


def render_predictive(posts):
    """Predictive Analysis - Functions 42-45"""
    st.markdown('<div class="section-header">Predictive Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Topic Spread Prediction", "Trend Persistence", "Emerging Trends", "Topic Revival"])

    with tab1:
        st.markdown("**Topic Spread Prediction** - Forecasting topic propagation")
        daily = posts.groupby("date").agg(
            count=("post_id", "count"),
            engagement=("engagement", "sum"),
        ).reset_index()
        daily["count_ma"] = daily["count"].rolling(7, min_periods=1).mean()

        last_val = daily["count_ma"].iloc[-1]
        growth_rate = (daily["count_ma"].iloc[-1] - daily["count_ma"].iloc[-7]) / max(daily["count_ma"].iloc[-7], 1)

        future_dates = pd.date_range(daily["date"].max() + timedelta(days=1), periods=14)
        predictions = []
        current = last_val
        for i, d in enumerate(future_dates):
            decay = max(0.5, 1 - i * 0.03)
            noise = np.random.normal(0, current * 0.1)
            current = current * (1 + growth_rate * decay) + noise
            current = max(10, current)
            predictions.append({"date": d, "predicted": current,
                              "lower": current * 0.7, "upper": current * 1.3})
        pred_df = pd.DataFrame(predictions)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=daily["date"], y=daily["count"],
            name="Historical", line=dict(color=COLORS["primary"]),
        ))
        fig_pred.add_trace(go.Scatter(
            x=daily["date"], y=daily["count_ma"],
            name="7-Day MA", line=dict(color=COLORS["secondary"], dash="dot"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=pred_df["date"], y=pred_df["predicted"],
            name="Prediction", line=dict(color=COLORS["warning"], dash="dash"),
        ))
        fig_pred.add_trace(go.Scatter(
            x=pd.concat([pred_df["date"], pred_df["date"][::-1]]),
            y=pd.concat([pred_df["upper"], pred_df["lower"][::-1]]),
            fill="toself", fillcolor="rgba(245,158,11,0.1)",
            line=dict(width=0), name="Confidence Interval",
        ))
        fig_pred.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Post Volume Prediction (14-day forecast)",
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        st.markdown("**Trend Persistence Prediction** - Estimating how long trends will last")
        topic_trends = []
        for topic in ARABIC_TOPICS:
            t_posts = posts[posts["topic"] == topic]
            if len(t_posts) < 5:
                continue
            daily_t = t_posts.groupby("date").size()
            peak_day = daily_t.idxmax()
            peak_val = daily_t.max()
            last_7 = daily_t.tail(7).mean()
            momentum = last_7 / max(peak_val, 1)
            persistence = min(100, momentum * 100 + np.random.normal(0, 10))
            topic_trends.append({
                "topic": topic,
                "peak_volume": peak_val,
                "current_volume": last_7,
                "momentum": momentum,
                "persistence_score": max(0, persistence),
                "predicted_days": max(1, int(persistence / 10)),
            })
        trend_df = pd.DataFrame(topic_trends).sort_values("persistence_score", ascending=False)

        fig_pers = go.Figure(go.Bar(
            x=trend_df["persistence_score"],
            y=trend_df["topic"],
            orientation="h",
            marker=dict(
                color=trend_df["persistence_score"],
                colorscale="RdYlGn",
            ),
            text=trend_df["predicted_days"].apply(lambda x: f"~{x} days"),
            textposition="outside",
        ))
        fig_pers.update_layout(
            height=600, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Trend Persistence Score",
            xaxis_title="Persistence Score (0-100)",
            margin=dict(l=200, r=80),
        )
        st.plotly_chart(fig_pers, use_container_width=True)

    with tab3:
        st.markdown("**Emerging Trend Detection** - Topics likely to become trending")
        recent_3d = posts[posts["date"] >= posts["date"].max() - timedelta(days=3)]
        prev_3d = posts[(posts["date"] < posts["date"].max() - timedelta(days=3)) &
                       (posts["date"] >= posts["date"].max() - timedelta(days=6))]

        recent_topics = recent_3d["topic"].value_counts()
        prev_topics = prev_3d["topic"].value_counts()

        emerging = pd.DataFrame({"recent": recent_topics, "previous": prev_topics}).fillna(0)
        emerging["growth_rate"] = (emerging["recent"] - emerging["previous"]) / emerging["previous"].replace(0, 1)
        emerging["trend_probability"] = (emerging["growth_rate"].clip(0, 5) / 5 * 100).astype(int)
        emerging = emerging.sort_values("growth_rate", ascending=False)

        fig_em = go.Figure(go.Bar(
            x=emerging["trend_probability"],
            y=emerging.index,
            orientation="h",
            marker=dict(color=emerging["trend_probability"], colorscale="Viridis"),
            text=emerging["trend_probability"].apply(lambda x: f"{x}%"),
            textposition="outside",
        ))
        fig_em.update_layout(
            height=600, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Trend Probability Score",
            xaxis_title="Probability (%)",
            margin=dict(l=200, r=80),
        )
        st.plotly_chart(fig_em, use_container_width=True)

    with tab4:
        st.markdown("**Topic Revival Detection** - Identifying resurgence of previously inactive topics")
        topic_lifecycles = []
        for topic in ARABIC_TOPICS:
            t_posts = posts[posts["topic"] == topic].sort_values("date")
            if len(t_posts) < 5:
                continue
            daily_t = t_posts.groupby("date").size().reset_index(name="count")
            daily_t["ma7"] = daily_t["count"].rolling(7, min_periods=1).mean()

            if len(daily_t) > 14:
                mid = len(daily_t) // 2
                first_half_avg = daily_t["ma7"].iloc[:mid].mean()
                second_half_avg = daily_t["ma7"].iloc[mid:].mean()
                is_revival = second_half_avg > first_half_avg * 1.5 and first_half_avg < daily_t["ma7"].median()
            else:
                is_revival = False

            topic_lifecycles.append({
                "topic": topic,
                "is_revival": is_revival,
                "total_posts": len(t_posts),
                "recent_activity": daily_t["count"].tail(7).sum(),
            })

        lc_df = pd.DataFrame(topic_lifecycles)
        revivals = lc_df[lc_df["is_revival"]]

        if len(revivals) > 0:
            st.success(f"Detected {len(revivals)} topic(s) showing revival patterns")
            st.dataframe(revivals, use_container_width=True)
        else:
            st.info("No significant topic revivals detected in the current period")

        for topic in ARABIC_TOPICS[:5]:
            t_posts = posts[posts["topic"] == topic]
            daily_t = t_posts.groupby("date").size().reset_index(name="count")
            daily_t["ma7"] = daily_t["count"].rolling(7, min_periods=1).mean()
            fig_lc = go.Figure(go.Scatter(
                x=daily_t["date"], y=daily_t["ma7"],
                fill="tozeroy", name=topic,
            ))
            fig_lc.update_layout(
                height=150, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title=topic, margin=dict(t=30, b=20, l=50, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_lc, use_container_width=True)


def render_brand_monitoring(posts):
    """Brand Monitoring & Commercial - Functions 46-50"""
    st.markdown('<div class="section-header">Brand Monitoring & Commercial Intelligence</div>', unsafe_allow_html=True)

    brands = st.multiselect(
        "Select Topics/Brands to Monitor",
        ARABIC_TOPICS,
        default=ARABIC_TOPICS[:3],
        key="brand_select"
    )

    if not brands:
        st.warning("Please select at least one topic/brand to monitor.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Brand Health", "Campaign Analysis", "Competitor Comparison", "Audience Satisfaction"])

    with tab1:
        for brand in brands:
            brand_posts = posts[posts["topic"] == brand]
            if len(brand_posts) == 0:
                continue

            st.markdown(f"### {brand}")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(metric_card("Mentions", format_number(len(brand_posts))), unsafe_allow_html=True)
            with c2:
                avg_sent = brand_posts["sentiment_score"].mean()
                st.markdown(metric_card("Sentiment", f"{avg_sent:.2f}",
                                       delta_type="positive" if avg_sent > 0 else "negative"), unsafe_allow_html=True)
            with c3:
                st.markdown(metric_card("Reach", format_number(brand_posts["reach"].sum())), unsafe_allow_html=True)
            with c4:
                positive_ratio = (brand_posts["sentiment"] == "إيجابي").mean() * 100
                st.markdown(metric_card("Positive %", f"{positive_ratio:.0f}%"), unsafe_allow_html=True)

            daily_brand = brand_posts.groupby("date").agg(
                mentions=("post_id", "count"),
                sentiment=("sentiment_score", "mean"),
            ).reset_index()

            fig_brand = make_subplots(specs=[[{"secondary_y": True}]])
            fig_brand.add_trace(go.Bar(
                x=daily_brand["date"], y=daily_brand["mentions"],
                name="Mentions", marker_color="rgba(99,102,241,0.4)"
            ), secondary_y=False)
            fig_brand.add_trace(go.Scatter(
                x=daily_brand["date"], y=daily_brand["sentiment"],
                name="Sentiment", line=dict(color=COLORS["success"])
            ), secondary_y=True)
            fig_brand.update_layout(
                height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_brand, use_container_width=True)
            st.markdown("---")

    with tab2:
        st.markdown("**Campaign Performance Analysis**")
        campaign_metrics = []
        for brand in brands:
            brand_posts = posts[posts["topic"] == brand]
            campaign_metrics.append({
                "Brand/Topic": brand,
                "Total Mentions": len(brand_posts),
                "Total Engagement": brand_posts["engagement"].sum(),
                "Avg Reach": brand_posts["reach"].mean(),
                "Positive %": (brand_posts["sentiment"] == "إيجابي").mean() * 100,
                "Virality Score": brand_posts["virality_score"].mean(),
            })
        camp_df = pd.DataFrame(campaign_metrics)
        st.dataframe(camp_df.style.format({
            "Total Engagement": "{:,.0f}",
            "Avg Reach": "{:,.0f}",
            "Positive %": "{:.1f}%",
            "Virality Score": "{:.1f}",
        }), use_container_width=True)

        fig_camp = go.Figure()
        fig_camp.add_trace(go.Scatterpolar(
            r=[camp_df["Total Mentions"].iloc[0] / camp_df["Total Mentions"].max() * 100 if len(camp_df) > 0 else 0,
               camp_df["Total Engagement"].iloc[0] / camp_df["Total Engagement"].max() * 100 if len(camp_df) > 0 else 0,
               camp_df["Avg Reach"].iloc[0] / camp_df["Avg Reach"].max() * 100 if len(camp_df) > 0 else 0,
               camp_df["Positive %"].iloc[0] if len(camp_df) > 0 else 0,
               camp_df["Virality Score"].iloc[0] if len(camp_df) > 0 else 0],
            theta=["Mentions", "Engagement", "Reach", "Positive %", "Virality"],
            fill="toself",
            name=brands[0] if brands else "",
        ))
        fig_camp.update_layout(
            height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Campaign Performance Radar",
            polar=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_camp, use_container_width=True)

    with tab3:
        st.markdown("**Competitor Comparison** - Side-by-side analysis")
        if len(brands) >= 2:
            comp_data = []
            for brand in brands:
                brand_posts = posts[posts["topic"] == brand]
                comp_data.append({
                    "brand": brand,
                    "mentions": len(brand_posts),
                    "engagement": brand_posts["engagement"].sum(),
                    "avg_sentiment": brand_posts["sentiment_score"].mean(),
                    "reach": brand_posts["reach"].sum(),
                    "positive_pct": (brand_posts["sentiment"] == "إيجابي").mean() * 100,
                })
            comp_df = pd.DataFrame(comp_data)

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(name="Mentions", x=comp_df["brand"], y=comp_df["mentions"], marker_color=COLORS["primary"]))
            fig_comp.update_layout(
                height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Mention Volume Comparison", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            fig_sent_comp = go.Figure(go.Bar(
                x=comp_df["brand"], y=comp_df["avg_sentiment"],
                marker_color=comp_df["avg_sentiment"].apply(
                    lambda x: COLORS["positive"] if x > 0 else COLORS["negative"]),
            ))
            fig_sent_comp.update_layout(
                height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                title="Sentiment Comparison", margin=dict(t=40, b=30),
            )
            st.plotly_chart(fig_sent_comp, use_container_width=True)
        else:
            st.info("Select at least 2 brands/topics for comparison.")

    with tab4:
        st.markdown("**Audience Satisfaction Analysis**")
        for brand in brands:
            brand_posts = posts[posts["topic"] == brand]
            satisfaction = (brand_posts["sentiment"] == "إيجابي").mean() * 100

            st.markdown(f"#### {brand}")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=satisfaction,
                title={"text": "Satisfaction Score"},
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["primary"]},
                    "steps": [
                        {"range": [0, 30], "color": "rgba(239,68,68,0.3)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.3)"},
                        {"range": [60, 100], "color": "rgba(16,185,129,0.3)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "thickness": 0.75, "value": satisfaction},
                },
            ))
            fig_gauge.update_layout(
                height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)


def render_reports(posts, accounts, alerts, narratives):
    """Reports & Export - Functions 51-55"""
    st.markdown('<div class="section-header">Reports & Export</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Executive Summary", "Custom Reports", "Export Data"])

    with tab1:
        st.markdown("### Executive Summary Report")
        st.markdown("---")

        report_period = st.selectbox("Report Period", ["Last 7 Days", "Last 30 Days", "Last 90 Days", "Custom"], key="rpt_period")

        if report_period == "Last 7 Days":
            cutoff = posts["date"].max() - timedelta(days=7)
        elif report_period == "Last 30 Days":
            cutoff = posts["date"].max() - timedelta(days=30)
        elif report_period == "Last 90 Days":
            cutoff = posts["date"].max() - timedelta(days=90)
        else:
            cutoff = posts["date"].min()

        period_posts = posts[posts["date"] >= cutoff]

        st.markdown(f"""
        **Report Period:** {cutoff} to {posts['date'].max()}

        **Key Metrics:**
        - Total Posts Monitored: **{format_number(len(period_posts))}**
        - Active Accounts: **{period_posts['username'].nunique()}**
        - Total Engagement: **{format_number(period_posts['engagement'].sum())}**
        - Average Sentiment: **{period_posts['sentiment_score'].mean():.2f}**
        - Active Alerts: **{len(alerts[alerts['status'] == 'جديد'])}**

        **Sentiment Breakdown:**
        - Positive: **{(period_posts['sentiment'] == 'إيجابي').mean()*100:.1f}%**
        - Negative: **{(period_posts['sentiment'] == 'سلبي').mean()*100:.1f}%**
        - Neutral: **{(period_posts['sentiment'] == 'محايد').mean()*100:.1f}%**

        **Top Topics:**
        """)

        top_topics = period_posts["topic"].value_counts().head(5)
        for i, (topic, count) in enumerate(top_topics.items(), 1):
            st.markdown(f"{i}. {topic} ({count} posts)")

        st.markdown("**AI Insights:**")
        insights = [
            f"Post volume {'increased' if len(period_posts) > len(posts) * 0.5 else 'decreased'} compared to the previous period.",
            f"Overall sentiment is {'positive' if period_posts['sentiment_score'].mean() > 0 else 'negative'} with a score of {period_posts['sentiment_score'].mean():.2f}.",
            f"The most discussed topic is '{top_topics.index[0]}' with {top_topics.values[0]} posts.",
            f"There are {len(alerts[alerts['severity'] == 'حرج'])} critical alerts requiring immediate attention.",
            f"Bot activity detected in {(accounts['is_bot_suspect'].mean()*100):.1f}% of monitored accounts.",
        ]
        for insight in insights:
            st.markdown(f"- {insight}")

    with tab2:
        st.markdown("### Custom Report Builder")
        report_sections = st.multiselect(
            "Select Report Sections",
            ["Volume Analysis", "Sentiment Analysis", "Trend Analysis", "Narrative Analysis",
             "Network Analysis", "Alert Summary", "Bot Detection", "Brand Monitoring"],
            default=["Volume Analysis", "Sentiment Analysis"],
            key="rpt_sections"
        )

        if "Volume Analysis" in report_sections:
            daily = posts.groupby("date").size().reset_index(name="count")
            fig = go.Figure(go.Scatter(x=daily["date"], y=daily["count"], fill="tozeroy",
                                       fillcolor="rgba(99,102,241,0.2)", line=dict(color=COLORS["primary"])))
            fig.update_layout(height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            title="Post Volume", margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        if "Sentiment Analysis" in report_sections:
            sent = posts["sentiment"].value_counts()
            fig = go.Figure(go.Pie(labels=sent.index, values=sent.values, hole=0.5,
                                  marker_colors=[COLORS["positive"], COLORS["negative"], COLORS["neutral"]]))
            fig.update_layout(height=300, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                            title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Data Export")

        export_type = st.selectbox("Select Data to Export", [
            "Posts Data", "Accounts Data", "Alerts Data", "Narratives Data",
            "Full Report (All Data)"
        ], key="export_type")

        file_format = st.radio("File Format", ["CSV", "JSON"], horizontal=True, key="export_format")

        if st.button("Generate Export", key="btn_export"):
            if export_type == "Posts Data":
                data = posts
            elif export_type == "Accounts Data":
                data = accounts
            elif export_type == "Alerts Data":
                data = alerts
            elif export_type == "Narratives Data":
                data = narratives
            else:
                data = posts

            if file_format == "CSV":
                csv = data.to_csv(index=False)
                st.download_button("Download CSV", csv, f"{export_type.lower().replace(' ', '_')}.csv",
                                  "text/csv", key="dl_csv")
            else:
                json_str = data.to_json(orient="records", date_format="iso", force_ascii=False)
                st.download_button("Download JSON", json_str, f"{export_type.lower().replace(' ', '_')}.json",
                                  "application/json", key="dl_json")


def render_settings():
    """User Management & Settings - Functions 56-60"""
    st.markdown('<div class="section-header">Settings & User Management</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["User Management", "System Settings", "Data Sources", "Case Management"])

    with tab1:
        st.markdown("### User & Permissions Management")

        users = pd.DataFrame([
            {"User": "admin", "Role": "Administrator", "Status": "Active", "Last Login": "2026-05-19 08:30"},
            {"User": "analyst_1", "Role": "Senior Analyst", "Status": "Active", "Last Login": "2026-05-19 09:15"},
            {"User": "analyst_2", "Role": "Analyst", "Status": "Active", "Last Login": "2026-05-18 16:45"},
            {"User": "viewer_1", "Role": "Viewer", "Status": "Active", "Last Login": "2026-05-17 11:00"},
            {"User": "manager_1", "Role": "Manager", "Status": "Inactive", "Last Login": "2026-05-10 14:20"},
        ])
        st.dataframe(users, use_container_width=True)

        with st.expander("Add New User"):
            st.text_input("Username", key="new_username")
            st.text_input("Email", key="new_email")
            st.selectbox("Role", ["Administrator", "Senior Analyst", "Analyst", "Viewer", "Manager"], key="new_role")
            st.button("Create User", key="btn_create_user")

        st.markdown("### Role Permissions")
        permissions = pd.DataFrame({
            "Permission": ["View Dashboard", "Manage Keywords", "Export Data", "Manage Alerts",
                          "Configure System", "Manage Users", "Access Raw Data", "Create Reports"],
            "Administrator": [True] * 8,
            "Senior Analyst": [True, True, True, True, False, False, True, True],
            "Analyst": [True, True, True, False, False, False, False, True],
            "Viewer": [True, False, False, False, False, False, False, False],
            "Manager": [True, True, True, True, False, False, True, True],
        })
        st.dataframe(permissions, use_container_width=True)

    with tab2:
        st.markdown("### System Configuration")

        with st.expander("Data Collection Settings", expanded=True):
            st.slider("Collection Frequency (minutes)", 1, 60, 15, key="collect_freq")
            st.multiselect("Active Platforms", PLATFORMS, default=PLATFORMS, key="active_platforms")
            st.multiselect("Monitored Languages", LANGUAGES, default=LANGUAGES, key="active_langs")
            st.number_input("Max Posts per Collection", 100, 100000, 10000, key="max_posts")

        with st.expander("Analysis Settings"):
            st.slider("Sentiment Threshold (Positive)", 0.0, 1.0, 0.3, key="sent_pos_thresh")
            st.slider("Sentiment Threshold (Negative)", -1.0, 0.0, -0.3, key="sent_neg_thresh")
            st.slider("Bot Detection Sensitivity", 0.0, 1.0, 0.7, key="bot_sensitivity")
            st.slider("Coordination Detection Threshold", 0.0, 1.0, 0.8, key="coord_thresh")

        with st.expander("Display Settings"):
            st.selectbox("Default Language", ["Arabic (RTL)", "English"], key="display_lang")
            st.selectbox("Default Time Zone", ["Asia/Riyadh", "UTC", "Asia/Dubai", "Africa/Cairo"], key="display_tz")
            st.number_input("Dashboard Refresh Interval (seconds)", 10, 300, 60, key="refresh_interval")

    with tab3:
        st.markdown("### Data Source Configuration")

        sources = pd.DataFrame([
            {"Source": "Twitter/X API", "Status": "Connected", "Last Sync": "2026-05-19 09:00", "Records": "1.2M"},
            {"Source": "Facebook Graph API", "Status": "Connected", "Last Sync": "2026-05-19 08:45", "Records": "850K"},
            {"Source": "Instagram API", "Status": "Connected", "Last Sync": "2026-05-19 09:10", "Records": "620K"},
            {"Source": "YouTube Data API", "Status": "Connected", "Last Sync": "2026-05-19 08:30", "Records": "340K"},
            {"Source": "TikTok API", "Status": "Limited", "Last Sync": "2026-05-19 07:00", "Records": "180K"},
            {"Source": "News Aggregator", "Status": "Connected", "Last Sync": "2026-05-19 09:15", "Records": "95K"},
            {"Source": "Reddit API", "Status": "Connected", "Last Sync": "2026-05-19 08:00", "Records": "45K"},
        ])
        st.dataframe(sources, use_container_width=True)

        with st.expander("Add New Data Source"):
            st.selectbox("Source Type", ["Social Media API", "News Feed", "RSS", "Custom Webhook", "File Upload"], key="new_source_type")
            st.text_input("API Endpoint", key="new_source_endpoint")
            st.text_input("API Key", type="password", key="new_source_key")
            st.button("Add Source", key="btn_add_source")

    with tab4:
        st.markdown("### Case Management")

        cases = pd.DataFrame([
            {"Case ID": "CASE-001", "Title": "Coordinated Campaign - Topic A",
             "Status": "Open", "Priority": "High", "Assigned": "analyst_1", "Created": "2026-05-15"},
            {"Case ID": "CASE-002", "Title": "Bot Network Detection",
             "Status": "In Progress", "Priority": "Critical", "Assigned": "analyst_2", "Created": "2026-05-16"},
            {"Case ID": "CASE-003", "Title": "Sentiment Shift Investigation",
             "Status": "Resolved", "Priority": "Medium", "Assigned": "analyst_1", "Created": "2026-05-12"},
            {"Case ID": "CASE-004", "Title": "Emerging Narrative Monitoring",
             "Status": "Open", "Priority": "Low", "Assigned": "analyst_2", "Created": "2026-05-18"},
        ])
        st.dataframe(cases, use_container_width=True)

        with st.expander("Create New Case"):
            st.text_input("Case Title", key="case_title")
            st.text_area("Description", key="case_desc")
            st.selectbox("Priority", ["Critical", "High", "Medium", "Low"], key="case_priority")
            st.selectbox("Assign To", ["analyst_1", "analyst_2", "analyst_3"], key="case_assign")
            st.button("Create Case", key="btn_create_case")

        st.markdown("### Analyst Notes")
        st.text_area("Add Note", placeholder="Enter analysis notes...", key="analyst_note")
        st.button("Save Note", key="btn_save_note")


def render_ai_insights(posts, narratives):
    """AI Recommendations & Intelligence - Functions 61-63"""
    st.markdown('<div class="section-header">AI Intelligence & Recommendations</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["AI Recommendations", "Strategic Suggestions", "Model Performance"])

    with tab1:
        st.markdown("### AI-Generated Recommendations")

        avg_sentiment = posts["sentiment_score"].mean()
        negative_ratio = (posts["sentiment"] == "سلبي").mean()
        bot_ratio = posts["is_bot_content"].mean()
        coord_ratio = posts["is_coordinated"].mean()
        high_risk = narratives[narratives["risk_level"].isin(["عالي", "حرج"])]

        recommendations = []

        if negative_ratio > 0.3:
            recommendations.append({
                "priority": "High",
                "category": "Sentiment",
                "recommendation": "Elevated negative sentiment detected. Consider proactive communication strategies to address public concerns.",
                "action": "Review top negative topics and prepare response narratives.",
            })
        if bot_ratio > 0.05:
            recommendations.append({
                "priority": "Critical",
                "category": "Security",
                "recommendation": f"Bot activity detected at {bot_ratio*100:.1f}%. Investigate and report inauthentic accounts.",
                "action": "Review flagged accounts and coordinate with platform providers.",
            })
        if coord_ratio > 0.03:
            recommendations.append({
                "priority": "High",
                "category": "Security",
                "recommendation": f"Coordinated inauthentic behavior detected ({coord_ratio*100:.1f}% of posts). Possible organized campaign.",
                "action": "Activate campaign response protocol and document evidence.",
            })
        if len(high_risk) > 0:
            recommendations.append({
                "priority": "Medium",
                "category": "Narratives",
                "recommendation": f"{len(high_risk)} high-risk narratives identified. Monitor closely for escalation.",
                "action": "Assign analysts to track narrative evolution and prepare counter-narratives if needed.",
            })

        recommendations.append({
            "priority": "Low",
            "category": "Optimization",
            "recommendation": "Consider expanding monitoring to additional regional platforms for comprehensive coverage.",
            "action": "Evaluate local forum and messaging platform integration options.",
        })
        recommendations.append({
            "priority": "Medium",
            "category": "Communication",
            "recommendation": "Peak engagement hours identified. Optimize official communication timing.",
            "action": "Schedule key announcements during peak activity windows (10AM-2PM, 7PM-10PM).",
        })

        for rec in recommendations:
            priority_color = {"Critical": COLORS["danger"], "High": COLORS["warning"],
                            "Medium": COLORS["info"], "Low": COLORS["success"]}.get(rec["priority"], COLORS["info"])
            st.markdown(f"""
            <div class="narrative-card" style="border-left: 4px solid {priority_color};">
                <div style="display: flex; justify-content: space-between;">
                    <strong style="color: #e2e8f0;">[{rec['category']}] {rec['recommendation']}</strong>
                    <span class="risk-badge" style="background: {priority_color}20; color: {priority_color};">{rec['priority']}</span>
                </div>
                <div style="margin-top: 8px; color: #94a3b8; font-size: 0.85rem;">
                    Suggested Action: {rec['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### Strategic Response Suggestions")
        st.markdown("Based on current data analysis, the following strategic approaches are recommended:")

        strategies = [
            {
                "title": "Proactive Communication Strategy",
                "description": "Deploy targeted messaging on top engagement platforms during peak hours to shape positive narratives.",
                "impact": 85,
            },
            {
                "title": "Crisis Preparedness Plan",
                "description": "Activate monitoring protocols for high-risk narratives with pre-approved response templates.",
                "impact": 92,
            },
            {
                "title": "Counter-Disinformation Framework",
                "description": "Implement systematic fact-checking workflow for flagged content with rapid response capability.",
                "impact": 78,
            },
            {
                "title": "Audience Engagement Optimization",
                "description": "Leverage influencer partnerships and community engagement to amplify positive narratives.",
                "impact": 70,
            },
            {
                "title": "Digital Literacy Initiative",
                "description": "Support public awareness campaigns about misinformation to build resilient information ecosystem.",
                "impact": 65,
            },
        ]

        for strategy in strategies:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="narrative-card">
                    <strong style="color: #e2e8f0;">{strategy['title']}</strong>
                    <p style="color: #94a3b8; margin-top: 5px;">{strategy['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=strategy["impact"],
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": COLORS["primary"]},
                        "steps": [
                            {"range": [0, 50], "color": "rgba(239,68,68,0.2)"},
                            {"range": [50, 75], "color": "rgba(245,158,11,0.2)"},
                            {"range": [75, 100], "color": "rgba(16,185,129,0.2)"},
                        ],
                    },
                ))
                fig_g.update_layout(height=150, margin=dict(t=20, b=10, l=20, r=20),
                                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True)

    with tab3:
        st.markdown("### Model Performance & Learning")

        model_metrics = pd.DataFrame([
            {"Model": "Sentiment Analysis (Arabic)", "Accuracy": 0.87, "F1-Score": 0.85, "Last Updated": "2026-05-15"},
            {"Model": "Stance Detection", "Accuracy": 0.82, "F1-Score": 0.80, "Last Updated": "2026-05-14"},
            {"Model": "Bot Detection", "Accuracy": 0.91, "F1-Score": 0.89, "Last Updated": "2026-05-16"},
            {"Model": "Topic Classification", "Accuracy": 0.88, "F1-Score": 0.86, "Last Updated": "2026-05-15"},
            {"Model": "Narrative Clustering", "Accuracy": 0.79, "F1-Score": 0.77, "Last Updated": "2026-05-13"},
            {"Model": "Coordination Detection", "Accuracy": 0.85, "F1-Score": 0.83, "Last Updated": "2026-05-16"},
        ])

        st.dataframe(model_metrics.style.format({
            "Accuracy": "{:.1%}",
            "F1-Score": "{:.1%}",
        }), use_container_width=True)

        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(name="Accuracy", x=model_metrics["Model"], y=model_metrics["Accuracy"],
                                 marker_color=COLORS["primary"]))
        fig_perf.add_trace(go.Bar(name="F1-Score", x=model_metrics["Model"], y=model_metrics["F1-Score"],
                                 marker_color=COLORS["secondary"]))
        fig_perf.update_layout(
            barmode="group", height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", title="Model Performance Metrics",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            margin=dict(t=40, b=100), xaxis_tickangle=30,
        )
        st.plotly_chart(fig_perf, use_container_width=True)

        st.markdown("**Continuous Learning Status:**")
        learning_data = pd.DataFrame({
            "Date": pd.date_range(end=datetime.now(), periods=30, freq="D"),
            "Accuracy": np.random.uniform(0.82, 0.92, 30).cumsum() / np.arange(1, 31) + 0.5,
        })
        learning_data["Accuracy"] = learning_data["Accuracy"].clip(0, 1)
        fig_learn = go.Figure(go.Scatter(
            x=learning_data["Date"], y=learning_data["Accuracy"],
            fill="tozeroy", fillcolor="rgba(99,102,241,0.2)",
            line=dict(color=COLORS["primary"]),
        ))
        fig_learn.update_layout(
            height=250, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            title="Model Accuracy Trend (30 Days)",
            yaxis=dict(range=[0.7, 1], tickformat=".0%"),
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(fig_learn, use_container_width=True)


# ---------------------------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------------------------
def main():
    inject_css()

    posts, accounts, alerts, narratives, network = load_data()

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #e2e8f0; font-size: 1.2rem; margin: 0;">IntegrateGlobals</h2>
            <p style="color: #94a3b8; font-size: 0.75rem; margin: 0;">Social Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        page = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Keyword & Hashtag Monitoring",
                "Sentiment & Stance Analysis",
                "Trend Detection & Analysis",
                "Narrative Analysis",
                "Network & Influencers",
                "Alert & Early Warning",
                "Security & Detection",
                "Predictive Analysis",
                "Brand Monitoring",
                "Reports & Export",
                "AI Intelligence",
                "Settings & Management",
            ],
            key="nav_page",
        )

        st.markdown("---")

        st.markdown('<div class="section-header" style="font-size: 0.9rem;">Quick Filters</div>', unsafe_allow_html=True)
        date_range = st.date_input(
            "Date Range",
            value=(posts["date"].min(), posts["date"].max()),
            key="global_date",
        )

        selected_platforms = st.multiselect("Platforms", PLATFORMS, key="global_platforms")
        selected_sentiments = st.multiselect("Sentiment", SENTIMENT_LABELS, key="global_sentiments")

        st.markdown("---")

        st.markdown(f"""
        <div style="text-align: center; color: #64748b; font-size: 0.7rem;">
            <p>Data: {posts['date'].min()} to {posts['date'].max()}</p>
            <p>{len(posts):,} posts | {len(accounts)} accounts</p>
            <p>IntegrateGlobals v1.0</p>
        </div>
        """, unsafe_allow_html=True)

    filtered_posts = posts.copy()
    if selected_platforms:
        filtered_posts = filtered_posts[filtered_posts["platform"].isin(selected_platforms)]
    if selected_sentiments:
        filtered_posts = filtered_posts[filtered_posts["sentiment"].isin(selected_sentiments)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered_posts = filtered_posts[
            (filtered_posts["date"] >= date_range[0]) & (filtered_posts["date"] <= date_range[1])
        ]

    if page == "Dashboard":
        render_dashboard(filtered_posts, accounts, alerts, narratives)
    elif page == "Keyword & Hashtag Monitoring":
        render_keyword_monitoring(filtered_posts)
    elif page == "Sentiment & Stance Analysis":
        render_sentiment_analysis(filtered_posts)
    elif page == "Trend Detection & Analysis":
        render_trend_analysis(filtered_posts)
    elif page == "Narrative Analysis":
        render_narrative_analysis(filtered_posts, narratives)
    elif page == "Network & Influencers":
        render_network_analysis(filtered_posts, accounts, network)
    elif page == "Alert & Early Warning":
        render_alerts(alerts, filtered_posts)
    elif page == "Security & Detection":
        render_security(filtered_posts, accounts, network)
    elif page == "Predictive Analysis":
        render_predictive(filtered_posts)
    elif page == "Brand Monitoring":
        render_brand_monitoring(filtered_posts)
    elif page == "Reports & Export":
        render_reports(filtered_posts, accounts, alerts, narratives)
    elif page == "AI Intelligence":
        render_ai_insights(filtered_posts, narratives)
    elif page == "Settings & Management":
        render_settings()

    st.markdown("""
    <div class="footer">
        <p>IntegrateGlobals - Social Intelligence & Digital Discourse Analysis Platform</p>
        <p>Integrate Dynamics | Data Analysis & Decision Making</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
