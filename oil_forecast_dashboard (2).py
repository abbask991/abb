# Streamlit Dashboard Code for Iraqi Oil Forecast Model

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🛢️ Iraqi Oil Risk Dashboard")
st.markdown("**Forecast Based on Global Macro Indicators – Updated Weekly**")

# Sample input data for Advanced Model (Expanded)
data = {
    "Indicator": [
        "Yield Curve Inversion",
        "Fed Funds Rate Change",
        "US Oil Inventories",
        "China/EU Manufacturing PMI",
        "Market Volatility (VIX)",
        "Shipping Index TD3C",
        "Shipping Index BDTI"
    ],
    "Status": [
        "Inverted (8 weeks)",
        "+25bps Hike",
        "+3.5 million barrels",
        "Below 50",
        "VIX at 25",
        "TD3C Rising",
        "BDTI Stable"
    ],
    "Risk Impact": [
        "High",
        "Moderate",
        "High",
        "Moderate",
        "Mild",
        "High",
        "Moderate"
    ],
    "Weight": [0.25, 0.15, 0.15, 0.10, 0.10, 0.15, 0.10],
    "Score": [1.0, 1.0, 0.9, 0.8, 0.7, 0.9, 0.6]
}

df = pd.DataFrame(data)

# Display table
st.subheader("📊 Weekly Indicator Assessment")
st.dataframe(df, use_container_width=True)

# Compute Risk Score
risk_score = sum(df["Weight"] * df["Score"])

# Gauge-like display for risk level
st.subheader("🔥 Composite Risk Score")
st.metric(label="Overall Risk Level", value=f"{risk_score:.3f}", delta=None)

# Risk level interpretation with policy recommendations
st.subheader("📌 Dynamic Policy Recommendations")
if risk_score >= 0.9:
    st.error("High Risk of Oil Price Decline – Immediate Policy Action Required")
    st.markdown("- Reevaluate budget benchmark oil price below $65\n"
                "- Increase allocations to Financial Stability Buffer\n"
                "- Urgently consult SOMO to implement hedging contracts\n"
                "- Review public spending plans for Q1–Q2 2026")
elif risk_score >= 0.7:
    st.warning("Moderate Risk – Caution in Fiscal Planning")
    st.markdown("- Maintain cautious assumptions for oil revenue\n"
                "- Begin early hedging scenario simulations with SOMO\n"
                "- Delay any expansionary spending beyond critical sectors")
else:
    st.success("Low Risk – Market Relatively Stable")
    st.markdown("- Proceed with current fiscal plan\n"
                "- Monitor inventories and interest rates for reversal\n"
                "- Preserve budgetary surplus in stabilization accounts")

# Pie Chart for Weight Distribution
fig, ax = plt.subplots()
ax.pie(df["Weight"], labels=df["Indicator"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
ax.set_title("Weight of Indicators in Oil Forecast Model")
st.pyplot(fig)

# Time Series Chart of Scores for Illustration
st.subheader("📈 Indicator Score Trend")
trend_data = pd.DataFrame({
    "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
    "Yield Curve Inversion": [0.8, 0.9, 1.0, 1.0],
    "Fed Funds Rate": [0.8, 0.9, 1.0, 1.0],
    "Oil Inventories": [0.6, 0.7, 0.9, 1.0],
    "PMI": [0.5, 0.7, 0.9, 1.0],
    "VIX": [0.4, 0.6, 0.7, 0.5],
    "TD3C": [0.5, 0.7, 0.8, 0.9],
    "BDTI": [0.6, 0.6, 0.6, 0.6]
})

trend_data.set_index("Week", inplace=True)
fig2, ax2 = plt.subplots(figsize=(12, 6))
for column in trend_data.columns:
    ax2.plot(trend_data.index, trend_data[column], label=column)

ax2.set_title("Weekly Score Trends of Indicators")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.2)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Footer
st.markdown("---")
st.caption("Model developed for policy simulation – Not for direct trading decisions.")
