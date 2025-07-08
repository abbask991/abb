# Streamlit Dashboard Code for Iraqi Oil Forecast Model

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🛢️ Iraqi Oil Risk Dashboard")
st.markdown("**Forecast Based on Global Macro Indicators – Updated Weekly**")

# Sample input data (simulated values for demonstration)
data = {
    "Indicator": [
        "Yield Curve Inversion",
        "Fed Funds Rate Change",
        "US Oil Inventories",
        "China/EU Manufacturing PMI",
        "Market Volatility (VIX)"
    ],
    "Status": [
        "Inverted (8 weeks)",
        "+25bps Hike",
        "+3.1 million barrels",
        "Below 50 (3 months)",
        "VIX Rising to 22.5"
    ],
    "Risk Impact": [
        "High",
        "Moderate",
        "High",
        "Moderate",
        "Mild"
    ],
    "Weight": [0.3, 0.2, 0.2, 0.15, 0.15],
    "Score": [1.0, 1.0, 1.0, 1.0, 0.5]
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

# Risk level interpretation
if risk_score >= 0.9:
    st.error("High Risk of Oil Price Decline – Immediate Policy Action Required")
elif risk_score >= 0.7:
    st.warning("Moderate Risk – Caution in Fiscal Planning")
else:
    st.success("Low Risk – Market Relatively Stable")

# Pie Chart for Weight Distribution
fig, ax = plt.subplots()
ax.pie(df["Weight"], labels=df["Indicator"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Set2.colors)
ax.set_title("Weight of Indicators in Oil Forecast Model")
st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("Model developed for policy simulation – Not for direct trading decisions.")
