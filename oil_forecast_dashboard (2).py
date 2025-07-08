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
st.subheader(" Weekly Indicator Assessment")
st.dataframe(df, use_container_width=True)

# Compute Risk Score
risk_score = sum(df["Weight"] * df["Score"])

# Gauge-like display for risk level
st.subheader(" Composite Risk Score")
st.metric(label="Overall Risk Level", value=f"{risk_score:.3f}", delta=None)

# Risk level interpretation with policy recommendations
st.subheader(" Dynamic Policy Recommendations")
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
st.subheader(" Indicator Score Trend")
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

# Streamlit Dashboard Code for Iraqi Oil Forecast Model – Updated Version

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
        "Market Volatility (VIX)",
        "Shipping Cost (TD3C)",
        "Tanker Demand Index (BDTI)"
    ],
    "Status": [
        "Inverted (8 weeks)",
        "+25bps Hike",
        "+3.1 million barrels",
        "Below 50 (3 months)",
        "VIX Rising to 22.5",
        "TD3C Up to 45,000",
        "BDTI Declining to 980"
    ],
    "Risk Impact": [
        "High",
        "Moderate",
        "High",
        "Moderate",
        "Mild",
        "Moderate",
        "Mild"
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

# Display Risk Level
st.subheader("🔥 Composite Risk Score")
st.metric(label="Overall Risk Level", value=f"{risk_score:.3f}", delta=None)

# Recommendations
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
ax.pie(df["Weight"], labels=df["Indicator"], autopct='%1.1f%%', startangle=90, colors=plt.cm.Set2.colors)
ax.set_title("Weight of Indicators in Oil Forecast Model")
st.pyplot(fig)

# Time Series Chart of Scores for Illustration
st.subheader("📈 Indicator Score Trend")
trend_data = pd.DataFrame({
    "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
    "Yield Curve": [0.8, 0.9, 1.0, 1.0],
    "Fed Funds Rate": [0.7, 0.8, 1.0, 1.0],
    "EIA Inventories": [0.6, 0.7, 0.9, 1.0],
    "PMI": [0.4, 0.6, 0.8, 1.0],
    "VIX": [0.3, 0.5, 0.7, 0.8],
    "TD3C": [0.5, 0.6, 0.8, 0.9],
    "BDTI": [0.4, 0.5, 0.6, 0.7]
})
trend_data.set_index("Week", inplace=True)

fig2, ax2 = plt.subplots(figsize=(10, 6))
for column in trend_data.columns:
    ax2.plot(trend_data.index, trend_data[column], label=column)

ax2.set_title("Weekly Score Trends of Indicators")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.2)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Composite Risk Score vs Brent Oil Price
st.subheader("📉 Risk Score vs Brent Oil Price")
composite = trend_data.copy()
composite["Composite Risk Score"] = (
    0.25 * composite["Yield Curve"] +
    0.15 * composite["Fed Funds Rate"] +
    0.15 * composite["EIA Inventories"] +
    0.10 * composite["PMI"] +
    0.10 * composite["VIX"] +
    0.15 * composite["TD3C"] +
    0.10 * composite["BDTI"]
)
composite["Brent Oil Price"] = [89, 85, 78, 74]

fig3, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Week")
ax1.set_ylabel("Composite Risk Score", color='tab:red')
ax1.plot(composite.index, composite["Composite Risk Score"], color='tab:red', marker='o', label="Risk Score")
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
ax2.set_ylabel("Brent Oil Price ($)", color='tab:blue')
ax2.plot(composite.index, composite["Brent Oil Price"], color='tab:blue', marker='s', linestyle='--', label="Brent Price")
ax2.tick_params(axis='y', labelcolor='tab:blue')
ax2.set_ylim(65, 95)

fig3.tight_layout()
st.pyplot(fig3)

# Footer
st.markdown("---")
st.caption("Model developed for fiscal policy and oil planning – not investment advice.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# Data Preparation
# ===============================
# Simulated weekly indicator scores
weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
# افترض أن عندك متغيرات: weeks, trend_values
trend_data = pd.DataFrame({
    "Week": weeks,
    "Trend_Score": trend_values
})

# ثم تبدأ الاستيراد (لكن يفضل أن تكون import دائماً في أعلى الملف)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# Data Preparation
# ===============================
# Simulated weekly indicator scores
weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
trend_data = pd.DataFrame({
    "Week": weeks,
    "Yield Curve": [0.8, 0.9, 1.0, 1.0],
    "Fed Funds Rate": [0.7, 0.8, 1.0, 1.0],
    "EIA Inventories": [0.6, 0.7, 0.9, 1.0],
    "PMI": [0.4, 0.6, 0.8, 1.0],
    "VIX": [0.3, 0.5, 0.7, 0.8],
    "TD3C": [0.5, 0.6, 0.8, 0.9],
    "BDTI": [0.4, 0.5, 0.6, 0.7]
})
trend_data.set_index("Week", inplace=True)

# Composite Risk Score calculation
composite = trend_data.copy()
composite["Composite Risk Score"] = (
    0.25 * composite["Yield Curve"] +
    0.15 * composite["Fed Funds Rate"] +
    0.15 * composite["EIA Inventories"] +
    0.10 * composite["PMI"] +
    0.10 * composite["VIX"] +
    0.15 * composite["TD3C"] +
    0.10 * composite["BDTI"]
)

# Add Brent and Basrah Light Prices
composite["Brent Oil Price"] = [89, 85, 78, 74]
composite["Basrah Light Price"] = [87, 83, 75, 71]

# ===============================
# Streamlit Dashboard
# ===============================
st.title("🛢️ Iraqi Oil Risk & Price Forecast")

st.subheader("📊 Weekly Risk & Price Comparison")
st.dataframe(composite, use_container_width=True)

# ===============================
# Chart: Risk Score vs Brent & Basrah
# ===============================
st.subheader("📉 Risk Score vs Brent & Basrah Light Prices")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Week")
ax1.set_ylabel("Composite Risk Score", color='tab:red')
ax1.plot(composite.index, composite["Composite Risk Score"], color='tab:red', marker='o', label="Risk Score")
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
ax2.set_ylabel("Oil Price ($)", color='black')
ax2.plot(composite.index, composite["Brent Oil Price"], color='tab:blue', marker='s', linestyle='--', label="Brent Price")
ax2.plot(composite.index, composite["Basrah Light Price"], color='tab:green', marker='^', linestyle='--', label="Basrah Light")
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(65, 95)

fig.tight_layout()
ax2.legend(loc='upper center')
st.pyplot(fig)

# ===============================
# Forecast: Next Week (Week 5)
# ===============================
st.subheader("📈 Simple Forecast – Week 5")
next_score = composite["Composite Risk Score"].iloc[-1]
next_brent = composite["Brent Oil Price"].iloc[-1] - (next_score - 0.9) * 20
next_basrah = next_brent - 2

forecast = {
    "Forecast Week": "Week 5",
    "Projected Risk Score": round(next_score, 3),
    "Projected Brent Price": round(next_brent, 2),
    "Projected Basrah Light Price": round(next_basrah, 2)
}

st.json(forecast)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Model developed for fiscal and oil strategy. Simulated for academic/policy insights.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# ===============================
# Data Preparation
# ===============================
weeks = ["Week 1", "Week 2", "Week 3", "Week 4"]
trend_data = pd.DataFrame({
    "Week": weeks,
    "Yield Curve": [0.8, 0.9, 1.0, 1.0],
    "Fed Funds Rate": [0.7, 0.8, 1.0, 1.0],
    "EIA Inventories": [0.6, 0.7, 0.9, 1.0],
    "PMI": [0.4, 0.6, 0.8, 1.0],
    "VIX": [0.3, 0.5, 0.7, 0.8],
    "TD3C": [0.5, 0.6, 0.8, 0.9],
    "BDTI": [0.4, 0.5, 0.6, 0.7]
})
trend_data.set_index("Week", inplace=True)

composite = trend_data.copy()
composite["Composite Risk Score"] = (
    0.25 * composite["Yield Curve"] +
    0.15 * composite["Fed Funds Rate"] +
    0.15 * composite["EIA Inventories"] +
    0.10 * composite["PMI"] +
    0.10 * composite["VIX"] +
    0.15 * composite["TD3C"] +
    0.10 * composite["BDTI"]
)

composite["Brent Oil Price"] = [89, 85, 78, 74]
composite["Basrah Light Price"] = [87, 83, 75, 71]

# ===============================
# Streamlit Dashboard
# ===============================
st.title("🛢️ Iraqi Oil Risk & Price Forecast Dashboard")

st.subheader("📊 Weekly Risk & Oil Price Assessment")
st.dataframe(composite, use_container_width=True)

# ===============================
# Chart: Risk Score vs Brent & Basrah
# ===============================
st.subheader("📉 Risk Score vs Brent & Basrah Light Prices")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("Week")
ax1.set_ylabel("Composite Risk Score", color='tab:red')
ax1.plot(composite.index, composite["Composite Risk Score"], color='tab:red', marker='o', label="Risk Score")
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.set_ylim(0, 1.1)

ax2 = ax1.twinx()
ax2.set_ylabel("Oil Price ($)", color='black')
ax2.plot(composite.index, composite["Brent Oil Price"], color='tab:blue', marker='s', linestyle='--', label="Brent Price")
ax2.plot(composite.index, composite["Basrah Light Price"], color='tab:green', marker='^', linestyle='--', label="Basrah Light")
ax2.tick_params(axis='y', labelcolor='black')
ax2.set_ylim(65, 95)

fig.tight_layout()
ax2.legend(loc='upper center')
st.pyplot(fig)

# ===============================
# Forecast for Week 5 (Simple Model)
# ===============================
st.subheader("📈 Basic Forecast – Week 5")
next_score = composite["Composite Risk Score"].iloc[-1]
next_brent = composite["Brent Oil Price"].iloc[-1] - (next_score - 0.9) * 20
next_basrah = next_brent - 2

st.json({
    "Forecast Week": "Week 5",
    "Projected Risk Score": round(next_score, 3),
    "Projected Brent Price": round(next_brent, 2),
    "Projected Basrah Light Price": round(next_basrah, 2)
})

# ===============================
# ARIMA Forecast – Brent Oil
# ===============================
st.subheader("🔮 ARIMA Forecast – Brent Oil Price (Next 3 Weeks)")
brent_series = pd.Series(composite["Brent Oil Price"].values, index=pd.RangeIndex(start=1, stop=5, step=1))
model = ARIMA(brent_series, order=(1, 1, 1))
model_fit = model.fit()
forecast_result = model_fit.forecast(steps=3)

forecast_weeks = [f"Week {i}" for i in range(5, 5 + 3)]
brent_forecast_df = pd.DataFrame({
    "Week": forecast_weeks,
    "Forecasted Brent Price": forecast_result.round(2)
})
st.dataframe(brent_forecast_df)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("ARIMA model generated for strategic insights. For academic and simulation purposes only.")

window = 3  # عدد الفترات
rolling_corr = data['Oil_Price'].rolling(window).corr(data['Fed_Rate_Δ'])

plt.figure(figsize=(8,4))
plt.plot(data['Date'], rolling_corr, label="Rolling Corr: Oil vs Fed Rate Δ")
plt.title("Rolling Correlation: Oil Price vs Fed Rate Change")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
for col in ["Oil_Price", "Fed_Rate_Δ", "PMI_Δ", "VIX_Δ", "EIA_Stock_Δ"]:
    plt.plot(data["Date"], data[col], label=col)

plt.title("Oil and Related Variables Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

