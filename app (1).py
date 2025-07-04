
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet

st.set_page_config(page_title="Market Analysis & Forecast", layout="wide")

st.title("📊 Market Analysis and Forecast Dashboard")

# Sidebar
st.sidebar.header("Configuration")
asset = st.sidebar.selectbox("Select Asset", {
    "Gold (GC=F)": "GC=F",
    "EUR/USD": "EURUSD=X",
    "Oil (CL=F)": "CL=F",
    "S&P 500": "^GSPC"
})
days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=30)

# Data download
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
df = yf.download(asset, start=start_date, end=end_date)

if df.empty:
    st.error("Failed to fetch data. Please check the ticker or try again later.")
    st.stop()

df = df.reset_index()
df['ds'] = df['Date']
df['y'] = df['Close']

# Forecast
model = Prophet()
model.fit(df[['ds', 'y']])
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)

# Chart: Price history + forecast
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Historical Price", line=dict(color='black')))
fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color='blue')))
fig.update_layout(title=f"{asset} - Price & Forecast", xaxis_title="Date", yaxis_title="Price", height=500)
st.plotly_chart(fig, use_container_width=True)

# Chart: Volatility (20-day rolling)
df['Return'] = df['y'].pct_change()
df['Volatility'] = df['Return'].rolling(window=20).std()
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=df['ds'], y=df['Volatility'], name="Volatility", line=dict(color='orange')))
fig_vol.update_layout(title=f"{asset} - 20-Day Rolling Volatility", xaxis_title="Date", yaxis_title="Volatility", height=400)
st.plotly_chart(fig_vol, use_container_width=True)

# Chart: Cumulative Return
df['Cumulative Return'] = (1 + df['Return']).cumprod()
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(x=df['ds'], y=df['Cumulative Return'], name="Cumulative Return", line=dict(color='green')))
fig_cum.update_layout(title=f"{asset} - Cumulative Return", xaxis_title="Date", yaxis_title="Growth", height=400)
st.plotly_chart(fig_cum, use_container_width=True)

# Show Forecast Data
st.subheader("📈 Forecast Table")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days).reset_index(drop=True))

# Summary stats
st.subheader("📋 Statistical Summary")
returns = df['Return'].dropna()
stats = {
    "Average Daily Return": returns.mean(),
    "Volatility": returns.std(),
    "Sharpe Ratio": returns.mean() / returns.std() if returns.std() != 0 else np.nan
}
st.json({k: f"{v:.4f}" for k, v in stats.items()})
