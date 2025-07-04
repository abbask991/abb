
import yfinance as yf
import datetime
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Asset list
tickers = {
    "Gold": "GC=F",
    "Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X"
}

# Streamlit UI
st.title("📈 Asset Forecast Dashboard")
asset = st.sidebar.selectbox("Select Asset", list(tickers.keys()))
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)

# Date range
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date.today()

# Load data
symbol = tickers[asset]
data = yf.download(symbol, start=start_date, end=end_date)["Close"]
data.dropna(inplace=True)

# Display raw data
st.subheader(f"{asset} - Historical Data")
st.line_chart(data)

# Fit ARIMA model
model = ARIMA(data, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=forecast_days)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
forecast_series = pd.Series(forecast, index=forecast_index)

# Combine and plot
fig, ax = plt.subplots()
data.plot(ax=ax, label="Historical")
forecast_series.plot(ax=ax, label="Forecast", color="red", linestyle="--")
plt.title(f"{asset} Price Forecast ({forecast_days} Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig)

# Show forecast table
st.subheader("📅 Forecasted Prices")
st.dataframe(forecast_series.rename("Forecast"))
# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crude Oil Dashboard", layout="centered")

st.title("📊 Crude Oil (WTI) Price Dashboard")

# Download Oil data from Yahoo Finance
symbol = "CL=F"  # Crude Oil Futures (WTI)
start_date = "2023-01-01"
end_date = "2025-07-01"

st.markdown("### Loading historical oil data...")

data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    st.error("⚠️ Failed to load data. Please check the symbol or your internet connection.")
else:
    # Show raw data
    st.markdown("### Sample of Raw Data")
    st.dataframe(data.tail())

    # Plot closing price
    st.markdown("### 📈 Closing Price Chart")
    st.line_chart(data['Close'])

    # Add simple moving average
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    st.markdown("### 📉 Technical Indicators")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='Close Price')
    ax.plot(data.index, data['SMA_20'], label='20-Day SMA', linestyle='--')
    ax.plot(data.index, data['SMA_50'], label='50-Day SMA', linestyle=':')
    ax.set_title('Crude Oil Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    st.pyplot(fig)
