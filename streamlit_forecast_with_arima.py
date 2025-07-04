
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
