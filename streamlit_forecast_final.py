import yfinance as yf
import datetime

# Tickers definition
tickers = {
    "Gold": "GC=F",
    "Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X"
}

# Define start and end date
start_date = datetime.date(2022, 1, 1)
end_date = datetime.date.today()

# Download data
data = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]

# Basic Streamlit app
import streamlit as st
import matplotlib.pyplot as plt

st.title("Financial Forecast Dashboard")
st.subheader("Historical Closing Prices")

# Show dataframe
st.dataframe(data.tail())

# Plot data
fig, ax = plt.subplots()
data.plot(ax=ax)
st.pyplot(fig)
