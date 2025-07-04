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

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل أسعار النفط", layout="centered")

st.title("📊 تحليل أسعار النفط الخام (WTI)")

symbol = "CL=F"
start_date = "2023-01-01"
end_date = "2025-07-01"

st.markdown("### تحميل البيانات من Yahoo Finance...")

data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    st.error("⚠️ تعذّر تحميل البيانات.")
else:
    st.markdown("### بيانات خام")
    st.dataframe(data.tail())

    st.markdown("### الرسم البياني اليومي")
    st.line_chart(data['Close'])

    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    st.markdown("### مؤشرات فنية: متوسطات متحركة")
    fig, ax = plt.subplots()
    ax.plot(data.index, data['Close'], label='السعر')
    ax.plot(data.index, data['SMA_20'], label='SMA 20', linestyle='--')
    ax.plot(data.index, data['SMA_50'], label='SMA 50', linestyle=':')
    ax.legend()
    st.pyplot(fig)

