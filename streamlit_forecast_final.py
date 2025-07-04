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

import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="تحليل الارتباط بين الأصول", layout="wide")
st.title("📊 تحليل الارتباط بين الأصول المالية الرئيسية")

# الرموز المالية للأصول
tickers = {
    "Gold": "GC=F",
    "Oil (WTI)": "CL=F",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "S&P 500": "^GSPC"
}

start_date = "2023-01-01"
end_date = "2025-07-01"

st.markdown("### تحميل البيانات من Yahoo Finance...")

# تحميل البيانات من Yahoo
data = yf.download(list(tickers.values()), start=start_date, end=end_date)["Close"]

# إعادة تسمية الأعمدة
data.columns = tickers.keys()

# حذف الصفوف الفارغة
data.dropna(inplace=True)

# عرض جزء من البيانات
st.markdown("### عينة من البيانات:")
st.dataframe(data.tail())

# حساب مصفوفة الارتباط
correlation_matrix = data.corr()

# رسم Heatmap للارتباط
st.markdown("### 🔥 خريطة الارتباط بين الأصول (Correlation Heatmap)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# شرح سريع
st.markdown("""
**ملاحظة**:
- القيم القريبة من **+1** تعني ارتباط موجب قوي.
- القيم القريبة من **-1** تعني ارتباط سلبي قوي.
- القيم القريبة من **0** تعني عدم وجود ارتباط مباشر.
""")
import yfinance as yf
import pandas as pd
import datetime

# Define date range
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=7)

# Asset list
assets = {
    "S&P 500": "^GSPC",
    "Gold": "GC=F",
    "Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN"
}

# Download data (only 'Close' to avoid 'Adj Close' issues)
raw_data = yf.download(list(assets.values()), start=start_date, end=end_date)["Close"]

# Rename columns to asset names
raw_data.columns = list(assets.keys())

# Drop rows with missing values
raw_data.dropna(inplace=True)

# Calculate weekly returns
weekly_returns = (raw_data.iloc[-1] - raw_data.iloc[0]) / raw_data.iloc[0] * 100
weekly_returns = weekly_returns.sort_values(ascending=False)

# Print summary
print("📊 Weekly Market Highlights\n")
print(weekly_returns.to_frame(name="7-Day Return (%)").round(2))

# Strategy summary
print("\n📌 Strategic Summary:")
top = weekly_returns.index[0]
bottom = weekly_returns.index[-1]

print(f"- ✅ Top Performer: {top} (+{weekly_returns[top]:.2f}%)")
print(f"- 🔻 Worst Performer: {bottom} ({weekly_returns[bottom]:.2f}%)")
print("- 🧠 Strategy: Focus on top performers and reconsider exposure to underperformers.")

import yfinance as yf
import pandas as pd
import datetime

# Define date range
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=7)

# Asset list
assets = {
    "S&P 500": "^GSPC",
    "Gold": "GC=F",
    "Oil": "CL=F",
    "EUR/USD": "EURUSD=X",
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN"
}

# Download data (only 'Close' to avoid 'Adj Close' issues)
raw_data = yf.download(list(assets.values()), start=start_date, end=end_date)["Close"]

# Rename columns to asset names
raw_data.columns = list(assets.keys())

# Drop rows with missing values
raw_data.dropna(inplace=True)

# Calculate weekly returns
weekly_returns = (raw_data.iloc[-1] - raw_data.iloc[0]) / raw_data.iloc[0] * 100
weekly_returns = weekly_returns.sort_values(ascending=False)

# Print summary
print("📊 Weekly Market Highlights\n")
print(weekly_returns.to_frame(name="7-Day Return (%)").round(2))

# Strategy summary
print("\n📌 Strategic Summary:")
top = weekly_returns.index[0]
bottom = weekly_returns.index[-1]

print(f"- ✅ Top Performer: {top} (+{weekly_returns[top]:.2f}%)")
print(f"- 🔻 Worst Performer: {bottom} ({weekly_returns[bottom]:.2f}%)")
print("- 🧠 Strategy: Focus on top performers and reconsider exposure to underperformers.")

