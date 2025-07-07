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

import streamlit as st
import datetime
from datetime import timedelta

# Force rerun with a refresh button
if st.button("🔁 Refresh Market Data"):
    st.experimental_rerun()

# Show current time
st.markdown(f"⏱️ Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Install required packages if needed:
# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate synthetic daily data (April to June 2024)
date_range = pd.date_range(start="2025-04-01", end="2025-06-30", freq="D")
np.random.seed(42)
n = len(date_range)

# Step 2: Simulate macroeconomic variables
gold_price = 1900 + np.cumsum(np.random.normal(0, 3, n))  # base trend + noise
interest_rate = 2.5 + np.random.normal(0, 0.03, n)         # small fluctuations
inflation = 2 + 0.02 * np.arange(n)/30 + np.random.normal(0, 0.05, n)  # rising
usd_index = 100 - 0.01 * np.arange(n) + np.random.normal(0, 0.1, n)    # slight decline
geo_risk = np.clip(50 + 5*np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 3, n), 0, 100)

# Step 3: Create DataFrame
df = pd.DataFrame({
    'GoldPrice': gold_price,
    'InterestRate': interest_rate,
    'Inflation': inflation,
    'USDIndex': usd_index,
    'GeopoliticalRisk': geo_risk
}, index=date_range)

# Step 4: Prepare features (X) and target (y)
X = df[['InterestRate', 'Inflation', 'USDIndex', 'GeopoliticalRisk']]
y = df['GoldPrice']

# Step 5: Train/Test Split (no shuffling for time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual Gold Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Gold Price', color='orange')
plt.title(f'Gold Price Forecast (Linear Regression)\nMSE={mse:.2f}, R²={r2:.2f}')
plt.xlabel('Date')
plt.ylabel('Gold Price (XAU/USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Install required packages if needed:
# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate synthetic daily data (April to June 2024)
date_range = pd.date_range(start="2025-04-01", end="2025-06-30", freq="D")
np.random.seed(42)
n = len(date_range)

# Step 2: Simulate macroeconomic variables
gold_price = 1900 + np.cumsum(np.random.normal(0, 3, n))  # base trend + noise
interest_rate = 2.5 + np.random.normal(0, 0.03, n)         # small fluctuations
inflation = 2 + 0.02 * np.arange(n)/30 + np.random.normal(0, 0.05, n)  # rising
usd_index = 100 - 0.01 * np.arange(n) + np.random.normal(0, 0.1, n)    # slight decline
geo_risk = np.clip(50 + 5*np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 3, n), 0, 100)

# Step 3: Create DataFrame
df = pd.DataFrame({
    'GoldPrice': gold_price,
    'InterestRate': interest_rate,
    'Inflation': inflation,
    'USDIndex': usd_index,
    'GeopoliticalRisk': geo_risk
}, index=date_range)

# Step 4: Prepare features (X) and target (y)
X = df[['InterestRate', 'Inflation', 'USDIndex', 'GeopoliticalRisk']]
y = df['GoldPrice']

# Step 5: Train/Test Split (no shuffling for time-series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 6: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 8: Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual Gold Price', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Gold Price', color='orange')
plt.title(f'Gold Price Forecast (Linear Regression)\nMSE={mse:.2f}, R²={r2:.2f}')
plt.xlabel('Date')
plt.ylabel('Gold Price (XAU/USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
