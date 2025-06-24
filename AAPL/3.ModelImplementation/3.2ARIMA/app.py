import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import load_data, run_arima_forecast, auto_select_arima

# Page setup
st.set_page_config(page_title="Stock Forecast", layout="wide")
st.title("📈 Stock Price Forecasting with ARIMA")

# Sidebar inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
forecast_steps = st.sidebar.slider("Forecast Days", 7, 365, 30)

# Load data
data = load_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data found! Check ticker or date range.")
    st.stop()

# Display raw data
st.subheader(f"Raw Data: {ticker}")
st.line_chart(data)

# Auto-select ARIMA parameters
p, d, q = auto_select_arima(data['Close'])
st.sidebar.markdown(f"**Auto-Selected Parameters**: `p={p}, d={d}, q={q}`")

# Manual override options
st.sidebar.subheader("Advanced Settings")
use_custom = st.sidebar.checkbox("Manual ARIMA Parameters")
if use_custom:
    p = st.sidebar.slider("p (AR)", 0, 5, p)
    d = st.sidebar.slider("d (I)", 0, 2, d)
    q = st.sidebar.slider("q (MA)", 0, 5, q)

# Run forecasting
forecast, model = run_arima_forecast(data['Close'], p, d, q, forecast_steps)
forecast_dates = pd.date_range(data.index[-1], periods=forecast_steps+1, freq='B')[1:]
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_dates)

# Combine historical + forecast
combined = pd.concat([data[['Close']], forecast_df])

# Plot results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(combined.index, combined['Close'], label='Historical', color='blue')
ax.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', 
        linestyle='--', color='red')
ax.fill_between(forecast_df.index, 
                forecast * 0.95, 
                forecast * 1.05, 
                color='orange', alpha=0.2)
ax.set_title(f"ARIMA({p},{d},{q}) Forecast for {ticker}")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Metrics
last_actual = data['Close'][-1]
forecast_change = (forecast[0] - last_actual) / last_actual * 100
st.metric("Next Day Forecast", f"${forecast[0]:.2f}", 
          f"{forecast_change:.2f}% from last close")

# Model summary
st.subheader("Model Summary")
st.text(str(model.summary()))

# Download forecast
st.download_button(
    label="Download Forecast CSV",
    data=forecast_df.to_csv(),
    file_name=f"{ticker}_forecast.csv",
    mime="text/csv"
)