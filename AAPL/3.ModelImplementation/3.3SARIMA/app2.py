from pyexpat import model
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt, exp
import warnings
from io import BytesIO

warnings.filterwarnings("ignore")

# ------------------------------------------
# 🔧 Step 1 : Preprocessing
def preprocess_data(ticker="MSFT", period="5y", apply_log=True, max_diff=3):
    df = yf.download(ticker, period=period)
    if df.empty:
        st.error("❌ Could not fetch stock data. Check ticker or network.")
        st.stop()

    df = df[['Close']].dropna()
    df.columns = ['Close']
    df.index = pd.to_datetime(df.index)

    if apply_log:
        df['LogClose'] = np.log(df['Close'])
    else:
        df['LogClose'] = df['Close']

    series = df['LogClose'].copy()
    differencing_order = 0
    #  checking  stationarity and removing using differencing
    def is_stationary(s):
        return adfuller(s.dropna())[1] < 0.05

    while not is_stationary(series) and differencing_order < max_diff:
        series = series.diff().dropna()
        differencing_order += 1

    return df, series, differencing_order
# ------------------------------------------
# Step 2: Splitting the Data
def train_test_split(series, test_size=40):
    train = series[:-test_size]
    test = series[-test_size:]
    return train, test
# ------------------------------------------
# Step 3: Build & Fit SARIMA
def train_sarima(train, order, seasonal_order):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)
    return results
# ------------------------------------------
# Step 4: Evaluate Model

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    st.subheader("📊 Evaluation Metrics")
    st.markdown(f"- **MAE**: {mae:.4f}")
    st.markdown(f"- **RMSE**: {rmse:.4f}")
    st.markdown(f"- **R²**: {r2:.4f}")
    return mae, rmse, r2


def download_csv(df, filename="forecast.csv"):
    buffer = BytesIO()
    df.to_csv(buffer, index=True)
    buffer.seek(0)
    st.download_button("⬇️ Download Forecast CSV", buffer, file_name=filename, mime="text/csv")

# ------------------------------------------
# 📈 Main Streamlit App
# ------------------------------------------

def main():
    st.title("📊 Stock Forecasting with SARIMA")
    st.markdown("Forecast stock prices with SARIMA model, parameter tuning, and download option.")

    # Inputs
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT)", value="AAPL")
    forecast_days = st.slider("Forecast Days", 30, 180, step=10, value=60)
    period = st.selectbox("Data Period", ["1y", "2y", "5y"], index=0)
    apply_log = st.checkbox("Apply Log Transformation", value=True)

    # Preprocess
    if st.button("Start Analysis"):
        with st.spinner("🔄 Fetching and preprocessing data..."):
            df, stationary_series, d_order = preprocess_data(ticker=ticker, period=period, apply_log=apply_log)
            st.session_state['df'] = df
            st.session_state['series'] = stationary_series
            st.session_state['d_order'] = d_order
            st.session_state['forecast_days'] = forecast_days
            st.session_state['apply_log'] = apply_log
            st.session_state['ticker'] = ticker

    # Only proceed if session_state contains data
    if 'df' in st.session_state:
        df = st.session_state['df']
        stationary_series = st.session_state['series']
        d_order = st.session_state['d_order']
        forecast_days = st.session_state['forecast_days']
        apply_log = st.session_state['apply_log']
        ticker = st.session_state['ticker']

        # Plots
        st.subheader("📈 Raw Closing Price")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(df['Close'], label="Original Close Price")
        ax1.set_title(f"{ticker} Closing Prices")
        ax1.grid(); ax1.legend()
        st.pyplot(fig1)

        st.subheader("🔄 Stationary Series")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(stationary_series, label="Stationary Series")
        ax2.set_title("Stationary Series")
        ax2.grid(); ax2.legend()
        st.pyplot(fig2)

        st.subheader("📊 ACF and PACF")
        fig3, ax3 = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(stationary_series, ax=ax3[0], lags=40)
        plot_pacf(stationary_series, ax=ax3[1], lags=40, method="ywm")
        st.pyplot(fig3)

        # User inputs for SARIMA
        st.markdown("### 🔧 Select SARIMA Parameters")
        col1, col2 = st.columns(2)
        with col1:
            p = st.number_input("p (AR)", min_value=0, max_value=10, value=1, key="p")
            d = st.number_input("d (Diff)", min_value=0, max_value=2, value=d_order, key="d")
            q = st.number_input("q (MA)", min_value=0, max_value=10, value=1, key="q")
        with col2:
            P = st.number_input("P (Seasonal AR)", min_value=0, max_value=5, value=0, key="P")
            D = st.number_input("D (Seasonal Diff)", min_value=0, max_value=2, value=0, key="D")
            Q = st.number_input("Q (Seasonal MA)", min_value=0, max_value=5, value=0, key="Q")
            s = st.number_input("s (Seasonal Period)", min_value=1, max_value=365, value=12, key="s")

        # Run forecast
        if st.button("Run Forecast"):
            with st.spinner("📈 Training SARIMA model..."):
                train, test = train_test_split(stationary_series, test_size=forecast_days)
                model = train_sarima(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
                st.subheader("📄 SARIMA Model Summary")
                st.text(model.summary())
                pred = model.get_forecast(steps=forecast_days)
                pred_mean = pd.Series(pred.predicted_mean.values, index=test.index)
                conf_int = pred.conf_int(); conf_int.index = test.index

                evaluate_model(test, pred_mean)

                st.subheader("🔮 Forecast Plot")
                fig4, ax4 = plt.subplots(figsize=(12, 6))
                ax4.plot(stationary_series, label="Stationary Series")
                ax4.plot(pred_mean.index, pred_mean, color='orange', label="Forecast")
                ax4.fill_between(pred_mean.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
                ax4.axvline(x=pred_mean.index[0], color='gray', linestyle='--', label="Forecast Start")
                ax4.set_title(f"{ticker} - Forecast ({forecast_days} days)")
                ax4.legend(); ax4.grid()
                st.pyplot(fig4)

                st.subheader("🔎 Residuals")
                residuals = test - pred_mean
                fig5, ax5 = plt.subplots(figsize=(10, 4))
                ax5.plot(residuals, label='Residuals')
                ax5.axhline(0, linestyle='--', color='black')
                ax5.set_title("Forecast Residuals")
                ax5.grid(); ax5.legend()
                st.pyplot(fig5)

                # Inverse transform if needed
                if apply_log:
                    forecast_values = np.exp(pred_mean)
                else:
                    forecast_values = pred_mean

                
                forecast_df = pd.DataFrame({
                    "Date": forecast_values.index,
                    "Forecasted Price": forecast_values.values
                }).set_index("Date")
                
                if apply_log:
                    #forecast_values = np.exp(pred_mean)
                    # Get last observed log-price value
                    last_log_value = df['LogClose'].iloc[-forecast_days - 1]

# Cumulatively sum the predicted diffs to reconstruct log prices
                    log_forecast = last_log_value + pred_mean.cumsum()

# Convert back to original price scale
                    forecast_values = np.exp(log_forecast)

                    actual_series = np.exp(df['LogClose'][-forecast_days:])  # ✅ fix here
                else:
                    last_log_value = df['LogClose'].iloc[-forecast_days - 1]
                    log_forecast = last_log_value + pred_mean.cumsum()
                    forecast_values = np.exp(log_forecast)
                    #forecast_values = pred_mean
                    actual_series = df['Close'][-forecast_days:]

                forecast_df = pd.DataFrame({
                    "Forecasted Price": forecast_values.values,
                    "Actual Price": actual_series.values
                }, index=forecast_values.index)
                st.subheader("📌 Forecasted Price for Next Day")
                st.markdown(f"**Next Day Forecast:** ${forecast_values.iloc[0]:.2f}")
                st.markdown(f"**Last Observed Price:** ${actual_series.iloc[0]:.2f}")
                st.subheader("📋 Forecast Table (Predicted vs Actual)")
                st.dataframe(forecast_df)
                download_csv(forecast_df, f"{ticker}_forecast_comparison.csv")

                

if __name__ == "__main__":
    main()
