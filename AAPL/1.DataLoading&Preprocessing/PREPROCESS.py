import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# --------------------------- STEP 1: Download Raw Data ---------------------------
ticker = "AAPL"  # you can replace this with any symbol
start_date = "2015-01-01"
end_date = "2024-12-31"

raw_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
raw_df.to_csv("raw_data.csv")
print("Raw data downloaded.")

# --------------------------- STEP 2: Clean the Data ---------------------------
df = raw_df.copy()
df.reset_index(inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

cleaned_df = df[['Close', 'High', 'Low']]  # use key price features
cleaned_df.to_csv("cleaned_data.csv")
print("Data cleaned.")

# --------------------------- STEP 3: Normalize (0-1 scaling) ---------------------------
minmax_scaler = MinMaxScaler()
normalized = pd.DataFrame(minmax_scaler.fit_transform(cleaned_df),
                          columns=['Close_Norm', 'High_Norm', 'Low_Norm'],
                          index=cleaned_df.index)
normalized.to_csv("normalized_data.csv")
print("Normalized data.")

# --------------------------- STEP 4: Standardize (mean=0, std=1) ---------------------------
standard_scaler = StandardScaler()
standardized = pd.DataFrame(standard_scaler.fit_transform(cleaned_df),
                            columns=['Close_Std', 'High_Std', 'Low_Std'],
                            index=cleaned_df.index)
standardized.to_csv("standardized_data.csv")
print("Standardized data.")

# --------------------------- STEP 5: Detect Stationarity ---------------------------
print("\nStationarity Test Results (ADF Test):")
for col in cleaned_df.columns:
    result = adfuller(cleaned_df[col])
    print(f"\n🔹 {col}")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] < 0.05:
        print("✅ Likely Stationary")
    else:
        print("⚠️  Likely Non-Stationary")

# Optional: Plot raw data for quick visual
plt.figure(figsize=(10, 5))
plt.plot(cleaned_df['Close'], label='Close Price')
plt.title(f"{ticker} Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
