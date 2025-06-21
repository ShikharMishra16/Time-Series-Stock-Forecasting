import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("AAPL_cleaned.csv", parse_dates=["Date"], dayfirst=True)
#renaming columns for prophete
df_prophet = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# STEP 2: Initialize and Fit Prophet Model
model = Prophet()
model.fit(df_prophet)

# STEP 3: Make Future DataFrame and Forecast
future = model.make_future_dataframe(periods=30)  # forecast 30 days into future
forecast = model.predict(future)

# STEP 4: Save Forecast Plots
# Forecast Line Plot
fig1 = model.plot(forecast)
fig1.savefig("AAPL_forecast_plot.png")
# Trend, Seasonality Components
fig2 = model.plot_components(forecast)
fig2.savefig("AAPL_forecast_components.png")

# STEP 5: Save Forecast Data
forecast.to_csv("AAPL_forecast_data.csv", index=False)
print("Outputs saved.")