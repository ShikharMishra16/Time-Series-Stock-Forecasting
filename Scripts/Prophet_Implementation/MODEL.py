import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

df = pd.read_csv("AAPL_cleaned.csv", parse_dates=["Date"], dayfirst=True)
#renaming columns for prophete
df_prophet = df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

#Fitting the model
model = Prophet()
model.fit(df_prophet)
#Future Forecasting
future = model.make_future_dataframe(periods=30)  # forecast 30 days into future
forecast = model.predict(future)
# Forecast Line Plot
fig1 = model.plot(forecast)
fig1.savefig("AAPL_forecast_plot.png")
# trends and other componenets
fig2 = model.plot_components(forecast)
fig2.savefig("AAPL_forecast_components.png")

forecast.to_csv("AAPL_forecast_data.csv", index=False)
print("Outputs saved.")# to verify the implementation of prophet model successfully
