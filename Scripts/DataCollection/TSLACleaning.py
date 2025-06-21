import pandas as pd
import matplotlib.pyplot as plt

#Load and clean
df = pd.read_csv(r"C:\Users\shikh\Desktop\TIME SERIES STOCK PROJECT\1.DataCollection\TSLA.csv", parse_dates=["Date"], index_col="Date")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.dropna()
print("Shape:", df.shape)
print(df.describe())
print("Missing values:\n", df.isnull().sum())

#Close Price
plt.figure(figsize=(12,6))
df['Close'].plot(title='TSLA Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid()
plt.show()

#Rolling thirty
df['Rolling thirty'] = df['Close'].rolling(window=30).mean()
plt.figure(figsize=(12,6))
df[['Close', 'Rolling thirty']].plot(title='30-Day Average')
plt.grid()
plt.show()

df.to_csv('TSLA_cleaned.csv')