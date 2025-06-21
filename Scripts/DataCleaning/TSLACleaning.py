import pandas as pd
import matplotlib.pyplot as plt

#Load and clean
df = pd.read_csv(r"C:\Users\shikh\Desktop\TIME SERIES STOCK PROJECT\1.DataCollection\TSLA.csv", parse_dates=["Date"], index_col="Date")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df = df.dropna()
print("Shape:", df.shape)
print(df.describe())
print("Missing values:\n", df.isnull().sum())

#Saving
df.to_csv('TSLA_cleaned.csv')
