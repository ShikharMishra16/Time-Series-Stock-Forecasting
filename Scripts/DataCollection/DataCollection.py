#fb, tsla, appl, msft
import yfinance as yf
df = yf.download('TSLA', start='2015-01-01', end='2025-06-01')
df.to_csv('TSLA.csv')

df = yf.download('AAPL', start='2015-01-01', end='2025-06-01')
df.to_csv('AAPL.csv')

df = yf.download('MSFT', start='2015-01-01', end='2025-06-01')
df.to_csv('MSFT.csv')

df = yf.download('NFLX', start='2015-01-01', end='2025-06-01')
df.to_csv('NFLX.csv')

df = yf.download('NVDA', start='2015-01-01', end='2025-06-01')
df.to_csv('NVDA.csv')