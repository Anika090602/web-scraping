import yfinance as yf
import matplotlib.pyplot as plt
import datetime

plt.style.use("fivethirtyeight")

goog = yf.Ticker("GOOGL")
goog_hist = goog.history(start=datetime.datetime(2010, 1, 1), end=datetime.datetime.today())

plt.figure(figsize=(16, 8))
plt.title("Closing Price History")
plt.plot(goog_hist["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Closing Price USD $", fontsize=18)
plt.show()

# Saving the historical data to a CSV file
goog_hist.to_csv("historical_stock_data.csv")
