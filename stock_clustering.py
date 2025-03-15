import pandas as pd
import yfinance as yf

# URL of the Wikipedia page
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Read the tables on the page
tables = pd.read_html(url)
print(tables, "\n \n")

# Convert S&P 500 file into a dataframe
sp500_df = pd.DataFrame(tables[0])
print(sp500_df.head())
print(sp500_df.info())

# Convert to Tickers
tickers = yf.Tickers(sp500_df["Symbol"].to_list())
print(tickers, "\n")

# Financial data of apple of 2024 - 2021
financial_data_apple = tickers.tickers["AAPL"].financials
print(financial_data_apple, "\n")