import yfinance as yf
import pandas as pd

# List of assets, each line corresponding to a category
'''
Line 1: Semiconductors, Line 2: General Software
Line 3: AI-Centric Software, Line 4: Data Centers
Line 5: Utilities, Line 6: Aerospace & Defense
'''
assets = ["NVDA","AVGO","MU","AMD","MRVL",
          "MSFT","CRWD","SNDK","TEL","CDNS",
          "SNOW","CRM","NET","ADBE","NOW",
          "EQIX","DLR","VRT","ETN","IRM",
          "NEE","CEG","SO","SRE",
          "LMT","RTX","GD"]

# Historical data taken from January 1, 2020 to May 31, 2026
# Januaary 1 omitted as it is a holiday, so the start date is January 2, 2020
start_date = "2020-01-02"
end_date = "2026-05-31"

# Importing daily data
df = yf.download(assets, start=start_date, end=end_date, auto_adjust=True,progress=False)
# daily = df.xs("Close", axis=1, level=0)

prices = df["Close"]

# Re-arranging as per the list 'assets'
prices = prices.reindex(columns=assets)

# Keeping assets that have data from the start date
start_ts = pd.Timestamp(start_date)

valid = [ ticker for ticker in assets if ticker in prices.columns
         and prices[ticker].first_valid_index() is not None 
         and prices[ticker].first_valid_index() <= start_ts ]

omitted = [ ticker for ticker in assets if ticker not in valid ]

# Acquire prices and returns for the valid assets
daily_prices = prices[valid].copy()
returns = daily_prices.pct_change()

daily_prices.columns = [ ticker + "_Price" for ticker in daily_prices.columns ]
returns.columns = [ ticker + "_Return" for ticker in returns.columns ]

daily = pd.concat([daily_prices, returns], axis=1)

# Testing the code
print("Omitted:", omitted)

# Exporting to .CSV and .XLSX formats
daily.to_csv("asset_prices_returns.csv")
daily.to_excel("asset_prices_returns.xlsx")
