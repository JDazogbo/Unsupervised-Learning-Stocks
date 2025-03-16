import pandas as pd
import yfinance as yf
import numpy as np
import great_expectations as gx

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

apple_finance = pd.DataFrame(tickers.tickers["AAPL"].financials)
print(apple_finance)

# Predefine the expected columns based on years and financial metrics
years = ['2024', '2023', '2022', '2021', '2020']
financial_metrics = apple_finance.index.values
expected_columns = [f"{year}-{metric}" for year in years for metric in financial_metrics]

# Initialize DataFrame with proper structure
financials_df = pd.DataFrame(columns=['Ticker', 'Industry'] + expected_columns)

# Iterate through each company in the S&P 500
for i, company in sp500_df.iterrows():
    ticker = company["Symbol"]
    industry = company["GICS Sector"]
    
    # Get financial data with proper date handling
    financial_data = yf.Ticker(ticker).financials
    financial_data.columns = pd.to_datetime(financial_data.columns).strftime('%Y')
        
    # Create row data with NaN initialization
    row_data = {col: np.nan for col in financials_df.columns}
    row_data.update({'Ticker': ticker, 'Industry': industry})
        
    # Fill available data
    for year in years:
        if year in financial_data.columns:
            for metric in financial_metrics:
                if metric in financial_data.index:
                    col_name = f"{year}-{metric}"
                    row_data[col_name] = financial_data.loc[metric, year]
        
    # Append to DataFrame
    financials_df = pd.concat([financials_df, pd.DataFrame([row_data])], ignore_index=True)

print(financials_df.head(10))
print(financials_df.info())
print(financials_df.describe())

'''
# Analyse the data collected and clean the data

# Creates a data context to manage expectations and validations
context = gx.get_context()

# Connects the pandas DataFrame to the data context and creates a batch to validate
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Defines the batch, specifying that the whole DataFrame should be validated
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": financials_df})
'''