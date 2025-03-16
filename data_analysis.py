import pandas as pd
import yfinance as yf
import numpy as np
import great_expectations as gx
import matplotlib.pyplot as plt
import seaborn as sns

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

print(financials_df.head(10), "\n")
print(financials_df.info(), "\n")
print(financials_df.describe(), "\n")

# Drop Columns with > 40% Missing Data
threshold = 0.4 * len(financials_df)
financials_df = financials_df.dropna(axis=1, thresh=threshold)
print("Remaining columns after dropping:", financials_df.shape[1], "\n")

# Removing Highly Correlated Columns (keeping one from each pair)
corr_matrix = financials_df.corr(numeric_only=True).abs()
# Create mask for upper triangle (excluding diagonal)
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
high_corr_mask = (corr_matrix > 0.9) & upper_triangle

# Find indices of correlated pairs
rows, cols = np.where(high_corr_mask)

# Collect columns to drop (older years)
to_drop = set()

for i, j in zip(rows, cols):
    col_i = corr_matrix.columns[i]
    col_j = corr_matrix.columns[j]
    
    # Extract years from first 4 characters
    year_i = int(col_i[:4])
    year_j = int(col_j[:4])
        
    # Compare years and mark older for deletion
    if year_i > year_j:
        to_drop.add(col_j)
    elif year_j > year_i:
        to_drop.add(col_i)
    else:
        # For same year, keep first occurrence
        if corr_matrix.columns.get_loc(col_i) < corr_matrix.columns.get_loc(col_j):
            to_drop.add(col_j)
        else:
            to_drop.add(col_i)

# Drop older columns while keeping most recent
financials_df = financials_df.drop(columns=list(to_drop))
print("Remaining columns after dropping:", financials_df.shape[1], "\n")

# Missing Values Heatmap
print(financials_df.info(), "\n")
plt.figure(figsize=(15, 6))
sns.heatmap(financials_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# Industry Frequency Bar Plot
plt.figure(figsize=(14, 6))
industry_grouping = financials_df.groupby("Industry").size().sort_values(ascending=True)
ax = industry_grouping.plot(kind="barh", color="skyblue", edgecolor="black")
plt.xlabel("Count", fontsize=12)
plt.ylabel("Industry", fontsize=12)
plt.title("Frequency of Industries in the Dataset", fontsize=14, fontweight="bold")
plt.show()

# Histograms for Numeric Columns
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 12))  # Adjust grid size for readability
axes = axes.flatten()

numeric_cols = financials_df.drop(columns=["Ticker", "Industry"]).columns
for i, col in enumerate(numeric_cols[:20]):  # Limit to first 20 histograms
    sns.histplot(financials_df[col], bins=20, kde=True, ax=axes[i], color="cornflowerblue")
    axes[i].set_title(col, fontsize=10)
    axes[i].tick_params(axis="x", labelsize=8)
    axes[i].grid(True)

plt.tight_layout()
plt.suptitle("Distribution of Financial Metrics", fontsize=16, fontweight="bold")
plt.subplots_adjust(top=0.95)
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(14, 10))
correlation_matrix = financials_df.corr(numeric_only=True)

sns.heatmap(
    correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1
)
plt.title("Correlation Matrix of Financial Metrics", fontsize=14, fontweight="bold")
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.yticks(rotation=30, fontsize=10)
plt.show()

'''
# Analyse the data collected and clean the data

# Creates a data context to manage expectations and validations
context = gx.get_context()

# Connects the pandas DataFrame to the data context and creates a batch to validate
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Defines the batch, specifying that the whole DataFrame should be validated
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": financials_df})'
'''