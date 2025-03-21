import pandas as pd
import yfinance as yf
import numpy as np
import great_expectations as gx
import matplotlib.pyplot as plt
import seaborn as sns

###### Data Collection ######

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
print(financials_df.describe(), "\n\n")


###### Data Cleaning ######

# Analyse the data collected and clean the data
# Creates a data context to manage expectations and validations
context = gx.get_context()

# Connects the pandas DataFrame to the data context and creates a batch to validate
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Defines the batch, specifying that the whole DataFrame should be validated
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": financials_df})

# Create various expectations for different columns in the dataset
expectation1 = gx.expectations.ExpectColumnValuesToBeUnique(column="Ticker")

expectation2 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Net Income")

expectation3 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Operating Expense")

expectation4 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Operating Expense")

expectation5 = gx.expectations.ExpectColumnValuesToBeInSet(column="Industry", value_set=["Information Technology", "Health Care", "Financials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Utilities", "Real Estate", "Energy", "Communication Services", "Materials"])

expectation6 = gx.expectations.ExpectColumnValuesToBeBetween(column="2024-Tax Rate For Calcs", min_value=0, max_value=1)

expectation7 = gx.expectations.ExpectTableRowCountToEqual(value=503)

expectation8= gx.expectations.ExpectColumnValuesToBeOfType(column="2024-Total Revenue", type_="float64")

expectation9 = gx.expectations.ExpectColumnValuesToBeOfType(column="2024-Gross Profit", type_="float64")

expectation10 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Operating Revenue")

expectation11 = gx.expectations.ExpectTableColumnCountToEqual(value=len(expected_columns)*5 + 2)

# Prints the validation result for the batch
for expectation in [expectation1, expectation2, expectation3, expectation4, expectation5, expectation6, expectation7, expectation8, expectation9, expectation10, expectation11]:
    validation_result = batch.validate(expectation)
    print(f"{validation_result}\n")

# Missing Values Heatmap
print(financials_df.info(), "\n")
plt.figure(figsize=(15, 6))
sns.heatmap(financials_df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap Not Cleaned")
plt.show()

# Drop rows with 30% or more missing values
financials_df = financials_df.dropna(axis=0, thresh=0.3*len(financials_df.columns))

# Fill missing values using recent years
metric_groups = {}
for col in financials_df.columns:
    if '-' in col and col not in ['Ticker', 'Industry']:
        _, metric = col.split('-', 1)
        metric_groups.setdefault(metric, []).append(col)

for metric, cols in metric_groups.items():
    # Sort columns by year descending
    sorted_cols = sorted(cols, key=lambda x: x.split('-')[0], reverse=True)
    
    # Backward fill using older years
    financials_df[sorted_cols] = financials_df[sorted_cols].bfill(axis=1)

# Drop Columns with > 30% Missing Data
threshold = 0.3 * len(financials_df)
financials_df = financials_df.dropna(axis=1, thresh=threshold)
print(financials_df.info(), "\n")

# Fill missing values using recent years
metric_groups = {}
for col in financials_df.columns:
    if '-' in col and col not in ['Ticker', 'Industry']:
        _, metric = col.split('-', 1)
        metric_groups.setdefault(metric, []).append(col)

# Fill the rest missing values using median
for metric, cols in metric_groups.items():
    # Calculate median for remaining NAs
    existing_cols = [c for c in cols if c in financials_df.columns]
    if len(existing_cols) == 0:
        continue
    all_values = financials_df[existing_cols].values.flatten()
    metric_median = np.nanmedian(all_values)
    financials_df[existing_cols] = financials_df[existing_cols].fillna(metric_median)


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


###### Data Analysis ######

# Analyse the cleaned data
# Creates a data context to manage expectations and validations
context = gx.get_context()

# Connects the pandas DataFrame to the data context and creates a batch to validate
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

# Defines the batch, specifying that the whole DataFrame should be validated
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": financials_df})

# Create various expectations for different columns in the dataset
expectation1 = gx.expectations.ExpectColumnValuesToBeUnique(column="Ticker")

expectation2 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Net Interest Income")

expectation3 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Operating Expense")

expectation4 = gx.expectations.ExpectColumnValuesToNotBeNull(column="2024-Operating Expense")

expectation5 = gx.expectations.ExpectColumnValuesToBeInSet(column="Industry", value_set=["Information Technology", "Health Care", "Financials", "Industrials", "Consumer Discretionary", "Consumer Staples", "Utilities", "Real Estate", "Energy", "Communication Services", "Materials"])

expectation6 = gx.expectations.ExpectColumnValuesToBeBetween(column="2024-Tax Rate For Calcs", min_value=0, max_value=1)

expectation7 = gx.expectations.ExpectTableRowCountToBeBetween(min_value=470, max_value=503)

expectation8= gx.expectations.ExpectColumnValuesToBeOfType(column="2024-Total Revenue", type_="float64")

expectation9 = gx.expectations.ExpectColumnValuesToBeOfType(column="2024-Operating Expense", type_="float64")

expectation10 = gx.expectations.ExpectTableColumnCountToBeBetween(min_value=20, max_value=50)

# Prints the validation result for the batch
for expectation in [expectation1, expectation2, expectation3, expectation4, expectation5, expectation6, expectation7, expectation8, expectation9, expectation10]:
    validation_result = batch.validate(expectation)
    print(f"{validation_result}\n")

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
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Correlation Matrix of Financial Metrics", fontsize=14, fontweight="bold")
plt.xticks(ha="right", fontsize=5)
plt.yticks(fontsize=5)
plt.show()

financials_df.to_csv("data.csv", index=False)