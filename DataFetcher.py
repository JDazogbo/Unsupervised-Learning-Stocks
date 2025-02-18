import requests
import pandas as pd

class DataFetcher:
    def __init__(self):
        """Initialize with empty tickers and metrics."""
        self.tickers = []
        self.selected_metrics = {}

    def set_tickers(self, *tickers):
        """Set the tickers to fetch data for."""
        self.tickers = list(tickers)

    def set_metrics(self, **metric_flags):
        """
        Set the financial metrics to fetch.
        Example usage: fetcher.set_metrics(AssetsNoncurrent=True, InventoryNet=True)
        """
        self.selected_metrics = {metric: flag for metric, flag in metric_flags.items() if flag}

    def get_cik(self, ticker):
        """Retrieve CIK for a given ticker."""
        url = "https://www.sec.gov/files/company_tickers.json"
        response = requests.get(url, headers={"User-Agent": "your@email.com"}).json()
        
        for company in response.values():
            if company["ticker"].upper() == ticker.upper():
                return str(company["cik_str"]).zfill(10)
        return None

    def get_financial_data(self, cik):
        """Fetch financial data from SEC for a given CIK."""
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        response = requests.get(url, headers={"User-Agent": "your@email.com"})
        if response.status_code != 200:
            return None
        return response.json()

    def extract_metrics(self, financial_data):
        """Extract only the selected metrics from the financial data."""
        extracted_data = {}

        for metric in self.selected_metrics:
            try:
                values = financial_data["facts"]["us-gaap"][metric]["units"]["USD"]
                latest_value = sorted(values, key=lambda x: x.get("end", ""), reverse=True)[0]["val"]
                extracted_data[metric] = latest_value
            except (KeyError, IndexError):
                extracted_data[metric] = None  # Handle missing data gracefully

        return extracted_data

    def fetch_data(self):
        """Fetch financial data for all selected tickers and return as a DataFrame."""
        if not self.tickers:
            raise ValueError("No tickers set. Use `set_tickers(*tickers)` first.")
        if not self.selected_metrics:
            raise ValueError("No metrics selected. Use `set_metrics(**metric_flags)` first.")

        final_data = []

        for ticker in self.tickers:
            print(f"Fetching data for {ticker}...")
            
            cik = self.get_cik(ticker)
            if not cik:
                print(f"❌ CIK not found for {ticker}. Skipping...")
                continue

            financial_data = self.get_financial_data(cik)
            if not financial_data:
                print(f"❌ Failed to fetch financial data for {ticker}. Skipping...")
                continue

            # Extract selected metrics
            metrics_data = self.extract_metrics(financial_data)
            metrics_data["Ticker"] = ticker  # Include ticker for reference
            final_data.append(metrics_data)

        return pd.DataFrame(final_data)


# ==================== Example Usage ==================== #
fetcher = DataFetcher()

# Step 1: Select Tickers
fetcher.set_tickers("CRWD", "MSFT", "GOOGL", "AMZN")

# Step 2: Select Metrics (Pass True for ones you want)
fetcher.set_metrics(
    AccountsReceivableNetCurrent=True,
    RevenueFromContractWithCustomerExcludingAssessedTax=True,
)

# Step 3: Fetch Data
df = fetcher.fetch_data()

# Display the DataFrame
print(df.head())
