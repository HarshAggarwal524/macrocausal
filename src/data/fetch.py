import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()  # Reads your .env file and loads FRED_API_KEY into the environment

def fetch_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads a single FRED time series and returns it as a DataFrame.

    Args:
        series_id: FRED series identifier (e.g. 'FEDFUNDS')
        start: Start date string, e.g. '2000-01-01'
        end: End date string, e.g. '2024-12-31'

    Returns:
        DataFrame with DateTimeIndex and one column named after series_id
    """
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))  
    # os.getenv reads the key that load_dotenv() loaded from your .env file

    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    # This is the actual API call — it goes out to FRED's server and returns
    # a pandas Series where the index is dates and values are the data points

    df = pd.DataFrame(series, columns=[series_id])
    # Wrap in a DataFrame and name the column after the series ID
    # so you always know which variable is in which column

    df.index = pd.to_datetime(df.index)
    # Make sure the index is recognised as dates, not plain strings

    return df


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads monthly closing prices from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. '^GSPC')
        start: Start date string
        end: End date string

    Returns:
        DataFrame with DateTimeIndex and one column named after ticker
    """
    raw = yf.download(ticker, start=start, end=end, interval="1mo",
                      auto_adjust=True, progress=False)
    # interval="1mo" gives monthly bars (one row per month)
    # auto_adjust=True corrects prices for stock splits and dividends
    # progress=False suppresses the download progress bar

    df = raw[["Close"]].rename(columns={"Close": ticker})
    # Keep only the closing price and rename the column to the ticker symbol

    return df


if __name__ == "__main__":
    START = "1990-01-01"
    END   = "2024-12-31"

    os.makedirs("data/raw", exist_ok=True)
    # Creates the data/raw folder if it doesn't already exist
    # exist_ok=True means it won't throw an error if the folder is already there

    print("Fetching FEDFUNDS...")
    fetch_fred("FEDFUNDS", START, END).to_csv("data/raw/fred_fedfunds.csv")

    print("Fetching CPIAUCSL...")
    fetch_fred("CPIAUCSL", START, END).to_csv("data/raw/fred_cpi.csv")

    print("Fetching GS10...")
    fetch_fred("GS10", START, END).to_csv("data/raw/fred_gs10.csv")

    print("Fetching S&P 500...")
    fetch_yahoo("^GSPC", START, END).to_csv("data/raw/yf_sp500.csv")

    print("Fetching VIX...")
    fetch_yahoo("^VIX", START, END).to_csv("data/raw/yf_vix.csv")

    print("Done. Check data/raw/")