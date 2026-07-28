import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DATA_DIR = PROJECT_ROOT / "asset data"
ASSET_RETURNS_FILE = ASSET_DATA_DIR / "asset_prices_returns.csv"
MARKET_CAP_OUTPUT_FILE = ASSET_DATA_DIR / "market_caps.csv"


def build_yfinance_session():
    """
    Build a yfinance-compatible session when curl_cffi is installed.
    If it is not available, yfinance will use its default session.
    """
    try:
        from curl_cffi import requests as curl_requests

        return curl_requests.Session(impersonate="chrome")
    except Exception:
        return None


def get_asset_list():
    returns = pd.read_csv(ASSET_RETURNS_FILE, nrows=1)
    return returns.columns[1:25].str.replace("_Price", "", regex=False).tolist()


def fetch_market_cap(asset, session=None, retries=3, sleep_seconds=1):
    """
    Fetch real live market cap from yfinance.
    No median/equal-weight fallback is used here. Missing values stay missing.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(asset, session=session) if session is not None else yf.Ticker(asset)

            market_cap = ticker.info.get("marketCap", np.nan)

            if pd.isna(market_cap):
                fast_info = getattr(ticker, "fast_info", {})
                market_cap = fast_info.get("market_cap", np.nan)

            if not pd.isna(market_cap):
                return int(market_cap), None

        except Exception as error:
            last_error = str(error)

        if attempt < retries:
            time.sleep(sleep_seconds)

    return np.nan, last_error


def main():
    asset_list = get_asset_list()
    session = build_yfinance_session()

    market_cap_rows = []

    for asset in asset_list:
        market_cap, error = fetch_market_cap(asset, session=session)

        market_cap_rows.append({
            "Asset": asset,
            "MarketCap": market_cap,
            "FetchError": error
        })

        if pd.isna(market_cap):
            print(f"{asset}: missing market cap")
        else:
            print(f"{asset}: {market_cap:,.0f}")

    market_caps = pd.DataFrame(market_cap_rows)
    market_caps.to_csv(MARKET_CAP_OUTPUT_FILE, index=False)

    missing = market_caps[market_caps["MarketCap"].isna()]

    print(f"\nSaved: {MARKET_CAP_OUTPUT_FILE}")

    if not missing.empty:
        print("\nMissing market caps:")
        print(missing[["Asset", "FetchError"]].to_string(index=False))
        sys.exit(1)

    print("\nAll market caps fetched successfully.")


if __name__ == "__main__":
    main()
