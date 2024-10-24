import streamlit as st
import yfinance as yf
import pandas as pd

def get_top_stocks():
    # List of top stock tickers
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "BRK-B", "V", "JNJ",
        "WMT", "JPM", "XOM", "BAC", "PG", "PFE", "DIS", "KO", "PEP", "CSCO",
        "ORCL", "NFLX", "INTC", "CVX", "MA", "NKE", "T", "VZ", "ADBE", "PLTR"
    ]

    # Fetch stock data with daily fallback for better availability
    data = yf.download(tickers, period="2d", interval="1d")

    if data.empty or len(data["Adj Close"].dropna()) < 2:
        st.warning("Not enough data points to calculate changes. Try again later.")
        return pd.DataFrame()  # Stop further execution if data is insufficient

    # Fill NaN values
    data = data.ffill().bfill()

    # Extract latest and previous prices
    current_data = data["Adj Close"].iloc[-1].reset_index()
    previous_close_data = data["Adj Close"].iloc[-2].reset_index()

    # Rename columns for clarity
    current_data.columns = ["Ticker", "Price"]
    previous_close_data.columns = ["Ticker", "Previous Price"]

    # Merge data and calculate changes
    combined_data = pd.merge(current_data, previous_close_data, on="Ticker")
    combined_data["Change"] = combined_data["Price"] - combined_data["Previous Price"]

    return combined_data

def display_ticker_tape():
    st.markdown(
        """
        <style>
        .ticker-tape {
            white-space: nowrap;
            overflow: hidden;
            box-sizing: border-box;
            padding: 10px 0;
        }
        .ticker-tape div {
            display: inline-block;
            padding-left: 100%;
            animation: ticker 30s linear infinite;
        }
        @keyframes ticker {
            0%   { transform: translateX(0); }
            100% { transform: translateX(-100%); }
        }
        .ticker-tape .up { color: green; }
        .ticker-tape .down { color: red; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    stock_data = get_top_stocks()

    if stock_data.empty:
        return  # Stop execution if no data is available

    stock_data["Display"] = stock_data.apply(
        lambda row: f"<span class='{ 'up' if row['Change'] >= 0 else 'down' }'>{row['Ticker']}: ${row['Price']:.2f} ({row['Change']:+.2f})</span>",
        axis=1,
    )

    ticker_html = (
        "<div class='ticker-tape'><div>"
        + " &nbsp; | &nbsp; ".join(stock_data["Display"])
        + "</div></div>"
    )
    st.markdown(ticker_html, unsafe_allow_html=True)

if __name__ == "__main__":
    display_ticker_tape()
