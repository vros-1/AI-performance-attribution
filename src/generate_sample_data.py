from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

# ------------ CONFIG -------------
START = "2023-01-01"
END = "2024-01-01"

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
# ---------------------------------


def make_sector_level_data():
    """
    Use SPDR sector ETFs as sector proxies and build
    portfolio vs benchmark returns + weights.
    """
    sector_map = {
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLE": "Energy",
        "XLF": "Financials",
        "XLV": "Health Care",
        "XLI": "Industrials",
        "XLB": "Materials",
        "XLK": "Information Technology",
        "XLU": "Utilities",
        "XLRE": "Real Estate",
    }

    tickers = list(sector_map.keys())

    # Download adjusted close prices
    data = yf.download(
    tickers,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
    )
    # Use only the Close prices; this collapses the MultiIndex to tickers as columns
    prices = data["Close"].dropna()
    rets = prices.pct_change().dropna()

    # Create a simple benchmark weight scheme (fixed weights)
    # You can tweak these if you want something more realistic
    n = len(tickers)
    base_weights = np.array([1 / n] * n)  # equal-weight benchmark

    # Make a small active tilt for the portfolio
    rng = np.random.default_rng(42)
    active_tilts = rng.normal(loc=0.0, scale=0.02, size=n)  # +/- 2% tilts
    port_weights = base_weights + active_tilts
    # Normalize to sum to 1
    port_weights = port_weights / port_weights.sum()

    df_list = []

    for i, ticker in enumerate(tickers):
        sector_name = sector_map[ticker]
        df_tmp = pd.DataFrame({
            "date": rets.index,
            "sector": sector_name,
            "etf": ticker,
            "benchmark_return": rets[ticker].values,
            "portfolio_return": rets[ticker].values
        })
        df_tmp["benchmark_weight"] = base_weights[i]
        df_tmp["portfolio_weight"] = port_weights[i]
        df_list.append(df_tmp)

    df = pd.concat(df_list, ignore_index=True)

    # Simple attribution-style columns
    df["excess_return"] = df["portfolio_return"] - df["benchmark_return"]
    df["active_weight"] = df["portfolio_weight"] - df["benchmark_weight"]
    df["allocation_effect"] = df["active_weight"] * df["benchmark_return"]
    df["selection_effect"] = df["benchmark_weight"] * (
        df["portfolio_return"] - df["benchmark_return"]
    )
    df["interaction_effect"] = df["active_weight"] * (
        df["portfolio_return"] - df["benchmark_return"]
    )

    out_path = DATA_DIR / "sector_attribution_sample.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved sector-level sample to: {out_path.resolve()}")


def make_security_level_data():
    """
    Use a handful of large-cap stocks as the portfolio
    and SPY as the benchmark.
    """
    portfolio_tickers = ["AAPL", "MSFT", "AMZN", "TSLA", "JNJ", "XOM", "JPM", "NVDA"]
    benchmark_ticker = "SPY"

    all_tickers = portfolio_tickers + [benchmark_ticker]

    data = yf.download(
    all_tickers,
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
    )
    prices = data["Close"].dropna()
    rets = prices.pct_change().dropna()
    # Build some simple, static portfolio weights
    n = len(portfolio_tickers)
    base_port_weights = np.array([1 / n] * n)

    # Small random active tilts at security level
    rng = np.random.default_rng(7)
    tilts = rng.normal(loc=0.0, scale=0.01, size=n)  # +/- 1% tilts
    port_weights = base_port_weights + tilts
    port_weights = port_weights / port_weights.sum()

    # Compute total portfolio and benchmark returns by date
    port_rets = (rets[portfolio_tickers] * port_weights).sum(axis=1)
    bench_rets = rets[benchmark_ticker]

    # Long-form security-level table
    rows = []
    for i, ticker in enumerate(portfolio_tickers):
        df_tmp = pd.DataFrame({
            "date": rets.index,
            "security": ticker,
            "security_return": rets[ticker].values,
        })
        df_tmp["portfolio_weight"] = port_weights[i]
        # For simplicity, treat benchmark weight as equal weight in that subset
        df_tmp["benchmark_weight"] = base_port_weights[i]
        rows.append(df_tmp)

    df_sec = pd.concat(rows, ignore_index=True)

    # Attach portfolio & benchmark-level returns to every row (for attribution math)
    date_to_port_ret = port_rets.to_dict()
    date_to_bench_ret = bench_rets.to_dict()

    df_sec["portfolio_return_total"] = df_sec["date"].map(date_to_port_ret)
    df_sec["benchmark_return_total"] = df_sec["date"].map(date_to_bench_ret)

    # Simple “security-level” effects (you can refine later)
    df_sec["excess_return_total"] = (
        df_sec["portfolio_return_total"] - df_sec["benchmark_return_total"]
    )
    df_sec["active_weight"] = df_sec["portfolio_weight"] - df_sec["benchmark_weight"]
    df_sec["selection_contribution"] = (
        df_sec["benchmark_weight"] * (df_sec["security_return"] - df_sec["benchmark_return_total"])
    )

    out_path = DATA_DIR / "security_attribution_sample.csv"
    df_sec.to_csv(out_path, index=False)
    print(f"Saved security-level sample to: {out_path.resolve()}")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    make_sector_level_data()
    make_security_level_data()
    