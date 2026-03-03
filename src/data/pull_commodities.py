"""
Pull commodity price data from Yahoo Finance for Channel B features.

Usage:
    python src/data/pull_commodities.py

No API key required. Uses yfinance to pull daily futures prices for
commodities relevant to Consumer Staples, aggregates to quarterly,
and computes QoQ/YoY percentage changes.

Output:
    data/raw/yf_commodities_daily.parquet  — raw daily close prices
    data/processed/channel_b_commodities.parquet — quarterly features with changes
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import CompustatConfig, Commodities

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMPUSTAT_CONFIG = CompustatConfig()
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def pull_all_series(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Pull all commodity series from Yahoo Finance.

    Uses adjusted close prices from rolling front-month futures contracts.
    """
    if start_date is None:
        start_date = f"{COMPUSTAT_CONFIG.start_year - 1}-01-01"  # 1 year before for YoY
    if end_date is None:
        end_date = f"{COMPUSTAT_CONFIG.end_year}-12-31"

    commodities = Commodities.all()
    tickers = {c.name: c.yf_ticker for c in commodities}
    ticker_str = " ".join(tickers.values())

    print(f"Pulling {len(tickers)} series from Yahoo Finance ({start_date} to {end_date})...")
    for name, ticker in tickers.items():
        print(f"  {ticker:12s} → {name}")

    # Pull all tickers in one batch call
    raw = yf.download(
        ticker_str,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    # yf.download returns MultiIndex columns: (Price, Ticker)
    # Extract Close prices and rename to our commodity names
    closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    ticker_to_name = {v: k for k, v in tickers.items()}
    closes = closes.rename(columns=ticker_to_name)

    # Keep only our commodity columns (drop any that failed)
    available = [c for c in tickers.keys() if c in closes.columns]
    missing = [c for c in tickers.keys() if c not in closes.columns]
    if missing:
        print(f"\n  WARNING: Missing series: {missing}")

    df = closes[available].copy()
    df.index.name = "date"

    print(f"\nRaw daily data: {df.shape[0]} days × {df.shape[1]} columns")
    print(f"Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"\nNull rates (raw daily):")
    for col in df.columns:
        null_pct = df[col].isna().mean() * 100
        print(f"  {col:20s}: {null_pct:5.1f}%")

    return df


def aggregate_to_quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily data to quarterly.

    Strategy: take the last valid observation in each quarter (quarter-end close).
    This aligns with Compustat's quarterly reporting dates.
    """
    quarterly = df.resample("QE").last()

    # Add fiscal year/quarter columns to match Compustat convention
    quarterly["fyearq"] = quarterly.index.year
    quarterly["fqtr"] = quarterly.index.quarter
    quarterly["datadate"] = quarterly.index

    # Forward-fill at most 1 quarter for any gaps
    commodity_cols = [c for c in quarterly.columns if c not in ["fyearq", "fqtr", "datadate"]]
    quarterly[commodity_cols] = quarterly[commodity_cols].ffill(limit=1)

    print(f"\nQuarterly aggregation: {len(quarterly)} quarters")

    return quarterly


def compute_commodity_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute QoQ percentage changes for each commodity.

    Percentage changes (not absolute deltas) because commodity prices
    operate on very different scales. Only QoQ is computed — the LSTM
    sees 8 quarters of history so it can learn seasonal/YoY patterns
    directly from the QoQ sequence.
    """
    df = df.copy()
    commodity_cols = Commodities.names()
    commodity_cols = [c for c in commodity_cols if c in df.columns]

    for col in commodity_cols:
        df[f"{col}_qoq"] = df[col].pct_change(1, fill_method=None)

    # Replace infinities from division by zero
    change_cols = [c for c in df.columns if c.endswith("_qoq")]
    for col in change_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


def build_channel_b(df_quarterly: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final Channel B feature set.

    Output columns per commodity:
        {name}_qoq   — quarter-over-quarter % change
    """
    df = df_quarterly.copy()
    commodity_cols = [c for c in Commodities.names() if c in df.columns]

    # Build final feature set
    feature_cols = [f"{col}_qoq" for col in commodity_cols]

    id_cols = ["datadate", "fyearq", "fqtr"]
    keep_cols = [c for c in id_cols + feature_cols if c in df.columns]
    out = df[keep_cols].copy()

    print(f"\nChannel B features: {len(feature_cols)} columns")
    print(f"  {len(commodity_cols)} commodities × 1 feature (QoQ %)")
    print(f"\nNull rates (final):")
    for col in feature_cols:
        if col in out.columns:
            null_pct = out[col].isna().mean() * 100
            print(f"  {col:25s}: {null_pct:5.1f}%")

    return out


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Pull raw data
    raw_df = pull_all_series()

    # Save raw daily data
    raw_path = RAW_DIR / "yf_commodities_daily.parquet"
    raw_df.to_parquet(raw_path)
    print(f"\nSaved raw data: {raw_path}")

    # Aggregate to quarterly
    quarterly = aggregate_to_quarterly(raw_df)

    # Compute changes
    quarterly = compute_commodity_changes(quarterly)

    # Build final features
    channel_b = build_channel_b(quarterly)

    # Save processed
    out_path = PROC_DIR / "channel_b_commodities.parquet"
    channel_b.to_parquet(out_path, index=False)
    print(f"\nSaved processed data: {out_path}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Quarters: {len(channel_b)}")
    print(f"Date range: {channel_b['fyearq'].min()}Q{int(channel_b['fqtr'].min())} "
          f"to {channel_b['fyearq'].max()}Q{int(channel_b['fqtr'].max())}")
    print(f"Feature columns: {len([c for c in channel_b.columns if c not in ['datadate', 'fyearq', 'fqtr']])}")

    print("\nDone.")


if __name__ == "__main__":
    main()