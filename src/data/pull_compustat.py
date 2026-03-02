"""
Pull quarterly fundamentals from WRDS Compustat for Consumer Staples (GICS 30).

Usage:
    python src/data/pull_compustat.py

First run will prompt for WRDS credentials (cached in ~/.pgpass after).

Output:
    data/raw/compustat_fundq.parquet — raw quarterly fundamentals
    data/raw/company_universe.parquet — company metadata (gvkey, name, ticker, GICS)
"""

import os
import sys
import wrds
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GICS_SECTOR = "30"  # Consumer Staples
START_YEAR = 2004
END_YEAR = 2024
MIN_QUARTERS = 40  # require at least 10 years of data to keep a company

# Raw Compustat variables we need for ratio computation
COMPUSTAT_VARS = [
    "gvkey",        # company identifier
    "datadate",     # fiscal quarter end date
    "fyearq",       # fiscal year
    "fqtr",         # fiscal quarter (1-4)
    "rdq",          # report date of quarterly earnings
    # Income statement
    "revtq",        # revenue (total) - quarterly
    "cogsq",        # cost of goods sold - quarterly
    "xsgaq",        # SG&A expense - quarterly
    "oiadpq",       # operating income after depreciation - quarterly
    # Balance sheet
    "actq",         # current assets (total)
    "lctq",         # current liabilities (total)
    "invtq",        # inventories (total)
    "atq",          # assets (total)
    "ltq",          # liabilities (total)
    "seqq",         # stockholders equity (total)
    # Cash flow (note: quarterly cash flow items are limited in fundq;
    # oancfq/capxq don't exist — only YTD versions like oancfy/capxy)
]

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def pull_universe(db: wrds.Connection) -> pd.DataFrame:
    """Pull Consumer Staples companies from Compustat company table."""
    query = f"""
        SELECT DISTINCT
            a.gvkey,
            a.conm AS company_name,
            a.gsector,
            a.gind,
            a.gsubind
        FROM comp.company AS a
        WHERE a.gsector = '{GICS_SECTOR}'
    """
    df = db.raw_sql(query)
    print(f"[Universe] Found {len(df)} Consumer Staples companies in Compustat.")
    return df


def pull_fundamentals(db: wrds.Connection, gvkeys: list[str]) -> pd.DataFrame:
    """Pull quarterly fundamentals for the given gvkeys."""
    vars_str = ", ".join(COMPUSTAT_VARS)

    # wrds parameterized queries don't handle large IN lists well,
    # so we chunk if needed. For ~200 companies this is fine in one go.
    gvkey_str = "', '".join(gvkeys)

    query = f"""
        SELECT {vars_str}
        FROM comp.fundq
        WHERE gvkey IN ('{gvkey_str}')
          AND fyearq >= {START_YEAR}
          AND fyearq <= {END_YEAR}
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
        ORDER BY gvkey, datadate
    """
    df = db.raw_sql(query)
    print(f"[Fundamentals] Pulled {len(df)} company-quarter rows.")
    return df


def filter_universe(fundq: pd.DataFrame) -> pd.DataFrame:
    """Keep only companies with sufficient history and non-null revenue."""
    # Count non-null revenue quarters per company
    coverage = (
        fundq.dropna(subset=["revtq"])
        .groupby("gvkey")
        .size()
        .reset_index(name="n_quarters")
    )
    valid = coverage[coverage["n_quarters"] >= MIN_QUARTERS]["gvkey"]
    filtered = fundq[fundq["gvkey"].isin(valid)].copy()
    n_companies = filtered["gvkey"].nunique()
    print(f"[Filter] {n_companies} companies with >= {MIN_QUARTERS} quarters of revenue data.")
    return filtered


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to WRDS...")
    # Reads from environment variables to avoid repeated prompts.
    # Set these in your shell:
    #   export WRDS_USERNAME=your_username
    #   export WRDS_PASSWORD=your_password
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")

    if username and password:
        db = wrds.Connection(wrds_username=username, wrds_password=password)
    else:
        print("Tip: set WRDS_USERNAME and WRDS_PASSWORD env vars to skip login prompts.")
        db = wrds.Connection()

    # Step 1: Get universe
    universe = pull_universe(db)

    # Step 2: Pull fundamentals
    fundq = pull_fundamentals(db, universe["gvkey"].tolist())

    # Step 3: Filter for data quality
    fundq = filter_universe(fundq)

    # Update universe to match filtered companies
    valid_gvkeys = fundq["gvkey"].unique()
    universe = universe[universe["gvkey"].isin(valid_gvkeys)].copy()

    # Step 4: Save
    fundq.to_parquet(RAW_DIR / "compustat_fundq.parquet", index=False)
    universe.to_parquet(RAW_DIR / "company_universe.parquet", index=False)

    print(f"\nSaved to {RAW_DIR}/")
    print(f"  compustat_fundq.parquet  — {len(fundq)} rows, {fundq['gvkey'].nunique()} companies")
    print(f"  company_universe.parquet — {len(universe)} companies")

    # Quick summary
    print("\n--- Data Summary ---")
    print(f"Date range: {fundq['datadate'].min()} to {fundq['datadate'].max()}")
    print(f"Companies: {fundq['gvkey'].nunique()}")
    print(f"Quarters per company (median): {fundq.groupby('gvkey').size().median():.0f}")
    print(f"\nNull rates:")
    for col in ["revtq", "cogsq", "xsgaq", "oiadpq", "invtq", "atq"]:
        null_pct = fundq[col].isna().mean() * 100
        print(f"  {col:10s}: {null_pct:5.1f}%")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()