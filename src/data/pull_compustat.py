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
import wrds
import pandas as pd
from pathlib import Path

# Add project root to path so we can import schema
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import CompustatVar, CompustatConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = CompustatConfig()
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
        WHERE a.gsector = '{CONFIG.gics_sector}'
    """
    df = db.raw_sql(query)
    print(f"[Universe] Found {len(df)} Consumer Staples companies in Compustat.")
    return df


def pull_fundamentals(db: wrds.Connection, gvkeys: list[str]) -> pd.DataFrame:
    """Pull quarterly fundamentals for the given gvkeys."""
    vars_str = ", ".join(CompustatVar.all_vars())
    gvkey_str = "', '".join(gvkeys)

    query = f"""
        SELECT {vars_str}
        FROM comp.fundq
        WHERE gvkey IN ('{gvkey_str}')
          AND fyearq >= {CONFIG.start_year}
          AND fyearq <= {CONFIG.end_year}
          AND indfmt = '{CONFIG.ind_fmt}'
          AND datafmt = '{CONFIG.data_fmt}'
          AND popsrc = '{CONFIG.pop_src}'
          AND consol = '{CONFIG.consol}'
        ORDER BY gvkey, datadate
    """
    df = db.raw_sql(query)
    print(f"[Fundamentals] Pulled {len(df)} company-quarter rows.")
    return df


def filter_universe(fundq: pd.DataFrame) -> pd.DataFrame:
    """Keep only companies with sufficient history and non-null revenue."""
    coverage = (
        fundq.dropna(subset=[CompustatVar.REVTQ])
        .groupby(CompustatVar.GVKEY)
        .size()
        .reset_index(name="n_quarters")
    )
    valid = coverage[coverage["n_quarters"] >= CONFIG.min_quarters][CompustatVar.GVKEY]
    filtered = fundq[fundq[CompustatVar.GVKEY].isin(valid)].copy()
    n_companies = filtered[CompustatVar.GVKEY].nunique()
    print(f"[Filter] {n_companies} companies with >= {CONFIG.min_quarters} quarters of revenue data.")
    return filtered


def apply_revenue_floor(fundq: pd.DataFrame) -> pd.DataFrame:
    """Drop company-quarters with revenue below the configured minimum.

    This prevents near-zero denominators from producing extreme ratio values
    (e.g. operating margin of -88x when revenue is $0.01M).
    """
    before = len(fundq)
    fundq = fundq[
        fundq[CompustatVar.REVTQ].isna() | (fundq[CompustatVar.REVTQ] >= CONFIG.min_revenue)
    ].copy()
    dropped = before - len(fundq)
    print(f"[Revenue floor] Dropped {dropped:,} rows with revenue < ${CONFIG.min_revenue}M "
          f"({dropped / before * 100:.1f}%)")
    return fundq


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to WRDS...")
    username = os.environ.get("WRDS_USERNAME")
    password = os.environ.get("WRDS_PASSWORD")

    if username and password:
        db = wrds.Connection(wrds_username=username, wrds_password=password)
    else:
        print("Tip: set WRDS_USERNAME and WRDS_PASSWORD env vars to skip login prompts.")
        db = wrds.Connection()

    universe = pull_universe(db)
    fundq = pull_fundamentals(db, universe[CompustatVar.GVKEY].tolist())
    fundq = apply_revenue_floor(fundq)
    fundq = filter_universe(fundq)

    valid_gvkeys = fundq[CompustatVar.GVKEY].unique()
    universe = universe[universe[CompustatVar.GVKEY].isin(valid_gvkeys)].copy()

    fundq.to_parquet(RAW_DIR / "compustat_fundq.parquet", index=False)
    universe.to_parquet(RAW_DIR / "company_universe.parquet", index=False)

    print(f"\nSaved to {RAW_DIR}/")
    print(f"  compustat_fundq.parquet  — {len(fundq)} rows, {fundq[CompustatVar.GVKEY].nunique()} companies")
    print(f"  company_universe.parquet — {len(universe)} companies")

    print("\n--- Data Summary ---")
    print(f"Date range: {fundq[CompustatVar.DATADATE].min()} to {fundq[CompustatVar.DATADATE].max()}")
    print(f"Companies: {fundq[CompustatVar.GVKEY].nunique()}")
    print(f"Quarters per company (median): {fundq.groupby(CompustatVar.GVKEY).size().median():.0f}")
    print(f"\nNull rates:")
    for col in CompustatVar.all_numeric():
        null_pct = fundq[col].isna().mean() * 100
        print(f"  {col:10s}: {null_pct:5.1f}%")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()