"""
Compute Channel A financial ratios from raw Compustat quarterly data.

Input:  data/raw/compustat_fundq.parquet
Output: data/processed/channel_a_ratios.parquet

Uses ratio definitions from src.schema — all column names and metadata
are defined centrally there.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path so we can import schema
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import (
    Ratios,
    Targets,
    WinsorizeBounds,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

BOUNDS = WinsorizeBounds()


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial ratios from raw Compustat variables.

    Ratio formulas are documented in schema.RATIO_DEFS. The actual
    computation is done here because some ratios have compound numerators
    (e.g. gross_margin = (revtq - cogsq) / revtq) that can't be expressed
    as a simple column / column lookup.
    """
    out = df[["gvkey", "datadate", "fyearq", "fqtr"]].copy()

    # Each ratio is computed explicitly so we can handle compound numerators.
    # The mapping from ratio name -> formula is documented in schema.RatioDef.
    out["gross_margin"] = (df["revtq"] - df["cogsq"]) / df["revtq"]
    out["operating_margin"] = df["oiadpq"] / df["revtq"]
    out["sga_to_revenue"] = df["xsgaq"] / df["revtq"]
    out["inventory_turnover"] = df["cogsq"] / df["invtq"]
    out["asset_turnover"] = df["revtq"] / df["atq"]
    out["current_ratio"] = df["actq"] / df["lctq"]

    # Replace infinities with NaN (division by zero cases)
    ratio_names = Ratios.names()
    for col in ratio_names:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out


def winsorize_by_quarter(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorize ratios at configured percentile bounds within each calendar quarter."""
    df = df.copy()
    df["yq"] = df["fyearq"].astype("Int64").astype(str) + "Q" + df["fqtr"].astype("Int64").astype(str)

    for col in Ratios.names():
        bounds = df.groupby("yq")[col].quantile([BOUNDS.lower, BOUNDS.upper]).unstack()
        bounds.columns = ["lower", "upper"]
        df = df.merge(bounds, left_on="yq", right_index=True, how="left")
        df[col] = df[col].clip(lower=df["lower"], upper=df["upper"])
        df = df.drop(columns=["lower", "upper"])

    df = df.drop(columns=["yq"])
    return df


def compute_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute quarter-over-quarter changes for each ratio, per company."""
    df = df.sort_values(["gvkey", "datadate"]).copy()

    for ratio in Ratios.all():
        df[ratio.delta_name] = df.groupby("gvkey")[ratio.name].diff()

    return df


def compute_yoy_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute year-over-year changes (vs same quarter last year) to handle seasonality."""
    df = df.sort_values(["gvkey", "datadate"]).copy()

    for ratio in Ratios.all():
        df[ratio.yoy_name] = df.groupby("gvkey")[ratio.name].diff(4)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction targets:
        - target_gross_margin: next quarter's gross margin (level)
        - target_gm_delta: next quarter's GM minus current GM (change)
        - target_gm_direction: 1 if next quarter GM > current, else 0
    """
    df = df.sort_values(["gvkey", "datadate"]).copy()

    df[Targets.GROSS_MARGIN] = df.groupby("gvkey")["gross_margin"].shift(-1)

    # Delta: next quarter GM - current GM
    df[Targets.GM_DELTA] = df[Targets.GROSS_MARGIN] - df["gross_margin"]

    # Direction: did margin go up?
    gm_diff = df[Targets.GM_DELTA].to_numpy(dtype=float)
    df[Targets.GM_DIRECTION] = np.where(
        np.isnan(gm_diff), np.nan,
        np.where(gm_diff > 0, 1.0, 0.0)
    )

    return df


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    fundq = pd.read_parquet(RAW_DIR / "compustat_fundq.parquet")
    for col in fundq.columns:
        if pd.api.types.is_numeric_dtype(fundq[col]):
            fundq[col] = fundq[col].astype(float)
    print(f"  {len(fundq)} rows, {fundq['gvkey'].nunique()} companies")

    print("Computing ratios...")
    ratios = compute_ratios(fundq)

    print("Winsorizing...")
    ratios = winsorize_by_quarter(ratios)

    print("Computing QoQ deltas...")
    ratios = compute_deltas(ratios)

    print("Computing YoY changes...")
    ratios = compute_yoy_changes(ratios)

    print("Creating targets...")
    ratios = create_target(ratios)

    # Summary
    ratio_names = Ratios.names()
    print(f"\nOutput shape: {ratios.shape}")
    print(f"Feature columns: {len([c for c in ratios.columns if c not in ['gvkey', 'datadate', 'fyearq', 'fqtr']])}")
    print(f"\nTarget coverage:")
    print(f"  {Targets.GROSS_MARGIN} non-null: {ratios[Targets.GROSS_MARGIN].notna().sum()}")
    dir_counts = ratios[Targets.GM_DIRECTION].value_counts(dropna=False)
    print(f"  {Targets.GM_DIRECTION} distribution:")
    print(f"    Up   (1.0): {dir_counts.get(1.0, 0)}")
    print(f"    Down (0.0): {dir_counts.get(0.0, 0)}")
    print(f"    NaN:        {ratios[Targets.GM_DIRECTION].isna().sum()}")

    print(f"\nNull rates for base ratios:")
    for name in ratio_names:
        null_pct = ratios[name].isna().mean() * 100
        print(f"  {name:25s}: {null_pct:5.1f}%")

    ratios.to_parquet(PROC_DIR / "channel_a_ratios.parquet", index=False)
    print(f"\nSaved to {PROC_DIR / 'channel_a_ratios.parquet'}")
    print("Done.")


if __name__ == "__main__":
    main()