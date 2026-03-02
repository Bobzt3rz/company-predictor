"""
Compute Channel A financial ratios from raw Compustat quarterly data.

Input:  data/raw/compustat_fundq.parquet
Output: data/processed/channel_a_ratios.parquet

Ratios computed:
    1. gross_margin        = (revtq - cogsq) / revtq
    2. operating_margin    = oiadpq / revtq
    3. sga_to_revenue      = xsgaq / revtq
    4. inventory_turnover  = cogsq / invtq
    5. asset_turnover      = revtq / atq
    6. current_ratio       = actq / lctq
    7. ocf_margin          = oancfq / revtq

All ratios are winsorized at [1st, 99th] percentile per quarter to handle outliers.
Quarter-over-quarter deltas are also computed for each ratio.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99


def compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute financial ratios from raw Compustat variables."""
    out = df[["gvkey", "datadate", "fyearq", "fqtr"]].copy()

    # Core ratios
    out["gross_margin"] = (df["revtq"] - df["cogsq"]) / df["revtq"]
    out["operating_margin"] = df["oiadpq"] / df["revtq"]
    out["sga_to_revenue"] = df["xsgaq"] / df["revtq"]
    out["inventory_turnover"] = df["cogsq"] / df["invtq"]
    out["asset_turnover"] = df["revtq"] / df["atq"]
    out["current_ratio"] = df["actq"] / df["lctq"]

    # Replace infinities with NaN (division by zero cases)
    ratio_cols = [
        "gross_margin", "operating_margin", "sga_to_revenue",
        "inventory_turnover", "asset_turnover", "current_ratio",
    ]
    for col in ratio_cols:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out, ratio_cols


def winsorize_by_quarter(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Winsorize ratios at 1st/99th percentile within each calendar quarter."""
    df = df.copy()
    df["yq"] = df["fyearq"].astype(str) + "Q" + df["fqtr"].astype(str)

    for col in cols:
        bounds = df.groupby("yq")[col].quantile([WINSORIZE_LOWER, WINSORIZE_UPPER]).unstack()
        bounds.columns = ["lower", "upper"]
        df = df.merge(bounds, left_on="yq", right_index=True, how="left")
        df[col] = df[col].clip(lower=df["lower"], upper=df["upper"])
        df = df.drop(columns=["lower", "upper"])

    df = df.drop(columns=["yq"])
    return df


def compute_deltas(df: pd.DataFrame, ratio_cols: list[str]) -> pd.DataFrame:
    """Compute quarter-over-quarter changes for each ratio, per company."""
    df = df.sort_values(["gvkey", "datadate"]).copy()

    for col in ratio_cols:
        delta_col = f"{col}_delta"
        df[delta_col] = df.groupby("gvkey")[col].diff()

    return df


def compute_yoy_changes(df: pd.DataFrame, ratio_cols: list[str]) -> pd.DataFrame:
    """Compute year-over-year changes (vs same quarter last year) to handle seasonality."""
    df = df.sort_values(["gvkey", "datadate"]).copy()

    for col in ratio_cols:
        yoy_col = f"{col}_yoy"
        # Shift by 4 quarters within each company
        df[yoy_col] = df.groupby("gvkey")[col].diff(4)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction targets:
        - target_gross_margin: next quarter's gross margin (level)
        - target_gm_direction: 1 if next quarter GM > current, else 0 (for directional accuracy)
    """
    df = df.sort_values(["gvkey", "datadate"]).copy()

    # Next quarter's gross margin (shift -1 within each company)
    df["target_gross_margin"] = df.groupby("gvkey")["gross_margin"].shift(-1)

    # Direction: did margin go up?
    gm_diff = (df["target_gross_margin"] - df["gross_margin"]).to_numpy(dtype=float, na_value=np.nan)
    df["target_gm_direction"] = np.where(
        np.isnan(gm_diff), np.nan,
        np.where(gm_diff > 0, 1.0, 0.0)
    )

    return df


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    fundq = pd.read_parquet(RAW_DIR / "compustat_fundq.parquet")
    # WRDS uses pandas nullable dtypes (pd.NA) — convert to standard numpy floats
    for col in fundq.columns:
        if pd.api.types.is_numeric_dtype(fundq[col]):
            fundq[col] = fundq[col].astype(float)
    print(f"  {len(fundq)} rows, {fundq['gvkey'].nunique()} companies")

    print("Computing ratios...")
    ratios, ratio_cols = compute_ratios(fundq)

    print("Winsorizing...")
    ratios = winsorize_by_quarter(ratios, ratio_cols)

    print("Computing QoQ deltas...")
    ratios = compute_deltas(ratios, ratio_cols)

    print("Computing YoY changes...")
    ratios = compute_yoy_changes(ratios, ratio_cols)

    print("Creating targets...")
    ratios = create_target(ratios)

    # Summary
    print(f"\nOutput shape: {ratios.shape}")
    print(f"Feature columns: {len([c for c in ratios.columns if c not in ['gvkey', 'datadate', 'fyearq', 'fqtr']])}")
    print(f"\nTarget coverage:")
    print(f"  target_gross_margin non-null: {ratios['target_gross_margin'].notna().sum()}")
    dir_counts = ratios["target_gm_direction"].value_counts(dropna=False)
    print(f"  target_gm_direction distribution:")
    print(f"    Up   (1.0): {dir_counts.get(1.0, 0)}")
    print(f"    Down (0.0): {dir_counts.get(0.0, 0)}")
    print(f"    NaN:        {ratios['target_gm_direction'].isna().sum()}")

    print(f"\nNull rates for base ratios:")
    for col in ratio_cols:
        null_pct = ratios[col].isna().mean() * 100
        print(f"  {col:25s}: {null_pct:5.1f}%")

    # Save
    ratios.to_parquet(PROC_DIR / "channel_a_ratios.parquet", index=False)
    print(f"\nSaved to {PROC_DIR / 'channel_a_ratios.parquet'}")
    print("Done.")


if __name__ == "__main__":
    main()