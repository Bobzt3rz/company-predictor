"""
Build sliding-window datasets from multiple feature channels for sequence models.

Input:
    data/processed/channel_a_ratios.parquet      — per-company financial ratios
    data/processed/channel_b_commodities.parquet  — market-wide commodity features
    (future) data/processed/channel_c_*.parquet   — additional channels

Output:
    data/processed/windows/
        train.npz    — X_train (N, window_size, num_features), y_train (N,)
        val.npz      — X_val, y_val
        test.npz     — X_test, y_test
        metadata.npz — feature_names, scaler params, split info, channel_slices

Each sample is a (window_size, num_features) matrix of sequential quarters
for a single company, paired with the next quarter's gross margin delta as
the target.

Channel architecture:
    Channels are categorized by their join granularity:
      - "company": per-company-per-quarter (e.g. financial ratios, SEC filings)
                   joined on (gvkey, fyearq, fqtr)
      - "market":  per-quarter only (e.g. commodity prices, macro indicators)
                   broadcast-joined on (fyearq, fqtr) so all companies share
                   the same values for a given quarter

    Adding a new channel:
      1. Write a feature script that outputs data/processed/channel_X_*.parquet
         with columns: [fyearq, fqtr, ...features...] (+ gvkey for company-level)
      2. Register a ChannelSource in CHANNEL_REGISTRY below
      3. Re-run this script — the new features flow through automatically

Temporal split ensures no data leakage:
    train: target quarter <= train_end_year
    val:   train_end_year < target quarter <= val_end_year
    test:  target quarter > val_end_year
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import (
    Ratios,
    Commodities,
    Targets,
    WindowConfig,
    feature_columns,
)

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
WINDOW_DIR = PROC_DIR / "windows"

CONFIG = WindowConfig()


# ---------------------------------------------------------------------------
# Channel registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChannelSource:
    """
    Describes how to load and merge one feature channel.

    Attributes:
        name:          Channel identifier (e.g. "a", "b", "c")
        parquet_path:  Path to the processed parquet file
        feature_names: List of feature column names to use from the file
        granularity:   "company" → join on (gvkey, fyearq, fqtr)
                       "market"  → broadcast-join on (fyearq, fqtr)
    """
    name: str
    parquet_path: Path
    feature_names: list[str]
    granularity: str  # "company" or "market"

    def __post_init__(self):
        if self.granularity not in ("company", "market"):
            raise ValueError(f"granularity must be 'company' or 'market', got '{self.granularity}'")


# Register all available channels here.
# The window builder will load whichever ones exist on disk.
CHANNEL_REGISTRY: tuple[ChannelSource, ...] = (
    ChannelSource(
        name="a",
        parquet_path=PROC_DIR / "channel_a_ratios.parquet",
        feature_names=Ratios.names() + Ratios.delta_names() + Ratios.yoy_names(),
        granularity="company",
    ),
    ChannelSource(
        name="b",
        parquet_path=PROC_DIR / "channel_b_commodities.parquet",
        feature_names=Commodities.feature_names(),
        granularity="market",
    ),
    # Future channels:
    # ChannelSource(
    #     name="c",
    #     parquet_path=PROC_DIR / "channel_c_sec_filings.parquet",
    #     feature_names=SecFilings.feature_names(),
    #     granularity="company",
    # ),
)


# ---------------------------------------------------------------------------
# Data loading and merging
# ---------------------------------------------------------------------------

def load_channel(source: ChannelSource) -> pd.DataFrame | None:
    """Load a single channel's parquet file, or return None if missing."""
    if not source.parquet_path.exists():
        print(f"  [Channel {source.name}] {source.parquet_path.name} not found — skipping")
        return None

    df = pd.read_parquet(source.parquet_path)

    # Validate expected feature columns exist
    available = [c for c in source.feature_names if c in df.columns]
    missing = [c for c in source.feature_names if c not in df.columns]
    if missing:
        print(f"  [Channel {source.name}] WARNING: missing columns: {missing}")

    print(f"  [Channel {source.name}] Loaded {source.parquet_path.name}: "
          f"{len(df):,} rows, {len(available)} features ({source.granularity})")

    return df


def merge_channels(
    channels: list[tuple[ChannelSource, pd.DataFrame]],
) -> tuple[pd.DataFrame, list[str], dict[str, tuple[int, int]]]:
    """
    Merge multiple channels into a single DataFrame.

    Company-level channels are joined on (gvkey, fyearq, fqtr).
    Market-level channels are broadcast-joined on (fyearq, fqtr).

    Returns:
        merged:        Combined DataFrame with all features
        feature_names: Ordered list of all feature column names
        channel_slices: Dict mapping channel name → (start_idx, end_idx) into
                        the feature array, for downstream channel-aware models
    """
    if not channels:
        raise ValueError("No channels to merge")

    # Start with the first company-level channel as the base
    # (it defines the universe of gvkey × quarter combinations)
    base_source, base_df = None, None
    for source, df in channels:
        if source.granularity == "company":
            base_source, base_df = source, df
            break

    if base_df is None:
        raise ValueError("At least one company-level channel is required as the base")

    # Ensure consistent types for join keys
    id_cols = ["gvkey", "fyearq", "fqtr"]
    merged = base_df.copy()
    for col in ["fyearq", "fqtr"]:
        merged[col] = merged[col].astype(float)

    # Track feature columns and their channel slices
    all_features = []
    channel_slices = {}

    for source, df in channels:
        available = [c for c in source.feature_names if c in df.columns]

        if source is base_source:
            # Base channel — already in merged, just record features
            start_idx = len(all_features)
            all_features.extend(available)
            channel_slices[source.name] = (start_idx, len(all_features))
            continue

        # Prepare join keys
        df = df.copy()
        for col in ["fyearq", "fqtr"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        if source.granularity == "company":
            join_keys = ["gvkey", "fyearq", "fqtr"]
        else:
            join_keys = ["fyearq", "fqtr"]

        # Only keep join keys + feature columns to avoid collisions
        keep_cols = [c for c in join_keys if c in df.columns] + available
        df = df[keep_cols].drop_duplicates(subset=[c for c in join_keys if c in df.columns])

        before = len(merged)
        merged = merged.merge(df, on=[c for c in join_keys if c in df.columns], how="left")
        after = len(merged)

        if after != before:
            print(f"  WARNING: merge with channel {source.name} changed row count "
                  f"({before:,} → {after:,})")

        start_idx = len(all_features)
        all_features.extend(available)
        channel_slices[source.name] = (start_idx, len(all_features))

    print(f"\nMerged dataset: {len(merged):,} rows, {len(all_features)} features")
    for name, (start, end) in channel_slices.items():
        print(f"  Channel {name}: features [{start}:{end}] ({end - start} cols)")

    return merged, all_features, channel_slices


# ---------------------------------------------------------------------------
# Windowing, splitting, normalizing (unchanged logic, cleaner interface)
# ---------------------------------------------------------------------------

@dataclass
class WindowDataset:
    """Container for a train/val/test split of windowed data."""
    X_train: np.ndarray       # (N_train, window_size, num_features)
    y_train: np.ndarray       # (N_train,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_mean: np.ndarray    # (num_features,) — for normalizing at inference
    train_std: np.ndarray     # (num_features,)
    channel_slices: dict[str, tuple[int, int]] = field(default_factory=dict)


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from the dataframe.

    Returns:
        X:           (N, window_size, num_features)
        y:           (N,)
        target_year: (N,) — fiscal year of the target quarter (for splitting)
        gvkeys:      (N,) — company ID for each window
    """
    X_windows = []
    y_targets = []
    target_years = []
    gvkeys = []

    for gvkey, gdf in df.groupby("gvkey"):
        gdf = gdf.sort_values("datadate").reset_index(drop=True)
        n = len(gdf)

        if n < window_size + 1:
            continue

        features = gdf[feature_cols].values        # (n, num_features)
        targets = gdf[target_col].values            # (n,)
        years = gdf["fyearq"].values                # (n,)

        for i in range(n - window_size):
            window = features[i : i + window_size]  # (window_size, num_features)
            target = targets[i + window_size - 1]   # target for the last row
            target_year = years[i + window_size]     # year of the prediction quarter

            # Skip if any NaN in window or target
            if np.isnan(window).any() or np.isnan(target):
                continue

            X_windows.append(window)
            y_targets.append(target)
            target_years.append(target_year)
            gvkeys.append(gvkey)

    X = np.array(X_windows, dtype=np.float32)
    y = np.array(y_targets, dtype=np.float32)
    target_years = np.array(target_years, dtype=np.float32)
    gvkeys = np.array(gvkeys)

    return X, y, target_years, gvkeys


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    target_years: np.ndarray,
    gvkeys: np.ndarray,
    train_end: int,
    val_end: int,
) -> dict:
    """Split into train/val/test by target year."""
    train_mask = target_years <= train_end
    val_mask = (target_years > train_end) & (target_years <= val_end)
    test_mask = target_years > val_end

    return {
        "train": (X[train_mask], y[train_mask], gvkeys[train_mask]),
        "val":   (X[val_mask],   y[val_mask],   gvkeys[val_mask]),
        "test":  (X[test_mask],  y[test_mask],  gvkeys[test_mask]),
    }


def normalize_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize using training set statistics only.

    Computed across all samples and timesteps in training data,
    per feature. This prevents data leakage from val/test.
    """
    flat = X_train.reshape(-1, X_train.shape[2])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    # Avoid division by zero for constant features
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_dataset(
    config: WindowConfig | None = None,
    channels: tuple[str, ...] | None = None,
) -> WindowDataset:
    """
    Full pipeline: load channels → merge → window → split → normalize.

    Args:
        config:   Window configuration (defaults to CONFIG)
        channels: Tuple of channel names to include, e.g. ("a",) or ("a", "b").
                  If None, includes all channels found on disk.

    Returns:
        WindowDataset with everything needed for training.
    """
    if config is None:
        config = CONFIG

    # --- Load and merge channels ---
    print("Loading feature channels...")
    loaded = []
    for source in CHANNEL_REGISTRY:
        if channels is not None and source.name not in channels:
            continue
        df = load_channel(source)
        if df is not None:
            loaded.append((source, df))

    if not loaded:
        raise FileNotFoundError(
            "No channel data found. Run the feature pipeline first:\n"
            "  python src/features/compute_ratios.py\n"
            "  python src/data/pull_commodities.py"
        )

    merged, feat_cols, channel_slices = merge_channels(loaded)

    # Sort by company + date for windowing
    merged = merged.dropna(subset=["fyearq", "fqtr"])
    merged = merged.sort_values(["gvkey", "datadate"]).reset_index(drop=True)

    target_col = Targets.GM_DELTA
    print(f"\n{len(feat_cols)} features from {len(loaded)} channels, target: {target_col}")

    # --- Build windows ---
    print(f"\nBuilding windows (size={config.window_size})...")
    X, y, target_years, gvkeys = build_windows(
        merged, feat_cols, target_col, config.window_size
    )
    print(f"  {len(X):,} clean windows from {len(np.unique(gvkeys))} companies")

    # --- Temporal split ---
    print(f"\nTemporal split (train <= {config.train_end_year}, "
          f"val <= {config.val_end_year}, test > {config.val_end_year})...")
    splits = temporal_split(X, y, target_years, gvkeys,
                            config.train_end_year, config.val_end_year)

    X_train, y_train, gvkeys_train = splits["train"]
    X_val, y_val, gvkeys_val = splits["val"]
    X_test, y_test, gvkeys_test = splits["test"]

    print(f"  Train: {len(X_train):,} windows, {len(np.unique(gvkeys_train))} companies")
    print(f"  Val:   {len(X_val):,} windows, {len(np.unique(gvkeys_val))} companies")
    print(f"  Test:  {len(X_test):,} windows, {len(np.unique(gvkeys_test))} companies")

    # --- Normalize ---
    if config.normalize:
        print("\nNormalizing features (z-score from training set)...")
        X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)
    else:
        flat = X_train.reshape(-1, X_train.shape[2])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)

    # --- Stats ---
    print(f"\nTarget statistics:")
    print(f"  Train — mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  Val   — mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    print(f"  Test  — mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")

    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name} samples: {len(ys):,}")

    return WindowDataset(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        feature_names=feat_cols,
        train_mean=mean,
        train_std=std,
        channel_slices=channel_slices,
    )


def save_dataset(ds: WindowDataset, output_dir: Path | None = None):
    """Save windowed dataset to .npz files."""
    if output_dir is None:
        output_dir = WINDOW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_dir / "train.npz", X=ds.X_train, y=ds.y_train)
    np.savez_compressed(output_dir / "val.npz", X=ds.X_val, y=ds.y_val)
    np.savez_compressed(output_dir / "test.npz", X=ds.X_test, y=ds.y_test)

    # Save metadata including channel slice info
    np.savez(
        output_dir / "metadata.npz",
        feature_names=np.array(ds.feature_names),
        train_mean=ds.train_mean,
        train_std=ds.train_std,
        # Store channel slices as a JSON-like string for easy recovery
        channel_slices_keys=np.array(list(ds.channel_slices.keys())),
        channel_slices_starts=np.array([v[0] for v in ds.channel_slices.values()]),
        channel_slices_ends=np.array([v[1] for v in ds.channel_slices.values()]),
    )

    print(f"\nSaved to {output_dir}/")
    print(f"  train.npz  — X: {ds.X_train.shape}, y: {ds.y_train.shape}")
    print(f"  val.npz    — X: {ds.X_val.shape},   y: {ds.y_val.shape}")
    print(f"  test.npz   — X: {ds.X_test.shape},  y: {ds.y_test.shape}")
    print(f"  metadata.npz — {len(ds.feature_names)} features, "
          f"{len(ds.channel_slices)} channels, scaler params")


def load_saved_dataset(input_dir: Path | None = None) -> WindowDataset:
    """Load a previously saved windowed dataset."""
    if input_dir is None:
        input_dir = WINDOW_DIR

    train = np.load(input_dir / "train.npz")
    val = np.load(input_dir / "val.npz")
    test = np.load(input_dir / "test.npz")
    meta = np.load(input_dir / "metadata.npz", allow_pickle=True)

    # Reconstruct channel slices
    channel_slices = {}
    if "channel_slices_keys" in meta:
        keys = meta["channel_slices_keys"].tolist()
        starts = meta["channel_slices_starts"].tolist()
        ends = meta["channel_slices_ends"].tolist()
        channel_slices = {k: (s, e) for k, s, e in zip(keys, starts, ends)}

    return WindowDataset(
        X_train=train["X"], y_train=train["y"],
        X_val=val["X"], y_val=val["y"],
        X_test=test["X"], y_test=test["y"],
        feature_names=meta["feature_names"].tolist(),
        train_mean=meta["train_mean"],
        train_std=meta["train_std"],
        channel_slices=channel_slices,
    )


def main():
    ds = build_dataset()
    save_dataset(ds)

    # Quick verification
    print(f"\n--- Verification ---")
    print(f"X_train shape: {ds.X_train.shape}  →  "
          f"(samples, {CONFIG.window_size} quarters, {len(ds.feature_names)} features)")
    print(f"y_train shape: {ds.y_train.shape}")
    print(f"Feature names: {ds.feature_names}")
    print(f"Channel slices: {ds.channel_slices}")

    for name, arr in [("X_train", ds.X_train), ("X_val", ds.X_val),
                       ("X_test", ds.X_test), ("y_train", ds.y_train),
                       ("y_val", ds.y_val), ("y_test", ds.y_test)]:
        n_nan = np.isnan(arr).sum()
        print(f"  {name} NaN count: {n_nan}")

    print("\nDone.")


if __name__ == "__main__":
    main()