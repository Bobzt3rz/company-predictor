"""
Build sliding-window datasets from Channel A ratios for sequence models.

Input:  data/processed/channel_a_ratios.parquet
Output: data/processed/windows/
            train.npz   — X_train (N, window_size, num_features), y_train (N,)
            val.npz     — X_val, y_val
            test.npz    — X_test, y_test
            metadata.npz — feature_names, scaler params, split info

Each sample is a (window_size, num_features) matrix of sequential quarters
for a single company, paired with the next quarter's gross margin delta as
the target.

Temporal split ensures no data leakage:
    train: target quarter <= train_end_year
    val:   train_end_year < target quarter <= val_end_year
    test:  target quarter > val_end_year
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import (
    Ratios,
    Targets,
    WindowConfig,
    feature_columns,
)

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
WINDOW_DIR = PROC_DIR / "windows"

CONFIG = WindowConfig()


@dataclass
class WindowDataset:
    """Container for a train/val/test split of windowed data."""
    X_train: np.ndarray   # (N_train, window_size, num_features)
    y_train: np.ndarray   # (N_train,)
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    train_mean: np.ndarray   # (num_features,) — for normalizing at inference
    train_std: np.ndarray    # (num_features,)


def load_ratios() -> pd.DataFrame:
    """Load processed ratios and sort by company + date."""
    path = PROC_DIR / "channel_a_ratios.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            "  python src/data/pull_compustat.py\n"
            "  python src/features/compute_ratios.py"
        )

    df = pd.read_parquet(path)
    df = df.dropna(subset=["fyearq", "fqtr"])
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    return df


def build_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sliding windows from the dataframe.

    Returns:
        X:          (N, window_size, num_features)
        y:          (N,)
        target_year: (N,) — fiscal year of the target quarter (for splitting)
        gvkeys:     (N,) — company ID for each window (for diagnostics)
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
            target = targets[i + window_size - 1]   # target for the last row in the window
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
    # Reshape to (N * window_size, num_features) to compute stats
    flat = X_train.reshape(-1, X_train.shape[2])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    # Avoid division by zero for constant features
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


def build_dataset(config: WindowConfig | None = None) -> WindowDataset:
    """
    Full pipeline: load → window → split → normalize.

    Returns a WindowDataset with everything needed for training.
    """
    if config is None:
        config = CONFIG

    print("Loading processed ratios...")
    df = load_ratios()
    print(f"  {len(df):,} rows, {df['gvkey'].nunique()} companies")

    # Use all available feature columns
    feat_cols = [c for c in feature_columns() if c in df.columns]
    target_col = Targets.GM_DELTA
    print(f"  {len(feat_cols)} features, target: {target_col}")

    print(f"\nBuilding windows (size={config.window_size})...")
    X, y, target_years, gvkeys = build_windows(df, feat_cols, target_col, config.window_size)
    print(f"  {len(X):,} clean windows from {len(np.unique(gvkeys))} companies")

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

    if config.normalize:
        print("\nNormalizing features (z-score from training set)...")
        X_train, X_val, X_test, mean, std = normalize_features(X_train, X_val, X_test)
    else:
        flat = X_train.reshape(-1, X_train.shape[2])
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)

    # Target stats
    print(f"\nTarget statistics:")
    print(f"  Train — mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")
    print(f"  Val   — mean: {y_val.mean():.4f}, std: {y_val.std():.4f}")
    print(f"  Test  — mean: {y_test.mean():.4f}, std: {y_test.std():.4f}")

    # Direction balance
    for name, ys in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        # Target is next quarter GM level; compute implied direction from
        # whether target > current (last timestep) GM
        # But since we normalized X, use raw y stats instead
        n_total = len(ys)
        if n_total > 0:
            print(f"  {name} samples: {n_total:,}")

    return WindowDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feat_cols,
        train_mean=mean,
        train_std=std,
    )


def save_dataset(ds: WindowDataset, output_dir: Path | None = None):
    """Save windowed dataset to .npz files."""
    if output_dir is None:
        output_dir = WINDOW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "train.npz",
        X=ds.X_train, y=ds.y_train,
    )
    np.savez_compressed(
        output_dir / "val.npz",
        X=ds.X_val, y=ds.y_val,
    )
    np.savez_compressed(
        output_dir / "test.npz",
        X=ds.X_test, y=ds.y_test,
    )
    np.savez(
        output_dir / "metadata.npz",
        feature_names=np.array(ds.feature_names),
        train_mean=ds.train_mean,
        train_std=ds.train_std,
    )

    print(f"\nSaved to {output_dir}/")
    print(f"  train.npz  — X: {ds.X_train.shape}, y: {ds.y_train.shape}")
    print(f"  val.npz    — X: {ds.X_val.shape},   y: {ds.y_val.shape}")
    print(f"  test.npz   — X: {ds.X_test.shape},  y: {ds.y_test.shape}")
    print(f"  metadata.npz — {len(ds.feature_names)} features, scaler params")


def load_saved_dataset(input_dir: Path | None = None) -> WindowDataset:
    """Load a previously saved windowed dataset."""
    if input_dir is None:
        input_dir = WINDOW_DIR

    train = np.load(input_dir / "train.npz")
    val = np.load(input_dir / "val.npz")
    test = np.load(input_dir / "test.npz")
    meta = np.load(input_dir / "metadata.npz", allow_pickle=True)

    return WindowDataset(
        X_train=train["X"], y_train=train["y"],
        X_val=val["X"], y_val=val["y"],
        X_test=test["X"], y_test=test["y"],
        feature_names=meta["feature_names"].tolist(),
        train_mean=meta["train_mean"],
        train_std=meta["train_std"],
    )


def main():
    ds = build_dataset()
    save_dataset(ds)

    # Quick shape verification
    print(f"\n--- Verification ---")
    print(f"X_train shape: {ds.X_train.shape}  →  (samples, {CONFIG.window_size} quarters, {len(ds.feature_names)} features)")
    print(f"y_train shape: {ds.y_train.shape}")
    print(f"Feature names: {ds.feature_names}")

    # Verify no NaN leaked through
    for name, arr in [("X_train", ds.X_train), ("X_val", ds.X_val),
                       ("X_test", ds.X_test), ("y_train", ds.y_train),
                       ("y_val", ds.y_val), ("y_test", ds.y_test)]:
        n_nan = np.isnan(arr).sum()
        print(f"  {name} NaN count: {n_nan}")

    print("\nDone.")


if __name__ == "__main__":
    main()