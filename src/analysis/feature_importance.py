"""
Feature importance analysis for trained models.

Provides permutation importance: for each feature, shuffle it across
the test set and measure the drop in directional accuracy. Features
the model relies on will cause a large accuracy drop when shuffled.

Usage:
    # Standalone
    python src/analysis/feature_importance.py

    # From another module
    from analysis.feature_importance import permutation_importance
    results = permutation_importance(model, X_test, y_test, feature_names)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import BaselineModelConfig

ROOT = Path(__file__).resolve().parents[2]
WINDOW_DIR = ROOT / "data" / "processed" / "windows"
MODEL_DIR = ROOT / "outputs" / "models" / "baseline_lstm"


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

@dataclass
class FeatureImportance:
    """Results from permutation importance analysis."""
    feature_names: list[str]
    baseline_dir_acc: float          # directional accuracy with no shuffling
    importance_scores: np.ndarray    # (n_features,) mean accuracy drop per feature
    importance_std: np.ndarray       # (n_features,) std across repeats
    n_repeats: int

    def sorted_indices(self, descending: bool = True) -> np.ndarray:
        """Return feature indices sorted by importance."""
        order = np.argsort(self.importance_scores)
        if descending:
            order = order[::-1]
        return order

    def summary(self, top_k: int = 20) -> str:
        """Pretty-print the top-k most important features."""
        idx = self.sorted_indices()[:top_k]
        lines = [
            f"Permutation Importance ({self.n_repeats} repeats)",
            f"Baseline directional accuracy: {self.baseline_dir_acc:.4f} ({self.baseline_dir_acc*100:.1f}%)",
            "",
            f"{'Rank':<6s} {'Feature':<30s} {'Acc Drop':>10s} {'± Std':>10s} {'Shuffled Acc':>14s}",
            "-" * 72,
        ]
        for rank, i in enumerate(idx, 1):
            drop = self.importance_scores[i]
            std = self.importance_std[i]
            shuffled_acc = self.baseline_dir_acc - drop
            lines.append(
                f"{rank:<6d} {self.feature_names[i]:<30s} "
                f"{drop:>+10.4f} {std:>10.4f} "
                f"{shuffled_acc:>13.1%}"
            )
        return "\n".join(lines)


def _predict_numpy(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Get model predictions as a numpy array."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.zeros(len(X))),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            pred = model(X_batch.to(device))
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct direction predictions (positive vs negative delta)."""
    return float(np.mean((y_pred > 0) == (y_true > 0)))


def permutation_importance(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    device: torch.device | None = None,
    n_repeats: int = 10,
    metric: str = "directional_accuracy",
    seed: int = 42,
) -> FeatureImportance:
    """
    Compute permutation importance for each feature.

    For each feature, shuffles it across all samples (and timesteps)
    n_repeats times and measures the drop in the chosen metric.

    Args:
        model:         Trained model
        X:             (N, window_size, n_features) array
        y:             (N,) target array
        feature_names: List of feature names, length = n_features
        device:        Torch device
        n_repeats:     Number of shuffle repeats per feature
        metric:        "directional_accuracy" or "mse"
        seed:          Random seed for reproducibility

    Returns:
        FeatureImportance with scores for each feature
    """
    if device is None:
        device = next(model.parameters()).device

    rng = np.random.RandomState(seed)
    n_samples, window_size, n_features = X.shape

    # Baseline score (no shuffling)
    y_pred_base = _predict_numpy(model, X, device)

    if metric == "directional_accuracy":
        baseline_score = _directional_accuracy(y, y_pred_base)
        higher_is_better = True
    elif metric == "mse":
        baseline_score = float(np.mean((y - y_pred_base) ** 2))
        higher_is_better = False
    else:
        raise ValueError(f"Unknown metric: {metric}")

    print(f"Baseline {metric}: {baseline_score:.4f}")
    print(f"Computing permutation importance for {n_features} features "
          f"({n_repeats} repeats each)...")

    # For each feature, shuffle and measure score drop
    importance_scores = np.zeros((n_features, n_repeats))

    for f_idx in range(n_features):
        for r in range(n_repeats):
            # Copy and shuffle this feature across samples
            X_shuffled = X.copy()
            # Shuffle the entire feature column (across all samples),
            # keeping the temporal structure within each sample intact
            # but breaking the feature-to-target mapping
            perm = rng.permutation(n_samples)
            X_shuffled[:, :, f_idx] = X[perm, :, f_idx]

            y_pred_shuffled = _predict_numpy(model, X_shuffled, device)

            if metric == "directional_accuracy":
                shuffled_score = _directional_accuracy(y, y_pred_shuffled)
            else:
                shuffled_score = float(np.mean((y - y_pred_shuffled) ** 2))

            if higher_is_better:
                importance_scores[f_idx, r] = baseline_score - shuffled_score
            else:
                importance_scores[f_idx, r] = shuffled_score - baseline_score

    mean_importance = importance_scores.mean(axis=1)
    std_importance = importance_scores.std(axis=1)

    return FeatureImportance(
        feature_names=feature_names,
        baseline_dir_acc=baseline_score if metric == "directional_accuracy" else 0.0,
        importance_scores=mean_importance,
        importance_std=std_importance,
        n_repeats=n_repeats,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_feature_importance(
    result: FeatureImportance,
    output_path: Path,
    top_k: int = 20,
):
    """Horizontal bar chart of top-k most important features."""
    idx = result.sorted_indices()[:top_k]

    # Reverse so highest importance is at the top
    idx = idx[::-1]

    names = [result.feature_names[i] for i in idx]
    scores = result.importance_scores[idx]
    stds = result.importance_std[idx]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.35)))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#111827")

    colors = ["#22d3ee" if s > 0 else "#64748b" for s in scores]
    bars = ax.barh(range(len(names)), scores, xerr=stds, color=colors,
                   edgecolor="none", capsize=3, ecolor="#475569")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Accuracy Drop When Shuffled", color="#94a3b8")
    ax.set_title(f"Feature Importance (Top {top_k})", color="#e2e8f0", fontweight="bold")
    ax.axvline(x=0, color="#475569", linewidth=0.8, linestyle="--")
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main — run on saved model + data
# ---------------------------------------------------------------------------

def main():
    from models.baseline_lstm import LSTMPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(MODEL_DIR / "model.pt", map_location=device, weights_only=False)
    config = BaselineModelConfig(**checkpoint["config"])
    feature_names = checkpoint["feature_names"]
    num_features = checkpoint["num_features"]

    model = LSTMPredictor(num_features=num_features, config=config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"  {num_features} features, {config.hidden_dim}h {config.num_layers}L LSTM")

    # Load test data
    print("Loading test data...")
    test_data = np.load(WINDOW_DIR / "test.npz")
    X_test, y_test = test_data["X"], test_data["y"]
    print(f"  Test: {X_test.shape}")

    # Compute permutation importance
    result = permutation_importance(
        model, X_test, y_test, feature_names,
        device=device, n_repeats=10,
    )

    # Print results
    print("\n" + "=" * 72)
    print(result.summary(top_k=len(feature_names)))

    # Plot
    output_path = MODEL_DIR / "feature_importance.png"
    plot_feature_importance(result, output_path, top_k=min(len(feature_names), 33))

    # Save raw results
    np.savez(
        MODEL_DIR / "feature_importance.npz",
        feature_names=np.array(feature_names),
        importance_scores=result.importance_scores,
        importance_std=result.importance_std,
        baseline_dir_acc=result.baseline_dir_acc,
        n_repeats=result.n_repeats,
    )
    print(f"Saved: {MODEL_DIR / 'feature_importance.npz'}")

    print("\nDone.")


if __name__ == "__main__":
    main()