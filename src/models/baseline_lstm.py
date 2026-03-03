"""
Baseline LSTM model for next-quarter Gross Margin prediction.

Usage:
    python src/models/baseline_lstm.py

Reads:  data/processed/windows/{train,val,test}.npz
Output: outputs/models/baseline_lstm/
            model.pt            — best model weights
            training_curves.png — loss curves
            evaluation.txt      — metrics on all splits
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
import json
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from schema import BaselineModelConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
WINDOW_DIR = ROOT / "data" / "processed" / "windows"
OUTPUT_DIR = ROOT / "outputs" / "models" / "baseline_lstm"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LSTMPredictor(nn.Module):
    """LSTM-based predictor for next-quarter gross margin."""

    def __init__(self, num_features: int, config: BaselineModelConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.head = nn.Sequential(
            nn.Linear(config.effective_hidden, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_features)
        Returns:
            predictions: (batch_size,)
        """
        # lstm_out: (batch, seq_len, hidden * num_directions)
        # h_n:      (num_layers * num_directions, batch, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.config.bidirectional:
            # Concatenate final hidden states from both directions
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]

        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# Naive baselines
# ---------------------------------------------------------------------------

def naive_last_value(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Predict delta = 0 (random walk: next quarter GM = current quarter GM)."""
    return np.zeros(X.shape[0])


def naive_last_delta(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Predict delta = last observed delta (momentum baseline)."""
    delta_idx = feature_names.index("gross_margin_delta")
    return X[:, -1, delta_idx]


def naive_mean_delta(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Predict delta = mean delta across the window."""
    delta_idx = feature_names.index("gross_margin_delta")
    return X[:, :, delta_idx].mean(axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute regression and directional metrics for delta targets."""
    # Regression
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Directional accuracy: since target is delta, direction = sign
    # Did we correctly predict whether GM goes up (>0) or down (<0)?
    pred_direction = y_pred > 0
    true_direction = y_true > 0
    directional_acc = np.mean(pred_direction == true_direction)

    # Directional accuracy excluding near-zero deltas (|delta| < 0.005)
    # These are essentially noise and hard to call directionally
    significant = np.abs(y_true) >= 0.005
    if significant.sum() > 0:
        dir_acc_significant = np.mean(
            (y_pred[significant] > 0) == (y_true[significant] > 0)
        )
    else:
        dir_acc_significant = 0.0

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "directional_accuracy": float(directional_acc),
        "dir_acc_significant": float(dir_acc_significant),
        "n_samples": int(len(y_true)),
        "n_significant": int(significant.sum()),
    }

    if label:
        print(f"\n  {label}:")
        print(f"    MAE:            {mae:.6f}")
        print(f"    RMSE:           {rmse:.6f}")
        print(f"    R²:             {r2:.4f}")
        print(f"    Dir. Acc:       {directional_acc:.4f} ({directional_acc * 100:.1f}%)")
        print(f"    Dir. Acc (sig): {dir_acc_significant:.4f} ({dir_acc_significant * 100:.1f}%) "
              f"[{significant.sum():,} samples with |Δ| >= 0.5pp]")
        print(f"    Samples:        {len(y_true):,}")

    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BaselineModelConfig,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Train the model with early stopping."""

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=config.patience)

    history = {"train_loss": [], "val_loss": [], "lr": []}

    print(f"\nTraining for up to {config.max_epochs} epochs (patience={config.patience})...")
    print(f"{'Epoch':>6s} {'Train Loss':>12s} {'Val Loss':>12s} {'LR':>10s} {'Time':>7s}")
    print("-" * 52)

    for epoch in range(1, config.max_epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        scheduler.step(val_loss)

        # Print every 5 epochs or on improvement
        if epoch % 5 == 0 or epoch == 1 or val_loss <= early_stop.best_loss:
            marker = " ✓" if val_loss <= early_stop.best_loss else ""
            print(f"{epoch:>6d} {train_loss:>12.6f} {val_loss:>12.6f} {current_lr:>10.6f} {elapsed:>6.1f}s{marker}")

        if early_stop.step(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch} (best val loss: {early_stop.best_loss:.6f})")
            break

    # Restore best weights
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)

    return model, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_path: Path):
    """Plot training and validation loss curves."""
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0f1a")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#64748b")
        ax.spines[:].set_color("#1e293b")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], color="#22d3ee", linewidth=1.5, label="Train")
    ax1.plot(epochs, history["val_loss"], color="#f59e0b", linewidth=1.5, label="Val")
    ax1.set_xlabel("Epoch", color="#94a3b8")
    ax1.set_ylabel("MSE Loss", color="#94a3b8")
    ax1.set_title("Training Curves", color="#e2e8f0", fontweight="bold")
    ax1.legend()
    ax1.set_yscale("log")

    # Learning rate
    ax2.plot(epochs, history["lr"], color="#a78bfa", linewidth=1.5)
    ax2.set_xlabel("Epoch", color="#94a3b8")
    ax2.set_ylabel("Learning Rate", color="#94a3b8")
    ax2.set_title("Learning Rate Schedule", color="#e2e8f0", fontweight="bold")
    ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str, output_path: Path):
    """Scatter plot of predicted vs actual."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0a0f1a")
    ax.set_facecolor("#111827")

    ax.scatter(y_true, y_pred, alpha=0.2, s=10, c="#22d3ee", edgecolors="none")

    lims = [min(y_true.min(), y_pred.min()) - 0.02,
            max(y_true.max(), y_pred.max()) + 0.02]
    ax.plot(lims, lims, "--", color="#64748b", linewidth=1, alpha=0.6)

    ax.set_xlabel("Actual GM Delta", color="#94a3b8")
    ax.set_ylabel("Predicted GM Delta", color="#94a3b8")
    ax.set_title(title, color="#e2e8f0", fontweight="bold")
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#1e293b")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Predict helper
# ---------------------------------------------------------------------------

def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Get predictions from the model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = BaselineModelConfig()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading windowed data...")
    train_data = np.load(WINDOW_DIR / "train.npz")
    val_data = np.load(WINDOW_DIR / "val.npz")
    test_data = np.load(WINDOW_DIR / "test.npz")
    meta = np.load(WINDOW_DIR / "metadata.npz", allow_pickle=True)

    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val = val_data["X"], val_data["y"]
    X_test, y_test = test_data["X"], test_data["y"]
    feature_names = meta["feature_names"].tolist()

    num_features = X_train.shape[2]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Features: {num_features}")

    # DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=config.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config.batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=config.batch_size, shuffle=False,
    )

    # --- Naive baselines ---
    # For delta-based baselines, we need unnormalized features
    train_mean = meta["train_mean"]
    train_std = meta["train_std"]
    X_test_raw = X_test * train_std + train_mean

    print("\n" + "=" * 60)
    print("NAIVE BASELINES (Test Set)")
    print("=" * 60)

    # Random walk: predict delta = 0 (no change)
    naive_zero_pred = naive_last_value(X_test_raw, feature_names)
    naive_zero_metrics = compute_metrics(y_test, naive_zero_pred, "Zero Delta (random walk)")

    # Momentum: predict delta = last observed delta
    naive_momentum_pred = naive_last_delta(X_test_raw, feature_names)
    naive_momentum_metrics = compute_metrics(y_test, naive_momentum_pred, "Last Delta (momentum)")

    # Mean delta: predict delta = mean of window deltas
    naive_mean_pred = naive_mean_delta(X_test_raw, feature_names)
    naive_mean_metrics = compute_metrics(y_test, naive_mean_pred, "Window Mean Delta")

    # Global mean delta from training set
    global_mean_pred = np.full_like(y_test, y_train.mean())
    global_mean_metrics = compute_metrics(y_test, global_mean_pred, "Global Mean Delta")

    # --- Train LSTM ---
    print("\n" + "=" * 60)
    print("LSTM MODEL")
    print("=" * 60)

    model = LSTMPredictor(num_features=num_features, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Architecture: {config.num_layers}L LSTM (h={config.hidden_dim}) → MLP → 1")

    model, history = train_model(model, train_loader, val_loader, config, device)

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    splits = {
        "Train": (train_loader, y_train),
        "Val":   (val_loader, y_val),
        "Test":  (test_loader, y_test),
    }

    all_metrics = {}
    for split_name, (loader, y_true) in splits.items():
        y_pred = predict(model, loader, device)
        metrics = compute_metrics(y_true, y_pred, f"LSTM — {split_name}")
        all_metrics[split_name] = metrics

    # --- Comparison table ---
    print("\n" + "=" * 60)
    print("COMPARISON (Test Set)")
    print("=" * 60)
    print(f"\n{'Model':<25s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s} {'Dir Acc':>10s} {'Dir(sig)':>10s}")
    print("-" * 77)
    for name, m in [("Global Mean Delta", global_mean_metrics),
                     ("Zero Delta (RW)", naive_zero_metrics),
                     ("Last Delta (Momentum)", naive_momentum_metrics),
                     ("Window Mean Delta", naive_mean_metrics),
                     ("LSTM", all_metrics["Test"])]:
        print(f"{name:<25s} {m['mae']:>10.6f} {m['rmse']:>10.6f} {m['r2']:>10.4f} "
              f"{m['directional_accuracy']*100:>9.1f}% {m['dir_acc_significant']*100:>9.1f}%")

    # --- Save ---
    torch.save({
        "model_state": model.state_dict(),
        "config": config.__dict__,
        "num_features": num_features,
        "feature_names": feature_names,
        "metrics": all_metrics,
    }, OUTPUT_DIR / "model.pt")
    print(f"\nModel saved: {OUTPUT_DIR / 'model.pt'}")

    # Save metrics as JSON for easy reading
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        all_results = {
            "baselines": {
                "global_mean_delta": global_mean_metrics,
                "zero_delta": naive_zero_metrics,
                "last_delta_momentum": naive_momentum_metrics,
                "window_mean_delta": naive_mean_metrics,
            },
            "lstm": all_metrics,
        }
        json.dump(all_results, f, indent=2)

    # Plot
    plot_training_curves(history, OUTPUT_DIR / "training_curves.png")

    y_test_pred = predict(model, test_loader, device)
    plot_predictions(y_test, y_test_pred, "LSTM — Test Set (GM Delta)", OUTPUT_DIR / "test_predictions.png")
    plot_predictions(y_test, naive_zero_pred, "Zero Delta Baseline — Test Set", OUTPUT_DIR / "baseline_predictions.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()