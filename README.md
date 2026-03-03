# Multimodal Fundamental Forecaster (MMF)

Predict next-quarter **Gross Margin changes** for Consumer Staples (GICS 30) companies using a multimodal deep learning model that fuses financial ratios, commodity prices, and structured text features from SEC filings.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate company-predictor

# 2. Install PyTorch (if not already)
conda install pytorch -c pytorch

# 3. Pull raw data (requires WRDS account)
export WRDS_USERNAME=your_username
export WRDS_PASSWORD=your_password
python src/data/pull_compustat.py

# 4. Compute financial ratios + targets
python src/features/compute_ratios.py

# 5. Build sliding window dataset
python src/features/build_windows.py

# 6. Train baseline LSTM
python src/models/baseline_lstm.py
```

## Project Structure

```
company-predictor/
├── src/
│   ├── schema.py                  # Central schema: all column names, ratio
│   │                              # definitions, configs, and type definitions
│   │                              # shared across every module
│   ├── data/
│   │   └── pull_compustat.py      # Pull quarterly fundamentals from WRDS
│   │                              # Compustat, apply revenue floor, filter
│   │                              # for data quality
│   ├── features/
│   │   ├── compute_ratios.py      # Compute Channel A financial ratios,
│   │   │                          # winsorize, compute deltas/YoY, create targets
│   │   └── build_windows.py       # Build sliding window datasets for sequence
│   │                              # models. Temporal train/val/test split,
│   │                              # z-score normalization, NaN handling
│   └── models/
│       └── baseline_lstm.py       # Baseline 2-layer LSTM with naive baselines,
│                                  # training loop, early stopping, evaluation
├── data/
│   ├── raw/                       # Raw parquet files from WRDS (git-ignored)
│   └── processed/
│       └── windows/               # Train/val/test .npz tensors (git-ignored)
├── outputs/
│   └── models/
│       └── baseline_lstm/         # Model weights, metrics, training curves
├── environment.yml                # Conda environment
└── README.md
```

## Pipeline Overview

The data flows through four stages. Each stage reads from the previous stage's output:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  pull_compustat   │ ──→ │  compute_ratios   │ ──→ │  build_windows    │ ──→ │  baseline_lstm    │
│                   │     │                   │     │                   │     │                   │
│  WRDS SQL query   │     │  6 financial      │     │  Sliding window   │     │  2-layer LSTM     │
│  Revenue floor    │     │  ratios           │     │  (8 quarters)     │     │  Naive baselines  │
│  Quality filter   │     │  QoQ deltas       │     │  Temporal split   │     │  Early stopping   │
│  (≥40 quarters)   │     │  YoY changes      │     │  Z-score norm     │     │  Metrics + plots  │
│                   │     │  Winsorization     │     │  NaN drop         │     │                   │
│  → raw/*.parquet  │     │  Targets (Δ, dir) │     │  → windows/*.npz  │     │  → model.pt       │
│                   │     │  → ratios.parquet  │     │                   │     │  → metrics.json   │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Schema (`src/schema.py`)

All column names, ratio definitions, model configs, and pipeline parameters are defined centrally in `schema.py`. Every module imports from here — there are no hardcoded column strings anywhere else. Key components:

- **`CompustatVar`** — Enum of raw Compustat variable names (`revtq`, `cogsq`, etc.)
- **`RatioDef` / `Ratios`** — Dataclass defining each ratio (name, label, formula, color, display format) with convenience methods like `Ratios.names()`, `Ratios.get("gross_margin")`
- **`Targets`** — Enum of prediction target columns (`target_gm_delta`, `target_gm_direction`, etc.)
- **`CompustatConfig`** — Data pull parameters (GICS sector, date range, revenue floor)
- **`WindowConfig`** — Sliding window parameters (window size, train/val/test split years)
- **`BaselineModelConfig`** — LSTM hyperparameters (hidden dim, layers, learning rate, etc.)

To add a new ratio, add a `RatioDef` entry to `RATIO_DEFS` in `schema.py` and it will automatically flow through compute, windowing, and visualization.

## Channel A: Financial Ratios

Six ratios computed from Compustat quarterly fundamentals:

| Ratio | Formula | Type |
|-------|---------|------|
| Gross Margin | (revenue - COGS) / revenue | Percentage |
| Operating Margin | operating income / revenue | Percentage |
| SGA / Revenue | SG&A expense / revenue | Percentage |
| Inventory Turnover | COGS / inventory | Raw number |
| Asset Turnover | revenue / total assets | Percentage |
| Current Ratio | current assets / current liabilities | Raw number |

Each ratio also produces a QoQ delta and YoY change, giving 18 features total per quarter.

## Target

The model predicts **next-quarter gross margin delta** (the change in gross margin from current quarter to next quarter). This is more useful than predicting the level because margins are highly autocorrelated — a naive "predict last value" baseline gets R²=0.87 on levels but only 47% directional accuracy.

## Sliding Window Format

The LSTM expects 3D tensors of shape `(batch_size, 8, 18)`:

```
         feat_0  feat_1  feat_2  ...  feat_17
Q1 2020  [0.42    0.18    0.15   ...   0.02  ]
Q2 2020  [0.41    0.17    0.14   ...  -0.01  ]
   ...
Q4 2021  [0.46    0.22    0.16   ...   0.03  ]

Target: GM delta for Q1 2022
```

Windows are split temporally (no data leakage):
- **Train**: target quarter ≤ 2018
- **Val**: 2019–2021
- **Test**: 2022+

Features are z-score normalized using training set statistics only.

## Current Results (Baseline LSTM)

| Model | MAE | RMSE | R² | Dir Acc | Dir Acc (sig) |
|-------|-----|------|----|---------|---------------|
| Global Mean Delta | 0.0304 | 0.0670 | -0.0004 | 46.8% | 46.9% |
| Zero Delta (Random Walk) | 0.0304 | 0.0670 | -0.0002 | 46.8% | 46.9% |
| Last Delta (Momentum) | 0.0513 | 0.1126 | -1.8231 | 42.1% | 39.4% |
| **LSTM (Channel A)** | **0.0279** | **0.0561** | **0.2996** | **60.0%** | **61.7%** |

**Key metrics**:
- **MAE**: Average magnitude of prediction error (in GM percentage points)
- **R²**: Fraction of variance in actual deltas explained by the model (0.30 = 30%)
- **Dir Acc**: Did we correctly predict expand vs. compress?
- **Dir Acc (sig)**: Same but only for quarters with |Δ| ≥ 0.5pp (filters out noise)

The LSTM achieves 60% directional accuracy vs. ~47% for all naive baselines, correctly predicting margin direction on significant moves 62% of the time.

## Data Stats

- **265 companies** (Consumer Staples, GICS 30)
- **18,750 company-quarters** (2004–2024)
- **12,977 clean sliding windows** after NaN removal
- **Train/Val/Test**: 8,782 / 2,159 / 2,036 windows

## Build Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pull + Channel A ratios | ✅ Done |
| 2 | Sliding window dataset builder | ✅ Done |
| 3 | Baseline LSTM (Channel A only) | ✅ Done (60% dir acc) |
| 4 | Channel B: Commodity price features | ⬜ Next |
| 5 | Channel C: SEC filing text features (NLP) | ⬜ |
| 6 | Multimodal fusion model | ⬜ |
| 7 | Evaluation, ablation, & interpretability | ⬜ |

## Environment

Requires Python 3.11, WRDS account for data access, and CUDA GPU recommended for training. See `environment.yml` for full dependency list.