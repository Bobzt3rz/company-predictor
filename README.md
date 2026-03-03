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

# 4. Pull commodity prices (no auth needed)
python src/data/pull_commodities.py

# 5. Compute financial ratios + targets
python src/features/compute_ratios.py

# 6. Build sliding window dataset (merges all channels)
python src/features/build_windows.py

# 7. Train baseline LSTM
python src/models/baseline_lstm.py

# 8. Feature importance analysis
python src/analysis/feature_importance.py
```

## Project Structure

```
company-predictor/
├── src/
│   ├── schema.py                  # Central schema: column names, ratio/commodity
│   │                              # definitions, configs, channel registry metadata
│   ├── data/
│   │   ├── pull_compustat.py      # Pull quarterly fundamentals from WRDS Compustat
│   │   └── pull_commodities.py    # Pull commodity futures from Yahoo Finance
│   ├── features/
│   │   ├── compute_ratios.py      # Compute Channel A financial ratios
│   │   └── build_windows.py       # Multi-channel merge, sliding windows,
│   │                              # temporal split, z-score normalization
│   ├── models/
│   │   └── baseline_lstm.py       # Baseline 2-layer LSTM with naive baselines
│   └── analysis/
│       └── feature_importance.py  # Permutation importance analysis
├── data/
│   ├── raw/                       # Raw parquet files (git-ignored)
│   └── processed/
│       └── windows/               # Train/val/test .npz tensors (git-ignored)
├── outputs/
│   └── models/
│       └── baseline_lstm/         # Weights, metrics, plots, importance scores
├── environment.yml
└── README.md
```

## Pipeline Overview

```
┌──────────────┐     ┌──────────────┐
│ pull_compustat│     │pull_commodit.│
│              │     │              │
│ WRDS SQL     │     │ Yahoo Finance│
│ Revenue floor│     │ 15 futures   │
│ Quality filt.│     │ QoQ % change │
│              │     │              │
│ → fundq.pqt  │     │ → ch_b.pqt   │
└──────┬───────┘     └──────┬───────┘
       │                    │
       ▼                    │
┌──────────────┐            │
│compute_ratios│            │
│              │            │
│ 6 ratios     │            │
│ QoQ deltas   │            │
│ Winsorize    │            │
│ Targets      │            │
│              │            │
│ → ch_a.pqt   │            │
└──────┬───────┘            │
       │                    │
       ▼                    ▼
┌─────────────────────────────┐     ┌──────────────────┐
│       build_windows          │     │   baseline_lstm   │
│                              │     │                   │
│ Channel registry merges A+B  │ ──→ │ 2-layer LSTM      │
│ Company channels: (gvkey,q)  │     │ Naive baselines   │
│ Market channels:  (q) only   │     │ Early stopping    │
│ Sliding window (8 quarters)  │     │ Metrics + plots   │
│ Temporal train/val/test      │     │                   │
│ Z-score normalization        │     │ → model.pt        │
│                              │     │ → metrics.json    │
│ → windows/*.npz              │     └──────────────────┘
└─────────────────────────────┘
```

## Channel Architecture

Features are organized into channels, each producing a separate parquet file. The window builder merges them automatically based on their granularity:

| Channel | Granularity | Join Key | Features | Source |
|---------|-------------|----------|----------|--------|
| A — Financial Ratios | Per company | (gvkey, fyearq, fqtr) | 12 | WRDS Compustat |
| B — Commodity Prices | Market-wide | (fyearq, fqtr) | 15 | Yahoo Finance |
| C — SEC Filings (future) | Per company | (gvkey, fyearq, fqtr) | TBD | EDGAR |

Adding a new channel requires: (1) a feature script that outputs a parquet with `fyearq`/`fqtr` columns, (2) a `ChannelSource` entry in `build_windows.py`, and (3) schema definitions in `schema.py`.

### Channel A: Financial Ratios (12 features)

Six ratios computed from Compustat quarterly fundamentals, each with a QoQ delta:

| Ratio | Formula | + Delta |
|-------|---------|---------|
| Gross Margin | (revenue - COGS) / revenue | gross_margin_delta |
| Operating Margin | operating income / revenue | operating_margin_delta |
| SGA / Revenue | SG&A expense / revenue | sga_to_revenue_delta |
| Inventory Turnover | COGS / inventory | inventory_turnover_delta |
| Asset Turnover | revenue / total assets | asset_turnover_delta |
| Current Ratio | current assets / current liabilities | current_ratio_delta |

### Channel B: Commodity Prices (15 features)

Quarter-over-quarter percentage changes from Yahoo Finance futures:

| Category | Commodities | Rationale |
|----------|-------------|-----------|
| Agricultural | Corn, Wheat, Soybeans, Soybean Oil, Soybean Meal, Sugar, Cocoa, Coffee, Cotton, Live Cattle | Raw material & ingredient costs |
| Energy | Crude Oil (WTI), Natural Gas | Transport, packaging, processing |
| Metals | Gold, Copper | Inflation proxy, packaging (cans) |
| Macro | US Dollar Index (DXY) | FX exposure for multinationals |

## Target

The model predicts **next-quarter gross margin delta** (the change in gross margin from current quarter to next quarter). This is more useful than predicting the level because margins are highly autocorrelated — a naive "predict last value" baseline gets high R² on levels but only ~47% directional accuracy.

## Sliding Window Format

The LSTM expects 3D tensors of shape `(batch_size, 8, 27)`:

```
         Ch.A (12 features)          Ch.B (15 features)
         ratio₁ delta₁ ... ratio₆   corn  wheat ... usd
Q1 2020  [0.42   0.01  ...  1.82    +0.03 -0.05 ... +0.01]
Q2 2020  [0.41  -0.01  ...  1.79    +0.08 +0.02 ... -0.02]
   ...
Q4 2021  [0.46   0.02  ...  1.91    -0.04 +0.01 ... +0.03]

Target: GM delta for Q1 2022
```

Windows are split temporally (no data leakage):
- **Train**: target quarter ≤ 2018
- **Val**: 2019–2021
- **Test**: 2022+

Features are z-score normalized using training set statistics only. Channel slice indices are saved in metadata for future channel-aware models.

## Current Results

### Baseline LSTM (Channel A + B, h=256)

| Model | MAE | RMSE | R² | Dir Acc | Dir Acc (sig) |
|-------|-----|------|----|---------|---------------|
| Global Mean Delta | 0.0306 | 0.0675 | -0.0003 | 46.9% | 47.1% |
| Zero Delta (Random Walk) | 0.0306 | 0.0675 | -0.0002 | 46.9% | 47.1% |
| Last Delta (Momentum) | 0.0518 | 0.1134 | -1.8254 | 42.0% | 39.3% |
| **LSTM (A+B, h=256)** | **0.0284** | **0.0571** | **0.2830** | **59.2%** | **62.7%** |

**Key metrics**:
- **Dir Acc**: Did we correctly predict margin expand vs. compress? (59.2% vs ~47% baselines)
- **Dir Acc (sig)**: Same but only for quarters with |Δ| ≥ 0.5pp (62.7%, filters out noise)
- **R²**: Fraction of variance in actual deltas explained (0.28 = 28%)

### Feature Importance (Permutation)

Top features by accuracy drop when shuffled (256h model, test set):

| Rank | Feature | Acc Drop | Channel |
|------|---------|----------|---------|
| 1 | gross_margin_delta | +7.1% | A |
| 2 | gross_margin | +3.0% | A |
| 3 | operating_margin | +1.7% | A |
| 4 | sga_to_revenue | +1.3% | A |
| 5 | crude_oil_qoq | +1.2% | B |
| 6 | inventory_turnover_delta | +1.0% | A |
| 7 | wheat_qoq | +0.9% | B |
| 8 | cocoa_qoq | +0.7% | B |
| 9 | coffee_qoq | +0.6% | B |
| 10 | current_ratio | +0.5% | A |

Channel A financial ratios drive most of the signal, with crude oil, wheat, cocoa, and coffee as the most impactful commodity features. Several commodity features (sugar, natural gas, soybean meal) show near-zero or slightly negative importance, suggesting the model doesn't benefit from them in their current form.

## Data Stats

- **266 companies** (Consumer Staples, GICS 30)
- **~19,000 company-quarters** (2003–2026)
- **~14,000 clean sliding windows** after NaN removal
- **Train/Val/Test**: ~9,600 / ~2,200 / ~2,050 windows
- **15 commodity futures** (2003–2026, daily → quarterly)

## Build Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pull + Channel A ratios | ✅ Done |
| 2 | Sliding window dataset builder | ✅ Done |
| 3 | Baseline LSTM (Channel A only) | ✅ Done (60% dir acc) |
| 4 | Channel B: Commodity price features | ✅ Done (15 futures, Yahoo Finance) |
| 4b | Multi-channel window builder | ✅ Done (registry pattern) |
| 4c | Feature importance analysis | ✅ Done (permutation importance) |
| 5 | Channel C: SEC filing text features (NLP) | ⬜ Next |
| 6 | Multimodal fusion model | ⬜ |
| 7 | Evaluation, ablation, & interpretability | ⬜ |

## Environment

Requires Python 3.11, WRDS account for Compustat data, and CUDA GPU recommended for training. See `environment.yml` for full dependency list. Commodity data from Yahoo Finance requires no authentication.