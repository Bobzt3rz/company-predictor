# Multimodal Fundamental Forecaster (MMF)

Predict next-quarter **Gross Margin** for Consumer Staples (GICS 30) companies
using a multimodal model that fuses financial ratios, commodity prices, and
structured text features extracted from SEC filings.

## Project Structure

```
multimodal-forecaster/
├── configs/                  # Model and data config files
├── data/
│   ├── raw/                  # Raw data from WRDS, FRED, etc.
│   └── processed/            # Computed features, sliding windows
├── notebooks/                # Exploration and analysis
├── src/
│   ├── data/                 # Data pulling scripts
│   │   └── pull_compustat.py # WRDS Compustat quarterly fundamentals
│   ├── features/             # Feature engineering
│   │   └── compute_ratios.py # Channel A: financial ratios + targets
│   ├── models/               # Model architectures
│   └── evaluation/           # Metrics and evaluation
└── environment.yml           # Conda environment
```

## Setup

```bash
# Create environment
conda env create -f environment.yml
conda activate mmf

# Pull data (requires WRDS account — will prompt for credentials on first run)
python src/data/pull_compustat.py

# Compute Channel A features
python src/features/compute_ratios.py
```

## Build Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Data pull + Channel A ratios | ✅ Scripts ready |
| 2 | Sliding window dataset builder | ⬜ |
| 3 | Baseline model (Channel A only) | ⬜ |
| 4 | Channel B: Commodity features | ⬜ |
| 5 | Channel C: NLP text features | ⬜ |
| 6 | Evaluation & ablation | ⬜ |

## Evaluation Metrics

- **Primary**: Directional accuracy on gross margin change (did we predict compress/expand correctly?)
- **Secondary**: MAE on gross margin level prediction

## Target

- **Primary target**: Next-quarter Gross Margin
- **Secondary target**: Next-quarter Operating Margin# company-predictor
