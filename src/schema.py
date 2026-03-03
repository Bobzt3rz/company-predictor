"""
Central schema definitions for the Multimodal Fundamental Forecaster.

All column names, ratio definitions, variable groups, and display metadata
live here so that every module imports from one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Compustat raw variables
# ---------------------------------------------------------------------------

class CompustatVar(str, Enum):
    """Raw Compustat quarterly variables pulled from WRDS."""
    # Identifiers
    GVKEY = "gvkey"
    DATADATE = "datadate"
    FYEARQ = "fyearq"
    FQTR = "fqtr"
    RDQ = "rdq"

    # Income statement
    REVTQ = "revtq"
    COGSQ = "cogsq"
    XSGAQ = "xsgaq"
    OIADPQ = "oiadpq"

    # Balance sheet
    ACTQ = "actq"
    LCTQ = "lctq"
    INVTQ = "invtq"
    ATQ = "atq"
    LTQ = "ltq"
    SEQQ = "seqq"

    @classmethod
    def id_cols(cls) -> list[str]:
        return [cls.GVKEY, cls.DATADATE, cls.FYEARQ, cls.FQTR]

    @classmethod
    def income_stmt_cols(cls) -> list[str]:
        return [cls.REVTQ, cls.COGSQ, cls.XSGAQ, cls.OIADPQ]

    @classmethod
    def balance_sheet_cols(cls) -> list[str]:
        return [cls.ACTQ, cls.LCTQ, cls.INVTQ, cls.ATQ, cls.LTQ, cls.SEQQ]

    @classmethod
    def all_numeric(cls) -> list[str]:
        return cls.income_stmt_cols() + cls.balance_sheet_cols()

    @classmethod
    def all_vars(cls) -> list[str]:
        return [v.value for v in cls]


# ---------------------------------------------------------------------------
# Financial ratios (Channel A)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RatioDef:
    """Definition of a single financial ratio."""
    name: str
    label: str
    numerator: str
    denominator: str
    description: str
    is_percentage: bool = True  # display as percentage vs raw number
    color: str = "#22d3ee"

    @property
    def delta_name(self) -> str:
        return f"{self.name}_delta"

    @property
    def yoy_name(self) -> str:
        return f"{self.name}_yoy"


# Master list of Channel A ratios
RATIO_DEFS: tuple[RatioDef, ...] = (
    RatioDef(
        name="gross_margin",
        label="Gross Margin",
        numerator="(revtq - cogsq)",
        denominator="revtq",
        description="Revenue minus COGS as a fraction of revenue",
        color="#22d3ee",
    ),
    RatioDef(
        name="operating_margin",
        label="Operating Margin",
        numerator="oiadpq",
        denominator="revtq",
        description="Operating income after depreciation as a fraction of revenue",
        color="#10b981",
    ),
    RatioDef(
        name="sga_to_revenue",
        label="SGA / Revenue",
        numerator="xsgaq",
        denominator="revtq",
        description="Selling, general & administrative expense as a fraction of revenue",
        color="#f59e0b",
    ),
    RatioDef(
        name="inventory_turnover",
        label="Inventory Turnover",
        numerator="cogsq",
        denominator="invtq",
        description="COGS divided by inventory (times per quarter)",
        is_percentage=False,
        color="#a78bfa",
    ),
    RatioDef(
        name="asset_turnover",
        label="Asset Turnover",
        numerator="revtq",
        denominator="atq",
        description="Revenue divided by total assets",
        color="#f472b6",
    ),
    RatioDef(
        name="current_ratio",
        label="Current Ratio",
        numerator="actq",
        denominator="lctq",
        description="Current assets divided by current liabilities",
        is_percentage=False,
        color="#3b82f6",
    ),
)


class Ratios:
    """Convenience accessors for ratio definitions."""

    _by_name: dict[str, RatioDef] = {r.name: r for r in RATIO_DEFS}

    @classmethod
    def get(cls, name: str) -> RatioDef:
        return cls._by_name[name]

    @classmethod
    def all(cls) -> tuple[RatioDef, ...]:
        return RATIO_DEFS

    @classmethod
    def names(cls) -> list[str]:
        return [r.name for r in RATIO_DEFS]

    @classmethod
    def labels(cls) -> dict[str, str]:
        """Map of name -> display label."""
        return {r.name: r.label for r in RATIO_DEFS}

    @classmethod
    def colors(cls) -> dict[str, str]:
        """Map of name -> hex color."""
        return {r.name: r.color for r in RATIO_DEFS}

    @classmethod
    def delta_names(cls) -> list[str]:
        return [r.delta_name for r in RATIO_DEFS]

    @classmethod
    def yoy_names(cls) -> list[str]:
        return [r.yoy_name for r in RATIO_DEFS]

    @classmethod
    def percentage_ratios(cls) -> list[str]:
        return [r.name for r in RATIO_DEFS if r.is_percentage]

    @classmethod
    def raw_ratios(cls) -> list[str]:
        return [r.name for r in RATIO_DEFS if not r.is_percentage]


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------

class Targets(str, Enum):
    """Prediction target columns."""
    GROSS_MARGIN = "target_gross_margin"
    GM_DELTA = "target_gm_delta"
    GM_DIRECTION = "target_gm_direction"

    @classmethod
    def all(cls) -> list[str]:
        return [v.value for v in cls]


# ---------------------------------------------------------------------------
# Channel definitions (for future multimodal fusion)
# ---------------------------------------------------------------------------

class Channel(str, Enum):
    """Model input channels."""
    A_RATIOS = "channel_a"      # Financial ratios
    B_COMMODITIES = "channel_b"  # Commodity prices (future)
    C_TEXT = "channel_c"         # SEC filing text features (future)


# ---------------------------------------------------------------------------
# Data pipeline config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompustatConfig:
    """Configuration for the Compustat data pull."""
    gics_sector: str = "30"
    start_year: int = 2004
    end_year: int = 2024
    min_quarters: int = 40
    min_revenue: float = 1.0  # minimum quarterly revenue ($M) to keep

    # SQL filter values
    ind_fmt: str = "INDL"
    data_fmt: str = "STD"
    pop_src: str = "D"
    consol: str = "C"


@dataclass(frozen=True)
class WinsorizeBounds:
    """Winsorization percentile bounds."""
    lower: float = 0.01
    upper: float = 0.99


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for the sliding window dataset builder."""
    window_size: int = 8             # quarters of lookback
    train_end_year: int = 2018       # train: <= this year
    val_end_year: int = 2021         # val: train_end_year < year <= this
                                     # test: > val_end_year
    normalize: bool = True           # z-score normalize features
    drop_nan: bool = True            # drop windows containing any NaN


@dataclass(frozen=True)
class BaselineModelConfig:
    """Configuration for the baseline LSTM model."""
    # Architecture
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10             # early stopping patience

    # Derived
    @property
    def effective_hidden(self) -> int:
        return self.hidden_dim * (2 if self.bidirectional else 1)


# ---------------------------------------------------------------------------
# Convenience: column group builders
# ---------------------------------------------------------------------------

def feature_columns() -> list[str]:
    """All feature columns (ratios + deltas + yoy) — excludes IDs and targets."""
    return Ratios.names() + Ratios.delta_names() + Ratios.yoy_names()


def id_columns() -> list[str]:
    """Identifier columns in the processed dataset."""
    return ["gvkey", "datadate", "fyearq", "fqtr"]


def all_columns() -> list[str]:
    """All columns in the processed Channel A dataset."""
    return id_columns() + feature_columns() + Targets.all()