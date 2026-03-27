import numpy as np
import pandas as pd
from datetime import timedelta
from AlgorithmImports import *


# ─────────────────────────────────────────────────────────────────────────────
# Factor definitions
#
# Format: (group, morningstar_field, direction, accessor)
#   group:     which group this factor belongs to
#   field:     attribute name on the Morningstar fundamental object
#   direction: 1 = higher is better, -1 = lower is better
#   accessor:  None = plain float (ValuationRatios)
#              "OneYear" = MultiPeriodField (OperationRatios)
# ─────────────────────────────────────────────────────────────────────────────

FACTOR_MAP = {
    # ── Value ─────────────────────────────────────────────────────────────
    "pe_ratio":        ("value",    "ValuationRatios",  "PERatio",                  -1, None),
    "ev_ebitda":       ("value",    "ValuationRatios",  "EVToEBITDA",               -1, None),
    "pb_ratio":        ("value",    "ValuationRatios",  "PBRatio",                  -1, None),
    "fcf_yield":       ("value",    "ValuationRatios",  "FCFYield",                  1, None),

    # ── Quality ───────────────────────────────────────────────────────────
    "roe":             ("quality",  "OperationRatios",  "ROE",                       1, "OneYear"),
    "roic":            ("quality",  "OperationRatios",  "ROIC",                      1, "OneYear"),
    "gross_margin":    ("quality",  "OperationRatios",  "GrossMargin",               1, "OneYear"),
    "debt_to_equity":  ("quality",  "OperationRatios",  "LongTermDebtEquityRatio",  -1, "OneYear"),
    "income_quality":  ("quality",  "OperationRatios",  "FCFNetIncomeRatio",         1, "OneYear"),

    # ── Growth ────────────────────────────────────────────────────────────
    "revenue_growth":  ("growth",   "OperationRatios",  "RevenueGrowth",             1, "OneYear"),
    "earnings_growth": ("growth",   "OperationRatios",  "NetIncomeGrowth",           1, "OneYear"),
    "fcf_growth":      ("growth",   "OperationRatios",  "FCFGrowth",                 1, "OneYear"),
    "ebitda_growth":   ("growth",   "OperationRatios",  "OperationIncomeGrowth",     1, "OneYear"),
}

# Group membership — derived from FACTOR_MAP for convenience
FACTOR_GROUPS = {
    "value":    [f for f, v in FACTOR_MAP.items() if v[0] == "value"],
    "quality":  [f for f, v in FACTOR_MAP.items() if v[0] == "quality"],
    "growth":   [f for f, v in FACTOR_MAP.items() if v[0] == "growth"],
    "momentum": ["momentum_12_1"],
}

# Empirical weights — derived from 715-observation IC analysis
# Update via WeightOptimizer monthly
GROUP_WEIGHTS = {
    "value":    0.240,
    "quality":  0.475,
    "growth":   0.186,
    "momentum": 0.100,
}


# ─────────────────────────────────────────────────────────────────────────────
# Fundamental extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_fundamental(coarse: list, fine_dict: dict) -> pd.DataFrame:
    """
    Extracts all fundamental factors for each symbol in coarse
    using the fine fundamental objects stored in fine_dict.

    Returns a DataFrame indexed by Symbol with one column per factor.
    Missing or zero values are stored as NaN.
    """
    rows = []

    for symbol in coarse:
        f = fine_dict.get(symbol)
        if f is None:
            continue

        row = {"symbol": symbol}

        for factor_name, (group, obj_name, field, direction, accessor) in FACTOR_MAP.items():
            try:
                obj = getattr(f, obj_name)        # e.g. f.ValuationRatios
                raw = getattr(obj, field)          # e.g. .PERatio
                if accessor is not None:
                    raw = getattr(raw, accessor)   # e.g. .OneYear

                val = float(raw)
                row[factor_name] = val if (val != 0 and np.isfinite(val)) else np.nan

            except Exception:
                row[factor_name] = np.nan

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("symbol")


# ─────────────────────────────────────────────────────────────────────────────
# Momentum
# ─────────────────────────────────────────────────────────────────────────────

def compute_momentum(algorithm: QCAlgorithm, symbols: list,
                     lookback: int = 252) -> pd.Series:
    """
    12-1 month price momentum for each symbol.
    Uses closing prices from History() — multi-index DataFrame.
    Returns a Series indexed by Symbol.
    """
    results = {}

    try:
        history = algorithm.History(symbols, lookback + 30, Resolution.Daily)

        if history.empty or "close" not in history.columns:
            return pd.Series(dtype=float)

        # History returns multi-index ['symbol', 'time'] — unstack symbol level
        closes = history["close"].unstack(level=0)

        for sym in symbols:
            if sym not in closes.columns:
                results[sym] = np.nan
                continue

            prices = closes[sym].dropna()

            if len(prices) < lookback:
                results[sym] = np.nan
                continue

            p_12m = prices.iloc[-lookback]   # price ~12 months ago
            p_1m  = prices.iloc[-21]         # price ~1 month ago (skip most recent)

            if p_12m == 0 or not np.isfinite(p_12m) or not np.isfinite(p_1m):
                results[sym] = np.nan
                continue

            ret = (p_1m - p_12m) / p_12m
            results[sym] = ret if np.isfinite(ret) else np.nan

    except Exception as e:
        algorithm.Log(f"[factors] Momentum error: {e}")

    return pd.Series(results, name="momentum_12_1")