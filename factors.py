import numpy as np
import pandas as pd
from AlgorithmImports import *


# ─────────────────────────────────────────────
# Updated FACTOR_MAP — growth now sourced from
# OperationRatios directly, not FinancialStatements
# ─────────────────────────────────────────────

FACTOR_MAP = {
    # Value (4)
    "pe_ratio":        ("ValuationRatios",  "PERatio",                  -1, None),
    "ev_ebitda":       ("ValuationRatios",  "EVToEBITDA",               -1, None),
    "pb_ratio":        ("ValuationRatios",  "PBRatio",                  -1, None),
    "fcf_yield":       ("ValuationRatios",  "FCFYield",                  1, None),

    # Quality (5)
    "roe":             ("OperationRatios",  "ROE",                       1, "OneYear"),
    "roic":            ("OperationRatios",  "ROIC",                      1, "OneYear"),
    "gross_margin":    ("OperationRatios",  "GrossMargin",               1, "OneYear"),
    "debt_to_equity":  ("OperationRatios",  "LongTermDebtEquityRatio",  -1, "OneYear"),
    "income_quality":  ("OperationRatios",  "FCFNetIncomeRatio",         1, "OneYear"),

    # Growth (4) — all from OperationRatios, pre-computed by Morningstar
    "revenue_growth":  ("OperationRatios",  "RevenueGrowth",             1, "OneYear"),
    "earnings_growth": ("OperationRatios",  "NetIncomeGrowth",           1, "OneYear"),
    "fcf_growth":      ("OperationRatios",  "FCFGrowth",                 1, "OneYear"),
    "ebitda_growth":   ("OperationRatios",  "OperationIncomeGrowth",     1, "OneYear"),
}
FACTOR_GROUPS = {
    "value":    ["pe_ratio", "ev_ebitda", "pb_ratio", "fcf_yield"],
    "quality":  ["roe", "roic", "gross_margin", "debt_to_equity", "income_quality"],
    "growth":   ["revenue_growth", "earnings_growth", "fcf_growth", "ebitda_growth"],
    "momentum": ["momentum_12_1"],
}

# Derived from meridian_research.ipynb analysis of factor correlations and predictive power
GROUP_WEIGHTS = {
    "value":    0.240,
    "quality":  0.475,
    "growth":   0.186,
    "momentum": 0.100,
}

GROWTH_FACTOR_WINSOR = (0.10, 0.90)


# ─────────────────────────────────────────────
# Fundamental extraction (ValuationRatios + OperationRatios)
# ─────────────────────────────────────────────

def extract_fundamental(coarse, fine_dict: dict) -> pd.DataFrame:
    rows = []

    for symbol in coarse:
        f = fine_dict.get(symbol)
        if f is None:
            continue

        row = {"symbol": symbol}

        for factor, (group, field, _, accessor) in FACTOR_MAP.items():
            if group is None:
                # Growth factors — computed separately
                row[factor] = np.nan
                continue
            try:
                obj = getattr(f, group)
                raw = getattr(obj, field)
                if accessor is not None:
                    raw = getattr(raw, accessor)
                val = float(raw)
                row[factor] = val if val != 0 else np.nan
            except Exception:
                row[factor] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()

# ─────────────────────────────────────────────
# Momentum
# ─────────────────────────────────────────────

def compute_momentum(algorithm, symbols: list, lookback: int = 252) -> pd.Series:
    """12-1 month price momentum."""
    results = {}
    try:
        history = algorithm.History(symbols, lookback + 30, Resolution.Daily)
        if history.empty:
            return pd.Series(dtype=float)

        closes = history["close"].unstack(level=0)

        for symbol in symbols:
            if symbol not in closes.columns:
                results[symbol] = np.nan
                continue
            prices = closes[symbol].dropna()
            if len(prices) < lookback:
                results[symbol] = np.nan
                continue
            p_12m = prices.iloc[-lookback]
            p_1m  = prices.iloc[-21]
            if p_12m == 0:
                results[symbol] = np.nan
                continue
            results[symbol] = (p_1m - p_12m) / p_12m

    except Exception as e:
        algorithm.Debug(f"[factors] Momentum error: {e}")

    return pd.Series(results, name="momentum_12_1")