import numpy as np
import pandas as pd
from AlgorithmImports import *


# (group, field, direction, accessor)
# accessor: None = plain float, "OneYear" = MultiPeriodField
FACTOR_MAP = {
    "pe_ratio":       ("ValuationRatios",  "PERatio",                 -1, None),
    "ev_ebitda":      ("ValuationRatios",  "EVToEBITDA",              -1, None),
    "pb_ratio":       ("ValuationRatios",  "PBRatio",                 -1, None),
    "fcf_yield":      ("ValuationRatios",  "FCFYield",                 1, None),
    "roe":            ("OperationRatios",  "ROE",                      1, "OneYear"),
    "roic":           ("OperationRatios",  "ROIC",                     1, "OneYear"),
    "gross_margin":   ("OperationRatios",  "GrossMargin",              1, "OneYear"),
    "debt_to_equity": ("OperationRatios",  "LongTermDebtEquityRatio",  -1, "OneYear"),
}

FACTOR_GROUPS = {
    "value":    ["pe_ratio", "ev_ebitda", "pb_ratio", "fcf_yield"],
    "quality":  ["roe", "roic", "gross_margin", "debt_to_equity"],
    "momentum": ["momentum_12_1"],
}

# IC-derived blended weights — update from feature_selection.ipynb output
GROUP_WEIGHTS = {
    "value":    0.40,
    "quality":  0.40,
    "momentum": 0.20,
}


def extract_fundamental(coarse, fine_dict: dict) -> pd.DataFrame:
    rows = []

    for symbol in coarse:
        f = fine_dict.get(symbol)
        if f is None:
            continue

        row = {"symbol": symbol}

        for factor, (group, field, _, accessor) in FACTOR_MAP.items():
            try:
                obj = getattr(f, group)
                raw = getattr(obj, field)

                # MultiPeriodField — call .OneYear to get float
                if accessor is not None:
                    raw = getattr(raw, accessor)

                val = float(raw)
                # QC returns 0 for missing data — treat as NaN
                row[factor] = val if val != 0 else np.nan
            except Exception:
                row[factor] = np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()

def compute_momentum(algorithm, symbols: list, lookback: int = 252) -> pd.Series:
    """
    12-1 month momentum for a list of symbols.
    Uses QC History() call — same formula as Meridian's compute_momentum().
    """
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
            # 12-1 month: price 252 days ago to price 21 days ago
            p_12m = prices.iloc[-lookback]
            p_1m  = prices.iloc[-21]
            if p_12m == 0:
                results[symbol] = np.nan
                continue
            results[symbol] = (p_1m - p_12m) / p_12m

    except Exception as e:
        algorithm.Debug(f"[factors] Momentum error: {e}")

    return pd.Series(results, name="momentum_12_1")