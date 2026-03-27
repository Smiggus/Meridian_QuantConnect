# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
from .factors import FACTOR_MAP, FACTOR_GROUPS


def winsorise(df: pd.DataFrame,
              lower: float = 0.05,
              upper: float = 0.95) -> pd.DataFrame:
    """
    Clips each column to its percentile range.
    Growth factors use tighter bounds — handled here by checking FACTOR_GROUPS.
    """
    df = df.copy()
    growth_factors = set(FACTOR_GROUPS.get("growth", []))

    for col in df.columns:
        lo_q, hi_q = (0.10, 0.90) if col in growth_factors else (lower, upper)
        lo = df[col].quantile(lo_q)
        hi = df[col].quantile(hi_q)
        df[col] = df[col].clip(lo, hi)

    return df


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises each column to mean=0, std=1 across the cross-section.
    NaN z-scores (caused by zero std or all-NaN columns) replaced with 0 (neutral).
    """
    result = (df - df.mean()) / df.std(ddof=1)
    return result.fillna(0.0)


def apply_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flips sign for factors where lower raw value = better.
    After this, higher z-score always means more attractive regardless of factor.
    Unpacks 5-element FACTOR_MAP tuple: (group, obj_name, field, direction, accessor)
    """
    df = df.copy()
    for factor, (group, obj_name, field, direction, accessor) in FACTOR_MAP.items():
        if factor in df.columns and direction == -1:
            df[factor] = df[factor] * -1
    return df


def score_universe(
    features: pd.DataFrame,
    momentum: pd.Series,
    group_weights: dict = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline:
      1. Merge features + momentum
      2. Winsorise
      3. Cross-sectional z-score
      4. Direction adjustment
      5. Group scores (mean z-score per group)
      6. Composite (weighted group average)
      7. Quintile assignment

    group_weights overrides GROUP_WEIGHTS when provided —
    used by WeightOptimizer for dynamic monthly rebalancing.

    Returns DataFrame sorted by composite_score descending,
    with NaN composites dropped before returning.
    """
    from .factors import GROUP_WEIGHTS as DEFAULT_WEIGHTS

    if features.empty:
        return pd.DataFrame()

    weights = group_weights if group_weights is not None else DEFAULT_WEIGHTS

    # ── Step 1: Merge momentum ────────────────────────────────────────────
    df = features.copy()
    df["momentum_12_1"] = momentum

    # All factor columns present in the DataFrame
    all_factor_cols = list(FACTOR_MAP.keys()) + ["momentum_12_1"]
    factor_cols     = [c for c in all_factor_cols if c in df.columns]
    factors         = df[factor_cols].astype(float)

    if factors.dropna(how="all").shape[0] < 10:
        return pd.DataFrame()

    # ── Step 2: Winsorise ─────────────────────────────────────────────────
    factors_w = winsorise(factors)

    # ── Step 3: Z-score ───────────────────────────────────────────────────
    factors_z = cross_sectional_zscore(factors_w)

    # ── Step 4: Direction adjustment ──────────────────────────────────────
    factors_z = apply_direction(factors_z)

    # ── Step 5: Group scores ──────────────────────────────────────────────
    group_scores = pd.DataFrame(index=factors_z.index)
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in factors_z.columns]
        if available:
            group_scores[group] = factors_z[available].mean(axis=1, skipna=True)

    # ── Step 6: Composite ─────────────────────────────────────────────────
    composite = pd.Series(0.0, index=group_scores.index)
    for group, weight in weights.items():
        if group in group_scores.columns:
            # fillna(0) — missing group score treated as neutral, not excluded
            composite += group_scores[group].fillna(0.0) * weight
    composite.name = "composite_score"

    # Replace any inf/-inf that slipped through
    composite = composite.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ── Step 7: Quintiles ─────────────────────────────────────────────────
    n = composite.notna().sum()
    if n >= 5:
        quintiles = pd.qcut(composite, q=5, labels=[1, 2, 3, 4, 5])
    else:
        quintiles = pd.Series(np.nan, index=composite.index)
    quintiles.name = "quintile"

    # ── Assemble and clean ────────────────────────────────────────────────
    result = pd.concat([composite, quintiles, group_scores], axis=1) \
               .sort_values("composite_score", ascending=False)

    # Final guard — drop any row where composite is still NaN
    # (shouldn't happen after fillna above, but belt-and-braces)
    return result.dropna(subset=["composite_score"])