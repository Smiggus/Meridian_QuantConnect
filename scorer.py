# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd
from factors import FACTOR_MAP, FACTOR_GROUPS, GROUP_WEIGHTS


def winsorise(df: pd.DataFrame, lower: float = 0.05, upper: float = 0.95) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df[col] = df[col].clip(lo, hi)
    return df


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=1)


def apply_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for factor, (_, _, direction, _) in FACTOR_MAP.items():
        if factor in df.columns and direction == -1:
            df[factor] = df[factor] * -1
    return df


def score_universe(features: pd.DataFrame, momentum: pd.Series) -> pd.DataFrame:
    """
    Full scoring pipeline. Accepts raw features DataFrame + momentum Series.
    Returns scored DataFrame with composite_score and quintile columns.
    Mirrors scorer.score_universe() from Meridian.
    """
    if features.empty:
        return pd.DataFrame()

    df = features.copy()
    df["momentum_12_1"] = momentum

    factor_cols = list(FACTOR_MAP.keys()) + ["momentum_12_1"]
    factor_cols = [c for c in factor_cols if c in df.columns]
    factors = df[factor_cols].astype(float)

    # Need at least 10 stocks for meaningful z-scores
    valid_rows = factors.dropna(how="all")
    if len(valid_rows) < 10:
        return pd.DataFrame()

    factors_w = winsorise(factors)
    factors_z = cross_sectional_zscore(factors_w)
    factors_z = apply_direction(factors_z)

    # Group scores
    group_scores = pd.DataFrame(index=factors_z.index)
    for group, cols in FACTOR_GROUPS.items():
        available = [c for c in cols if c in factors_z.columns]
        if available:
            group_scores[group] = factors_z[available].mean(axis=1, skipna=True)

    # Composite score
    composite = pd.Series(0.0, index=group_scores.index)
    for group, weight in GROUP_WEIGHTS.items():
        if group in group_scores.columns:
            composite += group_scores[group].fillna(0) * weight
    composite.name = "composite_score"

    # Quintiles
    n = composite.notna().sum()
    if n >= 5:
        quintiles = pd.qcut(composite, q=5, labels=[1, 2, 3, 4, 5])
    else:
        quintiles = pd.Series(np.nan, index=composite.index)
    quintiles.name = "quintile"

    result = pd.concat([composite, quintiles, group_scores], axis=1)
    return result.sort_values("composite_score", ascending=False)