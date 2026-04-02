import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from AlgorithmImports import *

# 5-element tuple: (group, obj_name, field, direction, accessor)
from .factors import FACTOR_MAP, FACTOR_GROUPS


DEFAULT_WEIGHTS = {
    "value":    0.240,
    "quality":  0.475,
    "growth":   0.186,
    "momentum": 0.100,
}

BLEND_ALPHA = 0.50  # 50% IC-derived, 50% prior


class WeightOptimizer:
    """
    Runs a rolling Spearman IC analysis monthly using fine fundamental
    objects already populated by FineFilter — no separate API call needed.

    Derives GROUP_WEIGHTS empirically and blends with DEFAULT_WEIGHTS.
    Falls back to prior weights on any computation failure.
    """

    def __init__(self, algorithm: QCAlgorithm, fine_data_ref: dict):
        self._algo      = algorithm
        self._fine_data = fine_data_ref  # shared reference with FineFilter
        self._weights   = DEFAULT_WEIGHTS.copy()
        self._last_run  = None

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def weights(self) -> dict:
        return self._weights.copy()

    def should_update(self) -> bool:
        if self._last_run is None:
            return True
        return (self._algo.Time - self._last_run).days >= 28

    def update(self, symbols: list) -> dict:
        try:
            self._algo.Log("[weight_optimizer] Running monthly IC update...")
            new_weights    = self._compute_weights(symbols)
            self._weights  = new_weights
            self._last_run = self._algo.Time
            self._algo.Log(
                "[weight_optimizer] Updated: "
                + "  ".join(f"{k}={v:.3f}" for k, v in new_weights.items())
            )
        except Exception as e:
            self._algo.Log(f"[weight_optimizer] Failed — keeping prior: {e}")
        return self._weights.copy()

    # ── Internal computation ──────────────────────────────────────────────

    def _compute_weights(self, symbols: list) -> dict:
        factor_data = self._get_factor_snapshot(symbols)
        if factor_data.empty:
            self._algo.Log("[weight_optimizer] No factor data — skipping")
            return self._weights.copy()

        returns = self._get_trailing_returns(symbols)
        if returns.empty:
            self._algo.Log("[weight_optimizer] No return data — skipping")
            return self._weights.copy()

        factor_cols = list(FACTOR_MAP.keys())
        df = factor_data.copy()
        df["return"] = returns
        df = df.dropna(subset=["return"])
        df = df.dropna(thresh=int(len(factor_cols) * 0.60))

        if len(df) < 30:
            self._algo.Log(
                f"[weight_optimizer] Only {len(df)} obs — skipping"
            )
            return self._weights.copy()

        # ── Spearman IC per factor ────────────────────────────────────────
        ic_scores = {}
        for factor, (group, obj_name, field, direction, accessor) in FACTOR_MAP.items():
            if factor not in df.columns:
                continue
            valid = df[[factor, "return"]].dropna()
            if len(valid) < 20:
                continue
            try:
                ic, _ = stats.spearmanr(
                    valid[factor] * direction,
                    valid["return"]
                )
                ic = float(ic)
                ic_scores[factor] = max(ic, 0.0) if np.isfinite(ic) else 0.0
            except Exception:
                ic_scores[factor] = 0.0

        total = sum(ic_scores.values())
        if total == 0:
            self._algo.Log("[weight_optimizer] All IC scores zero — skipping")
            return self._weights.copy()

        # Normalise factor IC scores
        ic_norm = {k: v / total for k, v in ic_scores.items()}

        # ── Aggregate to group level ──────────────────────────────────────
        ic_group = {}
        for group, factors in FACTOR_GROUPS.items():
            ic_group[group] = sum(ic_norm.get(f, 0.0) for f in factors)

        # Momentum not in fundamental panel — set to 0 before normalising
        ic_group["momentum"] = 0.0

        group_total = sum(ic_group.values())
        if group_total > 0:
            ic_group = {k: v / group_total for k, v in ic_group.items()}

        # ── Blend with prior ──────────────────────────────────────────────
        blended = {}
        for group in DEFAULT_WEIGHTS:
            blended[group] = (
                BLEND_ALPHA * ic_group.get(group, 0.0)
                + (1 - BLEND_ALPHA) * DEFAULT_WEIGHTS[group]
            )

        blend_total = sum(blended.values())
        if blend_total > 0:
            blended = {k: v / blend_total for k, v in blended.items()}

        # Enforce minimum 5% momentum floor
        if blended.get("momentum", 0) < 0.05:
            blended["momentum"] = 0.05
            t = sum(blended.values())
            blended = {k: v / t for k, v in blended.items()}

        return blended

    def _get_factor_snapshot(self, symbols: list) -> pd.DataFrame:
        """
        Reads factor values directly from fine fundamental objects.
        Navigates the selector path attribute by attribute.
        e.g. "OperationRatios.ROE.OneYear" ->
             f.OperationRatios -> .ROE -> .OneYear
        """
        rows = []

        for symbol in symbols:
            f = self._fine_data.get(symbol)
            if f is None:
                continue

            row = {"symbol": str(symbol)}

            for factor, (group, obj_name, field, direction, accessor) in FACTOR_MAP.items():
                try:
                    obj = getattr(f, obj_name)    # e.g. f.ValuationRatios
                    raw = getattr(obj, field)      # e.g. .PERatio
                    if accessor is not None:
                        raw = getattr(raw, accessor)  # e.g. .OneYear

                    val = float(raw)
                    row[factor] = val if (val != 0 and np.isfinite(val)) else np.nan

                except Exception:
                    row[factor] = np.nan

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).set_index("symbol")

    def _get_trailing_returns(self, symbols: list, days: int = 756) -> pd.Series:
        """
        Trailing N-day return for each symbol using QC History API.
        Used as the IC label in weight computation.
        """
        results = {}

        try:
            history = self._algo.History(
                symbols, days + 60, Resolution.Daily
            )
            if history.empty or "close" not in history.columns:
                return pd.Series(dtype=float)

            closes = history["close"].unstack(level=0)

            for sym in symbols:
                if sym not in closes.columns:
                    continue
                prices = closes[sym].dropna()
                if len(prices) < int(days * 0.8):
                    continue
                p_start = prices.iloc[0]
                p_end   = prices.iloc[min(days - 1, len(prices) - 1)]
                if (p_start != 0
                        and np.isfinite(p_start)
                        and np.isfinite(p_end)):
                    ret = (p_end - p_start) / p_start
                    if np.isfinite(ret):
                        results[str(sym)] = ret

        except Exception as e:
            self._algo.Log(f"[weight_optimizer] Return history error: {e}")

        return pd.Series(results)
