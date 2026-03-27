import numpy as np
import pandas as pd
from datetime import timedelta
from AlgorithmImports import *
from .factors        import extract_fundamental, compute_momentum
from .scorer         import score_universe
from .weight_optimizer import WeightOptimizer


class MeridianAlphaModel(AlphaModel):
    """
    Emits Insight objects for every scored stock in the universe.

    Uses InsightWeightingPortfolioConstructionModel — positions are
    sized by Insight.Weight (normalised composite score).

    Direction driven by composite score sign — no arbitrary quintile cutoff.
    Magnitude and confidence set to None — not used by InsightWeighting PCM.
    """

    FLAT_THRESHOLD = 0.05
    INSIGHT_PERIOD = timedelta(days=8)

    def __init__(self, fine_data_ref: dict, optimizer: WeightOptimizer):
        self._fine_data  = fine_data_ref
        self._optimizer  = optimizer
        self._last_score = pd.DataFrame()
        self.Name        = "MeridianFundamentalAlpha"

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> list:
        return []

    def generate_insights(self, algorithm: QCAlgorithm) -> list:
        symbols = list(self._fine_data.keys())
        if len(symbols) < 20:
            return []

        if self._optimizer.should_update():
            self._optimizer.update(symbols)

        features = extract_fundamental(symbols, self._fine_data)
        if features.empty:
            return []

        momentum = compute_momentum(algorithm, symbols)
        scored   = score_universe(
            features, momentum,
            group_weights=self._optimizer.weights
        )
        if scored.empty:
            return []

        scored = scored.dropna(subset=["composite_score"])
        if scored.empty:
            return []

        self._last_score = scored

        # ── Compute normalised weights from composite scores ──────────────
        long_scores  = scored[scored["composite_score"] >  self.FLAT_THRESHOLD]["composite_score"]
        short_scores = scored[scored["composite_score"] < -self.FLAT_THRESHOLD]["composite_score"].abs()

        long_total  = long_scores.sum()  if not long_scores.empty  else 0
        short_total = short_scores.sum() if not short_scores.empty else 0

        MAX_WEIGHT = 0.05

        insights = []

        for symbol, row in scored.iterrows():
            if not algorithm.Securities.ContainsKey(symbol):
                continue
            sec = algorithm.Securities[symbol]
            if not sec.IsTradable or not sec.HasData or sec.Price <= 0:
                continue

            composite = float(row["composite_score"])
            if not np.isfinite(composite):
                continue

            if composite > self.FLAT_THRESHOLD:
                direction = InsightDirection.UP
                weight    = min(composite / long_total,  MAX_WEIGHT) \
                            if long_total > 0 else 0.0
            elif composite < -self.FLAT_THRESHOLD:
                direction = InsightDirection.DOWN
                weight    = min(abs(composite) / short_total, MAX_WEIGHT) \
                            if short_total > 0 else 0.0
            else:
                continue

            if weight <= 0 or not np.isfinite(weight):
                continue

            # Positional args only — named args broken in QC Python wrapper
            # Insight(symbol, period, type, direction, magnitude,
            #         confidence, sourceModel, weight)
            insights.append(
                Insight(
                    symbol,
                    self.INSIGHT_PERIOD,
                    InsightType.PRICE,
                    direction,
                    None,
                    None,
                    self.Name,
                    float(weight),
                )
            )

        up   = sum(1 for i in insights if i.Direction == InsightDirection.UP)
        down = sum(1 for i in insights if i.Direction == InsightDirection.DOWN)
        algorithm.Log(
            f"[alpha] {algorithm.Time.date()} — "
            f"{up} UP  {down} DOWN  from {len(scored)} scored"
        )
        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        pass

    @property
    def last_scores(self) -> pd.DataFrame:
        return self._last_score