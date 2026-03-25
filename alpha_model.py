from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import timedelta

from factors import extract_fundamental, compute_momentum
from scorer  import score_universe
from weight_optimizer import WeightOptimizer


class MeridianAlphaModel(AlphaModel):
    """
    Emits Insight objects for every scored stock in the universe.
    Direction, magnitude, and confidence all derived from composite score
    and rank position — no arbitrary quintile cutoffs.

    Designed for BlackLittermanOptimizationPortfolioConstructionModel:
      magnitude  = abs(composite_score) * MAGNITUDE_SCALE
                   → expected return view passed to B-L optimiser
      confidence = 1 - (rank-1)/n
                   → how strongly we hold this view
      direction  = UP / DOWN / FLAT based on composite score sign
    """

    MAGNITUDE_SCALE = 0.20   # score of 1.0 → 20% expected return view
    FLAT_THRESHOLD  = 0.05   # scores within ±0.05 of zero → FLAT
    INSIGHT_PERIOD  = timedelta(days=8)  # slightly longer than weekly rebalance

    def __init__(self, fine_data_ref: dict, optimizer: WeightOptimizer):
        self._fine_data  = fine_data_ref
        self._optimizer  = optimizer
        self._last_score = pd.DataFrame()
        self.Name        = "MeridianFundamentalAlpha"

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> list:
        # Insights emitted via EmitInsights in main.py's scheduler
        # This method exists for framework compliance
        return []

    def generate_insights(self, algorithm: QCAlgorithm) -> list:
        """
        Called directly from WeeklyRebalance in main.py.
        Returns list of Insight objects for all scored symbols.
        """
        symbols = list(self._fine_data.keys())
        if len(symbols) < 20:
            return []

        # ── Update weights monthly ────────────────────────────────────────
        if self._optimizer.should_update():
            self._optimizer.update(symbols)

        # ── Extract features and score ────────────────────────────────────
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

        self._last_score = scored

        # ── Build insights — full universe, score-driven ──────────────────
        insights = []
        n        = len(scored)

        for rank, (symbol, row) in enumerate(scored.iterrows(), start=1):

            # Validate tradeable
            if not algorithm.Securities.ContainsKey(symbol):
                continue
            sec = algorithm.Securities[symbol]
            if not sec.IsTradable or not sec.HasData or sec.Price <= 0:
                continue

            composite = float(row["composite_score"])

            # Direction — score sign, with flat zone around zero
            if composite > self.FLAT_THRESHOLD:
                direction = InsightDirection.UP
            elif composite < -self.FLAT_THRESHOLD:
                direction = InsightDirection.DOWN
            else:
                direction = InsightDirection.FLAT

            # Magnitude — expected return view for Black-Litterman
            # Must not be None or zero for B-L PCM
            magnitude = max(abs(composite) * self.MAGNITUDE_SCALE, 0.001)

            # Confidence — rank position decay (1.0 → ~0.0)
            confidence = 1.0 - (rank - 1) / n
            confidence = max(round(confidence, 4), 0.001)

            # Positional args only — named args broken in QC Python wrapper
            insights.append(
                Insight(
                    symbol,
                    self.INSIGHT_PERIOD,
                    InsightType.PRICE,
                    direction,
                    magnitude,
                    confidence,
                    self.Name
                )
            )

        algorithm.Log(
            f"[alpha] {algorithm.Time.date()} — "
            f"{sum(1 for i in insights if i.Direction == InsightDirection.UP)} UP  "
            f"{sum(1 for i in insights if i.Direction == InsightDirection.DOWN)} DOWN  "
            f"{sum(1 for i in insights if i.Direction == InsightDirection.FLAT)} FLAT  "
            f"from {n} scored stocks"
        )
        return insights

    def OnSecuritiesChanged(
        self, algorithm: QCAlgorithm, changes: SecurityChanges
    ) -> None:
        pass

    @property
    def last_scores(self) -> pd.DataFrame:
        return self._last_score