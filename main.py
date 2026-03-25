from AlgorithmImports import *
# Use C# implementation — Python B-L has a known bug
# from Portfolio.BlackLittermanOptimizationPortfolioConstructionModel \
#     import BlackLittermanOptimizationPortfolioConstructionModel

import pandas as pd
import numpy as np
from datetime import timedelta

from universe       import coarse_filter, fine_filter
from alpha_model    import MeridianAlphaModel
from weight_optimizer import WeightOptimizer
from risk_model     import MeridianRiskModel


class MeridianAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1_000_000)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # ── Universe ──────────────────────────────────────────────────────
        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.UniverseSettings.Resolution = Resolution.Daily

        # ── Shared state ──────────────────────────────────────────────────
        self._fine_data = {}
        self._optimizer = WeightOptimizer(self)
        self._alpha     = MeridianAlphaModel(self._fine_data, self._optimizer)

        # ── Algorithm Framework ───────────────────────────────────────────

        # Alpha
        self.AddAlpha(self._alpha)

        # Portfolio construction — Black-Litterman
        # Uses magnitude as the "investor view" and confidence as view uncertainty
        # portfolioBias = LONG_SHORT allows shorts on DOWN insights
        # tau = 0.05 — weight-on-views scalar (higher = more trust in views)
        self.SetPortfolioConstruction(
            BlackLittermanOptimizationPortfolioConstructionModel(
                Resolution.Daily,      # rebalance frequency
                PortfolioBias.LONG_SHORT,
                1,                     # lookback (periods)
                Resolution.Daily,      # lookback resolution
                0.0,                   # risk free rate
                0.05,                  # delta — risk aversion
                0.05                   # tau — weight on views
            )
        )

        # Execution — VWAP for large orders, Immediate as fallback
        # Use ImmediateExecutionModel initially, swap to VolumeWeightedAveragePriceExecutionModel
        # once satisfied with signals
        self.SetExecution(ImmediateExecutionModel())

        # Risk management
        self.AddRiskManagement(MeridianRiskModel())

        # ── Weekly rebalance ──────────────────────────────────────────────
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.WeeklyRebalance
        )

        # ── Benchmark + warmup ────────────────────────────────────────────
        self.SetBenchmark("SPY")
        self.SetWarmUp(timedelta(days=400))

        self.Log(
            "[meridian] Initialized — "
            "Black-Litterman PCM | full-universe insights | "
            "sector + drawdown risk model"
        )

    # ── Universe selection ─────────────────────────────────────────────────

    def CoarseFilter(self, coarse):
        return coarse_filter(coarse)

    def FineFilter(self, fine):
        self._fine_data.clear()
        self._fine_data.update({x.Symbol: x for x in fine})
        return fine_filter(fine)

    # ── Scheduled rebalance ────────────────────────────────────────────────

    def WeeklyRebalance(self):
        if self.IsWarmingUp:
            return

        if len(self._fine_data) < 20:
            self.Log(
                f"[meridian] Skipping — only {len(self._fine_data)} symbols"
            )
            return

        # Generate and emit insights from the alpha model
        insights = self._alpha.generate_insights(self)

        if not insights:
            self.Log("[meridian] No insights generated this week")
            return

        self.EmitInsights(insights)

        # Log weight update if it happened
        w = self._optimizer.weights
        self.Log(
            f"[meridian] {self.Time.date()} — "
            f"{len(insights)} insights emitted | "
            f"weights: V={w['value']:.2f} Q={w['quality']:.2f} "
            f"G={w['growth']:.2f} M={w['momentum']:.2f}"
        )

    def OnData(self, data):
        pass  # all logic schedule-driven