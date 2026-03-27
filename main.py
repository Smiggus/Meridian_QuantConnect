from AlgorithmImports import *
from datetime import timedelta
import numpy as np

from universe        import coarse_filter, fine_filter
from risk_model      import MeridianRiskModel
from meridian_alpha  import MeridianAlphaModel, WeightOptimizer


class MeridianAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetCash(1_000_000)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.UniverseSettings.Resolution = Resolution.Daily

        # Shared state
        self._fine_data = {}
        self._optimizer = WeightOptimizer(self, self._fine_data)
        self._alpha     = MeridianAlphaModel(self._fine_data, self._optimizer)

        # ── Algorithm Framework ───────────────────────────────────────────
        self.AddAlpha(self._alpha)

        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel(Resolution.Daily))
        #self.SetPortfolioConstruction(BlackLittermanOptimizationPortfolioConstructionModel(rebalance=Resolution.Daily))

        self.add_risk_management(MaximumDrawdownPercentPerSecurity())
        self.add_risk_management(MaximumSectorExposureRiskManagementModel())
        #self.add_risk_management(MaximumDrawdownPercentPortfolio(maximum_drawdown_percent: float = 0.2, is_trailing: bool = True))

        self.SetExecution(ImmediateExecutionModel())
        # self.AddRiskManagement(MeridianRiskModel())  # disabled for testing

        # Weekly rebalance
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.WeeklyRebalance
        )

        self.SetBenchmark("SPY")
        self.SetWarmUp(timedelta(days=400))
        self.Log("[meridian] Initialized — InsightWeighting PCM")

    def CoarseFilter(self, coarse):
        return coarse_filter(coarse)

    def FineFilter(self, fine):
        self._fine_data.clear()
        self._fine_data.update({x.Symbol: x for x in fine})
        return fine_filter(fine)

    def WeeklyRebalance(self):
        if self.IsWarmingUp:
            return
        if len(self._fine_data) < 20:
            return

        insights = self._alpha.generate_insights(self)
        if not insights:
            self.Log("[meridian] No insights this week")
            return

        self.EmitInsights(insights)

        w = self._optimizer.weights
        self.Log(
            f"[meridian] {self.Time.date()} — "
            f"{len(insights)} insights emitted | "
            f"V={w['value']:.2f} Q={w['quality']:.2f} "
            f"G={w['growth']:.2f} M={w['momentum']:.2f}"
        )

    def OnData(self, data):
        pass