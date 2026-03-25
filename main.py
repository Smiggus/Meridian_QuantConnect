from AlgorithmImports import *
import pandas as pd
import numpy as np

from universe import coarse_filter, fine_filter
from factors import extract_fundamental, compute_momentum
from scorer import score_universe
from portfolio import build_targets, get_exit_targets


class MeridianAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(1000000)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # Universe selection
        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.UniverseSettings.Resolution = Resolution.Daily

        # State
        self._fine_data   = {}
        self._last_scored = pd.DataFrame()
        self._rebalance   = False

        # Weekly rebalance — every Monday at market open
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.Rebalance
        )

        # Benchmark
        self.SetBenchmark("SPY")
        self.SetWarmUp(timedelta(days=400))

        self.Debug("[meridian] Initialized")

    def CoarseFilter(self, coarse):
        return coarse_filter(coarse)

    def FineFilter(self, fine):
        self._fine_data = {x.Symbol: x for x in fine}
        return fine_filter(fine)

    def OnData(self, data):
        pass  # All logic is schedule-driven

    def Rebalance(self):
        if self.IsWarmingUp:
            return

        symbols = list(self._fine_data.keys())
        if len(symbols) < 20:
            self.Debug(f"[meridian] Skipping rebalance — only {len(symbols)} symbols")
            return

        self.Debug(f"[meridian] Rebalancing {len(symbols)} symbols")

        # ── Step 1: Extract fundamentals ──────────────────────────────────
        features = extract_fundamental(symbols, self._fine_data)
        if features.empty:
            self.Debug("[meridian] No fundamental data — skipping")
            return

        # ── Step 2: Compute momentum ───────────────────────────────────────
        momentum = compute_momentum(self, symbols)

        # ── Step 3: Score universe ─────────────────────────────────────────
        scored = score_universe(features, momentum)
        if scored.empty:
            self.Debug("[meridian] Scoring returned empty — skipping")
            return

        self._last_scored = scored

        # ── Step 4: Build portfolio targets ───────────────────────────────
        targets = build_targets(scored, long_q=5, short_q=1, max_position=0.05)

        # Add exits for positions no longer in signal
        current = [x.Key for x in self.Portfolio if self.Portfolio[x.Key].Invested]
        exits   = get_exit_targets(current, targets)
        targets.update(exits)

        if not targets:
            return


        # ── Step 5: Execute ────────────────────────────────────────────────
        valid_targets = {
            symbol: weight
            for symbol, weight in targets.items()
            if self.Securities.ContainsKey(symbol)
            and self.Securities[symbol].IsTradable
            and self.Securities[symbol].HasData
            and self.Securities[symbol].Price > 0
        }

        if not valid_targets:
            self.Debug("[meridian] No valid tradeable targets this rebalance")
            return

        self.SetHoldings(
            [PortfolioTarget(symbol, weight)
            for symbol, weight in valid_targets.items()],
            True
        )

        long_count  = sum(1 for w in valid_targets.values() if w > 0)
        short_count = sum(1 for w in valid_targets.values() if w < 0)
        skipped     = len(targets) - len(valid_targets)
        self.Debug(f"[meridian] Long: {long_count}  Short: {short_count}  Skipped: {skipped}")
        
    def OnEndOfAlgorithm(self):
        self.Debug("[meridian] Final portfolio summary:")
        for symbol, holding in self.Portfolio.items():
            if holding.Invested:
                self.Debug(f"  {symbol}: {holding.Quantity} @ {holding.AveragePrice:.2f}")