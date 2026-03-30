from AlgorithmImports import *
from datetime import timedelta
import json
import os
import sys

import numpy as np

# ── lean-data-platform path ───────────────────────────────────────────────────
# Resolved relative to this file: Projects/Algo/Meridian_QuantConnect/ → ../../
# Override with LEAN_PLATFORM_PATH env var if the repo lives elsewhere.
_LEAN_PLATFORM_PATH = os.environ.get("LEAN_PLATFORM_PATH") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "lean-data-platform")
)
if os.path.isdir(_LEAN_PLATFORM_PATH) and _LEAN_PLATFORM_PATH not in sys.path:
    sys.path.insert(0, _LEAN_PLATFORM_PATH)

try:
    from lean_pipeline.coverage_checker import CoverageChecker
    from lean_pipeline.pipeline_client  import PipelineClient
    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False

from universe       import coarse_filter, fine_filter
from risk_model     import MeridianRiskModel
from meridian_alpha import MeridianAlphaModel, WeightOptimizer
from meridian_alpha.local_fundamental_adapter import LocalFundamentalAdapter


class MeridianAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2021, 12, 31)
        self.SetCash(1_000_000)

        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage,
            AccountType.Margin
        )

        # Shared state — populated by FineFilter (cloud) or
        # _load_local_fundamentals (local) before each weekly rebalance.
        self._fine_data     = {}
        self._local_symbols = {}   # ticker str → Symbol  (local mode only)

        self.is_local = (
            _PIPELINE_AVAILABLE
            and os.environ.get("QC_RUN_ENV", "").strip().lower() == "local"
        )

        if self.is_local:
            self._init_local_data()
        else:
            self._init_cloud_data()

        self.Log(
            f"[meridian] Initialized — "
            f"{'local pipeline (Databento + FMP)' if self.is_local else 'cloud (QC)'} | "
            f"cash={self.Portfolio.Cash:,.0f}"
        )

    # ── Cloud initialisation (unchanged from main) ────────────────────────────

    def _init_cloud_data(self) -> None:
        self.AddUniverse(self.CoarseFilter, self.FineFilter)
        self.UniverseSettings.Resolution = Resolution.Daily
        self._setup_framework()

    def CoarseFilter(self, coarse):
        return coarse_filter(coarse)

    def FineFilter(self, fine):
        self._fine_data.clear()
        self._fine_data.update({x.Symbol: x for x in fine})
        return fine_filter(fine)

    # ── Local initialisation ──────────────────────────────────────────────────

    def _init_local_data(self) -> None:
        """
        1. Load Russell 2000 universe from lean-data-platform/manifests/watchlist.json.
        2. Check PostgreSQL coverage for each ticker via CoverageChecker.
        3. Fire Databento (OHLCV) and FMP (fundamentals) Dagster jobs for any gaps.
        4. If all data is present: subscribe equities and set up the framework.

        Environment variables:
            QC_RUN_ENV=local          — activates this path
            LEAN_DATA_ROOT            — LEAN data directory (default: /app/data)
            LEAN_PLATFORM_PATH        — override path to lean-data-platform repo
            DAGSTER_HOST / DAGSTER_PORT — Dagster webserver (default: localhost:3000)
            PGHOST / PGPORT / PGDB / PGUSER / PGPASS — PostgreSQL credentials
        """
        lean_root      = os.environ.get("LEAN_DATA_ROOT", "/app/data")
        watchlist_path = os.path.join(_LEAN_PLATFORM_PATH, "manifests", "watchlist.json")

        with open(watchlist_path) as fh:
            watchlist = json.load(fh)
        universe = watchlist.get("universe", [])
        self.Log(f"[meridian] Universe: {len(universe)} tickers from watchlist")

        checker = CoverageChecker()
        client  = PipelineClient()

        start = self.StartDate.date()
        end   = self.EndDate.date()

        missing_ohlcv = []
        missing_funds = []

        for entry in universe:
            ticker  = entry["ticker"]
            dataset = entry.get("databento_dataset", "XNAS.ITCH")

            if not checker.is_ohlcv_covered(ticker, start, end):
                client.request_ohlcv(
                    ticker=ticker,
                    start_date=start,
                    end_date=end,
                    dataset=dataset,
                    lean_data_root=lean_root,
                )
                missing_ohlcv.append(ticker)

            if not checker.is_fundamentals_covered(ticker):
                client.request_fundamentals(ticker=ticker, lean_data_root=lean_root)
                missing_funds.append(ticker)

        if missing_ohlcv or missing_funds:
            msg = (
                f"[meridian] Data missing — "
                f"OHLCV: {len(missing_ohlcv)} tickers, "
                f"Fundamentals: {len(missing_funds)} tickers. "
                "Dagster jobs fired. Re-run after ingestion completes."
            )
            self.Log(msg)
            self.Quit(msg)
            return

        # All data confirmed present — subscribe equities (reads from LEAN ZIPs
        # written by the Databento pipeline's LeanDataWriter)
        for entry in universe:
            ticker = entry["ticker"]
            try:
                sym = self.AddEquity(ticker, Resolution.Daily).Symbol
                self._local_symbols[ticker] = sym
            except Exception as exc:
                self.Log(f"[meridian] Skipping {ticker}: {exc}")

        self.Log(f"[meridian] {len(self._local_symbols)} equities subscribed from local data")
        self._setup_framework()

    def _load_local_fundamentals(self) -> dict:
        """
        Reads the two most recent LEAN fine fundamental JSONs per ticker and
        returns { Symbol → LocalFundamentalAdapter }.

        Files live at: LEAN_DATA_ROOT/fundamental/fine/{ticker}/{YYYYMMDD}.json
        Written by lean-data-platform's FMP pipeline via LeanDataWriter.

        Called at the start of each WeeklyRebalance in local mode.
        """
        lean_root = os.environ.get("LEAN_DATA_ROOT", "/app/data")
        result    = {}

        for ticker, symbol in self._local_symbols.items():
            fine_dir = os.path.join(lean_root, "fundamental", "fine", ticker.lower())
            if not os.path.isdir(fine_dir):
                continue

            try:
                # YYYYMMDD.json filenames sort correctly as strings
                files = sorted(
                    [f for f in os.listdir(fine_dir) if f.endswith(".json")],
                    reverse=True,
                )
            except OSError:
                continue

            if not files:
                continue

            records = []
            for fname in files[:2]:  # newest + prior period for YoY growth factors
                try:
                    with open(os.path.join(fine_dir, fname)) as fh:
                        obj = json.load(fh)

                    income   = obj.get("FinancialStatements", {}).get("IncomeStatement",   {})
                    balance  = obj.get("FinancialStatements", {}).get("BalanceSheet",       {})
                    cashflow = obj.get("FinancialStatements", {}).get("CashFlowStatement", {})
                    val      = obj.get("ValuationRatios", {})

                    records.append({
                        # Income statement
                        "revenue":             income.get("TotalRevenue"),
                        "gross_profit":        income.get("GrossProfit"),
                        "ebitda":              income.get("Ebitda"),
                        "net_income":          income.get("NetIncome"),
                        "operating_income":    income.get("OperatingIncome"),
                        # Balance sheet
                        "total_assets":        balance.get("TotalAssets"),
                        "equity":              balance.get("CommonStockEquity"),
                        "total_debt":          balance.get("TotalDebt"),
                        # Cash flow
                        "free_cash_flow":      cashflow.get("FreeCashFlow"),
                        "operating_cash_flow": cashflow.get("OperatingCashFlow"),
                        # Pre-computed valuation ratios (from FMP via LeanDataWriter)
                        "pe_ratio":            val.get("PERatio"),
                        "pb_ratio":            val.get("PBRatio"),
                        "ev_ebitda":           val.get("EVToEBITDA"),
                        "roe":                 val.get("ReturnOnEquity"),
                        "roa":                 val.get("ReturnOnAssets"),
                        "debt_equity":         val.get("DebtToEquityRatio"),
                    })
                except Exception:
                    continue

            if records:
                result[symbol] = LocalFundamentalAdapter(records)

        return result

    # ── Shared framework setup ────────────────────────────────────────────────

    def _setup_framework(self) -> None:
        """
        Wires up alpha, PCM, risk management, execution, and schedule.
        Called by both _init_cloud_data and _init_local_data.
        """
        self._optimizer = WeightOptimizer(self, self._fine_data)
        self._alpha     = MeridianAlphaModel(self._fine_data, self._optimizer)

        self.AddAlpha(self._alpha)
        self.SetPortfolioConstruction(
            InsightWeightingPortfolioConstructionModel(Resolution.Daily)
        )
        # self.SetPortfolioConstruction(
        #     BlackLittermanOptimizationPortfolioConstructionModel(rebalance=Resolution.Daily)
        # )

        self.add_risk_management(MaximumDrawdownPercentPerSecurity())
        self.add_risk_management(MaximumSectorExposureRiskManagementModel())
        # self.add_risk_management(MaximumDrawdownPercentPortfolio(...))
        # self.AddRiskManagement(MeridianRiskModel())  # disabled for testing

        self.SetExecution(ImmediateExecutionModel())

        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.AfterMarketOpen("SPY", 30),
            self.WeeklyRebalance,
        )

        self.SetBenchmark("SPY")
        self.SetWarmUp(timedelta(days=400))

    # ── Weekly rebalance ──────────────────────────────────────────────────────

    def WeeklyRebalance(self):
        if self.IsWarmingUp:
            return

        if self.is_local:
            # Refresh fundamentals from FMP JSON files before scoring
            self._fine_data.clear()
            self._fine_data.update(self._load_local_fundamentals())
            if len(self._fine_data) < 20:
                self.Log("[meridian] Skipping — insufficient local fundamental data loaded")
                return
        else:
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
