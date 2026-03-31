from AlgorithmImports import *
from datetime import timedelta
import json
import os
import platform

import numpy as np

from lean_pipeline.coverage_checker import CoverageChecker
from lean_pipeline.pipeline_client  import PipelineClient

from universe       import coarse_filter, fine_filter
from risk_model     import MeridianRiskModel
from meridian_alpha import MeridianAlphaModel, WeightOptimizer
from meridian_alpha.local_fundamental_adapter import LocalFundamentalAdapter

# Hostnames that identify a local development machine.
# Add additional hostnames here if running on more than one local box.
_LOCAL_HOSTNAMES = {"HAL-107"}


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

        self.is_local = platform.node() in _LOCAL_HOSTNAMES

        self.Log(f"[meridian] Host: {platform.node()} — "
                 f"{'local pipeline (Databento + FMP)' if self.is_local else 'cloud (QC)'}")

        if self.is_local:
            self._init_local_data()
        else:
            self._init_cloud_data()

    # ── Cloud initialisation (unchanged) ─────────────────────────────────────

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
        1. Load Russell 2000 universe from manifests/watchlist.json.
        2. Check PostgreSQL coverage via CoverageChecker (reads pg-* parameters).
        3. Fire Databento / FMP Dagster jobs for any missing data.
        4. If all data is present: subscribe equities and set up the framework.

        Connection config is read from QC algorithm parameters (config.json):
            pg-host, pg-port, pg-db, pg-user, pg-pass
            dagster-host, dagster-port

        LEAN data root: /Data  (mounted by lean CLI from Algo/data/)
        Re-run after Dagster completes ingestion if data is missing.
        """
        # Bridge QC parameters → env vars so CoverageChecker / PipelineClient
        # pick them up without modification.
        for param, env_var in [
            ("pg-host",      "PGHOST"),
            ("pg-port",      "PGPORT"),
            ("pg-db",        "PGDB"),
            ("pg-user",      "PGUSER"),
            ("pg-pass",      "PGPASS"),
            ("dagster-host", "DAGSTER_HOST"),
            ("dagster-port", "DAGSTER_PORT"),
        ]:
            value = self.GetParameter(param)
            if value:
                os.environ[env_var] = value

        lean_root = "/Data"

        watchlist_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "manifests", "watchlist.json"
        )

        with open(watchlist_path) as fh:
            watchlist = json.load(fh)
        universe = watchlist.get("universe", [])
        self.Log(f"[meridian] Universe: {len(universe)} tickers from watchlist")

        checker  = CoverageChecker()
        client   = PipelineClient()
        start    = self.StartDate.date()
        end      = self.EndDate.date()

        all_tickers  = [e["ticker"] for e in universe]
        dataset_map  = {e["ticker"]: e.get("databento_dataset", "XNAS.ITCH") for e in universe}

        # Two bulk queries — one round-trip each regardless of universe size
        self.Log("[meridian] Checking coverage (bulk)...")
        missing_ohlcv = checker.get_uncovered_ohlcv(all_tickers, start, end)
        missing_funds = checker.get_uncovered_fundamentals(all_tickers)
        self.Log(f"[meridian] OHLCV missing: {len(missing_ohlcv)} | Fundamentals missing: {len(missing_funds)}")

        if missing_ohlcv or missing_funds:
            msg = (
                f"[meridian] Data not ready — "
                f"OHLCV: {len(missing_ohlcv)} tickers, "
                f"Fundamentals: {len(missing_funds)} tickers. "
                "Trigger bulk ingest via Dagster UI (http://192.168.17.4:3000) "
                "then re-run."
            )
            self.Log(msg)
            self.Quit(msg)
            return

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
        Files: /Data/fundamental/fine/{ticker}/{YYYYMMDD}.json
        """
        result = {}

        for ticker, symbol in self._local_symbols.items():
            fine_dir = os.path.join("/Data", "fundamental", "fine", ticker.lower())
            if not os.path.isdir(fine_dir):
                continue

            try:
                files = sorted(
                    [f for f in os.listdir(fine_dir) if f.endswith(".json")],
                    reverse=True,
                )
            except OSError:
                continue

            if not files:
                continue

            records = []
            for fname in files[:2]:
                try:
                    with open(os.path.join(fine_dir, fname)) as fh:
                        obj = json.load(fh)

                    income   = obj.get("FinancialStatements", {}).get("IncomeStatement",   {})
                    balance  = obj.get("FinancialStatements", {}).get("BalanceSheet",       {})
                    cashflow = obj.get("FinancialStatements", {}).get("CashFlowStatement", {})
                    val      = obj.get("ValuationRatios", {})

                    records.append({
                        "revenue":             income.get("TotalRevenue"),
                        "gross_profit":        income.get("GrossProfit"),
                        "ebitda":              income.get("Ebitda"),
                        "net_income":          income.get("NetIncome"),
                        "operating_income":    income.get("OperatingIncome"),
                        "total_assets":        balance.get("TotalAssets"),
                        "equity":              balance.get("CommonStockEquity"),
                        "total_debt":          balance.get("TotalDebt"),
                        "free_cash_flow":      cashflow.get("FreeCashFlow"),
                        "operating_cash_flow": cashflow.get("OperatingCashFlow"),
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
            self._fine_data.clear()
            self._fine_data.update(self._load_local_fundamentals())
            if len(self._fine_data) < 20:
                self.Log("[meridian] Skipping — insufficient local fundamental data")
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
