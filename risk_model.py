from AlgorithmImports import *
import numpy as np


# Max sector weight as fraction of gross portfolio value
MAX_SECTOR_WEIGHT = 0.30   # no sector > 30%

# Max single position weight
MAX_POSITION_WEIGHT = 0.05  # no single stock > 5%

# Portfolio drawdown halt — liquidate if NAV drops this much from peak
MAX_DRAWDOWN_HALT = 0.15   # 15% peak-to-trough halt


class MeridianRiskModel(RiskManagementModel):
    """
    Three-layer risk management:
      1. Sector exposure cap  — reduces overweight sector positions
      2. Single position cap  — clips any position above MAX_POSITION_WEIGHT
      3. Drawdown halt        — liquidates all on MAX_DRAWDOWN_HALT breach

    Sector classification uses Morningstar sector codes from QC fundamentals.
    """

    def __init__(self):
        self._peak_value = None

    def ManageRisk(self, algorithm, targets):

        portfolio_value = float(algorithm.Portfolio.TotalPortfolioValue)
        if portfolio_value <= 0:
            return []

        # Update peak
        if self._peak_value is None or portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        # ── Rule 3: Drawdown halt ─────────────────────────────────────────
        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        if drawdown >= MAX_DRAWDOWN_HALT:
            algorithm.Log(
                f"[risk] Drawdown halt triggered: "
                f"{drawdown:.1%} from peak ${self._peak_value:,.0f}"
            )
            return [
                PortfolioTarget(x.Key, 0)
                for x in algorithm.Portfolio
                if x.Value.Invested
            ]

        adjusted = []

        # ── Rule 2: Single position cap ───────────────────────────────────
        # target.Quantity from InsightWeightingPortfolioConstructionModel is a
        # portfolio fraction (e.g. 0.05 = 5%), so compare directly to the cap.
        for target in targets:
            pct = abs(target.Quantity)
            if pct > MAX_POSITION_WEIGHT:
                sign = 1 if target.Quantity > 0 else -1
                adjusted.append(
                    PortfolioTarget(target.Symbol, MAX_POSITION_WEIGHT * sign)
                )
            else:
                adjusted.append(target)

        # ── Rule 1: Sector exposure cap ───────────────────────────────────
        sector_weights = {}

        for target in adjusted:
            sym = target.Symbol
            if not algorithm.Securities.ContainsKey(sym):
                continue
            sec = algorithm.Securities[sym]
            if sec.Fundamentals is None:
                continue
            try:
                sector_code = int(
                    sec.Fundamentals.AssetClassification
                    .MorningstarSectorCode
                )
            except Exception:
                continue
            sector_weights.setdefault(sector_code, 0.0)
            sector_weights[sector_code] += abs(float(target.Quantity))

        overweight = {
            s: w for s, w in sector_weights.items()
            if w > MAX_SECTOR_WEIGHT
        }

        if not overweight:
            return adjusted

        # Scale down positions in overweight sectors proportionally
        final = []
        for target in adjusted:
            sym = target.Symbol
            if not algorithm.Securities.ContainsKey(sym):
                final.append(target)
                continue
            sec = algorithm.Securities[sym]
            if sec.Fundamentals is None:
                final.append(target)
                continue
            try:
                sector_code = int(
                    sec.Fundamentals.AssetClassification
                    .MorningstarSectorCode
                )
            except Exception:
                final.append(target)
                continue

            if sector_code in overweight:
                scale   = MAX_SECTOR_WEIGHT / overweight[sector_code]
                new_qty = float(target.Quantity) * scale
                algorithm.Log(
                    f"[risk] Sector {sector_code} overweight "
                    f"({overweight[sector_code]:.1%}) — scaling "
                    f"{sym} from {target.Quantity:.3f} to {new_qty:.3f}"
                )
                final.append(PortfolioTarget(sym, new_qty))
            else:
                final.append(target)

        return final