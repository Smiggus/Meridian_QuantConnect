"""
LocalFundamentalAdapter
────────────────────────
Wraps FMP fundamental data (as written by the Dagster pipeline's
LeanDataWriter) to present the same attribute interface as Morningstar
FineFundamental objects. This lets factors.extract_fundamental() work
unchanged in local backtests powered by Databento + FMP data.

Morningstar attribute paths consumed by factors.py FACTOR_MAP:

    f.ValuationRatios.PERatio              → _Scalar (float-like)
    f.ValuationRatios.EVToEBITDA           → _Scalar
    f.ValuationRatios.PBRatio              → _Scalar
    f.ValuationRatios.FCFYield             → _Scalar
    f.OperationRatios.ROE.OneYear          → _OneYear → _Scalar
    f.OperationRatios.ROIC.OneYear         → _OneYear → _Scalar
    f.OperationRatios.GrossMargin.OneYear  → _OneYear → _Scalar
    f.OperationRatios.LongTermDebtEquityRatio.OneYear
    f.OperationRatios.FCFNetIncomeRatio.OneYear
    f.OperationRatios.RevenueGrowth.OneYear
    f.OperationRatios.NetIncomeGrowth.OneYear
    f.OperationRatios.FCFGrowth.OneYear
    f.OperationRatios.OperationIncomeGrowth.OneYear

FMP → LEAN fine JSON field mapping (from LeanDataWriter._build_fine_json):

    FinancialStatements.IncomeStatement  → TotalRevenue, GrossProfit, Ebitda,
                                           NetIncome, OperatingIncome
    FinancialStatements.BalanceSheet     → TotalAssets, CommonStockEquity,
                                           TotalDebt
    FinancialStatements.CashFlowStatement → FreeCashFlow, OperatingCashFlow
    ValuationRatios                      → PERatio, PBRatio, EVToEBITDA,
                                           ReturnOnEquity, ReturnOnAssets,
                                           DebtToEquityRatio
"""

from __future__ import annotations
import math


# ── Leaf wrappers ─────────────────────────────────────────────────────────────

class _Scalar:
    """
    Float-like wrapper. float(obj) returns the stored value.
    NaN is used for missing / zero / non-finite inputs.
    """
    def __init__(self, value):
        try:
            v = float(value)
            self._v = v if math.isfinite(v) else float("nan")
        except (TypeError, ValueError):
            self._v = float("nan")

    def __float__(self):
        return self._v

    def __repr__(self):
        return f"_Scalar({self._v:.4g})"


class _OneYear:
    """
    Mimics Morningstar MultiPeriodField .OneYear accessor pattern.
    Usage: float(obj.OneYear)
    """
    __slots__ = ("OneYear",)

    def __init__(self, value):
        self.OneYear = _Scalar(value)


# ── Computed ratio helpers ────────────────────────────────────────────────────

def _div(numerator, denominator) -> float | None:
    """Safe division. Returns None if either input is missing or zero."""
    try:
        if numerator is not None and denominator and denominator != 0:
            result = float(numerator) / float(denominator)
            return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        pass
    return None


def _yoy(current, previous) -> float | None:
    """
    Year-over-year growth rate: (curr - prev) / abs(prev).
    Returns None if either period is missing, zero, or the result is non-finite.
    """
    try:
        if current is not None and previous and previous != 0:
            result = (float(current) - float(previous)) / abs(float(previous))
            return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        pass
    return None


# ── Attribute groups ─────────────────────────────────────────────────────────

class _ValuationRatios:
    """
    Exposes ValuationRatios scalars.
    FCFYield uses FCF / TotalAssets as a proxy (market cap unavailable locally).
    """
    def __init__(self, curr: dict):
        self.PERatio    = _Scalar(curr.get("pe_ratio"))
        self.EVToEBITDA = _Scalar(curr.get("ev_ebitda"))
        self.PBRatio    = _Scalar(curr.get("pb_ratio"))
        self.FCFYield   = _Scalar(_div(curr.get("free_cash_flow"), curr.get("total_assets")))


class _OperationRatios:
    """
    Exposes OperationRatios with .OneYear accessor on each field.
    Ratios are computed from raw FMP statement data where not directly available.
    Growth factors require a previous-period record for YoY calculation.
    """
    def __init__(self, curr: dict, prev: dict):
        equity    = curr.get("equity") or 0
        debt      = curr.get("total_debt") or 0
        op_income = curr.get("operating_income")
        revenue   = curr.get("revenue") or 0
        gross     = curr.get("gross_profit")
        net_income = curr.get("net_income") or 0
        fcf       = curr.get("free_cash_flow")
        invested_capital = equity + debt

        # Direct / single-period ratios
        self.ROE                  = _OneYear(curr.get("roe"))
        self.ROIC                 = _OneYear(_div(op_income, invested_capital))
        self.GrossMargin          = _OneYear(_div(gross, revenue))
        self.LongTermDebtEquityRatio = _OneYear(_div(debt, equity))
        self.FCFNetIncomeRatio    = _OneYear(_div(fcf, net_income))

        # YoY growth rates — require two periods; NaN when previous period absent
        self.RevenueGrowth         = _OneYear(_yoy(curr.get("revenue"),        prev.get("revenue")))
        self.NetIncomeGrowth       = _OneYear(_yoy(curr.get("net_income"),     prev.get("net_income")))
        self.FCFGrowth             = _OneYear(_yoy(curr.get("free_cash_flow"), prev.get("free_cash_flow")))
        self.OperationIncomeGrowth = _OneYear(_yoy(curr.get("ebitda"),         prev.get("ebitda")))


# ── Public adapter ────────────────────────────────────────────────────────────

class LocalFundamentalAdapter:
    """
    Wraps one or two FMP fundamental data dicts to mimic Morningstar
    FineFundamental objects for use in factors.extract_fundamental().

    Args:
        records: list of dicts ordered newest-first. Each dict holds fields
                 as parsed from the LEAN fine JSON written by LeanDataWriter.
                 At least one record is required; two are needed for growth
                 factor computation (RevenueGrowth, etc.).
    """

    __slots__ = ("ValuationRatios", "OperationRatios")

    def __init__(self, records: list[dict]):
        curr = records[0] if records else {}
        prev = records[1] if len(records) > 1 else {}
        self.ValuationRatios = _ValuationRatios(curr)
        self.OperationRatios = _OperationRatios(curr, prev)
