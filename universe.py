from AlgorithmImports import *


def coarse_filter(coarse) -> list:
    """
    Broad liquidity and price screen.
    Targets US large/mid cap equities with meaningful volume.
    """
    filtered = [
        x for x in coarse
        if x.HasFundamentalData
        and x.Price > 5                    # exclude penny stocks
        and x.DollarVolume > 10_000_000    # minimum $10m daily dollar volume
    ]
    # Sort by dollar volume and take top 500 — manageable for weekly scoring
    filtered.sort(key=lambda x: x.DollarVolume, reverse=True)
    return [x.Symbol for x in filtered[:500]]


def fine_filter(fine) -> list:
    """
    Fundamental quality screen — excludes financials (different ratio dynamics)
    and requires non-zero fundamentals to be present.
    """
    filtered = [
        x for x in fine
        if x.AssetClassification.MorningstarSectorCode != MorningstarSectorCode.FinancialServices
        and x.ValuationRatios.PERatio > 0
        and x.OperationRatios.ROE != 0
    ]
    return [x.Symbol for x in filtered]