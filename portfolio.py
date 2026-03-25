import pandas as pd
from AlgorithmImports import *


def build_targets(scored: pd.DataFrame,
                  long_q: int = 5,
                  short_q: int = 1,
                  max_position: float = 0.05) -> dict:
    """
    Converts quintile scores into portfolio weight targets.
    Returns {symbol: target_weight} — positive = long, negative = short.
    max_position: maximum weight per stock (default 5% = 20 stock minimum).
    """
    if scored.empty or "quintile" not in scored.columns:
        return {}

    long_leg  = scored[scored["quintile"].astype(int) == long_q]
    short_leg = scored[scored["quintile"].astype(int) == short_q]

    targets = {}

    if not long_leg.empty:
        long_weight = min(0.50 / len(long_leg), max_position)
        for symbol in long_leg.index:
            targets[symbol] = long_weight

    if not short_leg.empty:
        short_weight = min(0.50 / len(short_leg), max_position)
        for symbol in short_leg.index:
            targets[symbol] = -short_weight

    return targets


def get_exit_targets(current_holdings: list,
                     new_targets: dict) -> dict:
    """
    Returns zero-weight targets for positions no longer in the signal.
    Ensures clean exits when a stock rotates out of Q5 or Q1.
    """
    exits = {}
    for symbol in current_holdings:
        if symbol not in new_targets:
            exits[symbol] = 0.0
    return exits