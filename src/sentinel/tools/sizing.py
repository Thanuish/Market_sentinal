"""Deterministic position sizing — the execution guardrail.

Pure functions, no LLM anywhere near this module. The language-model side of
the system may only pass (signal_strength, price, portfolio state) in; the
number of shares comes out of hardcoded, unit-tested math.

Sizing = fractional Kelly, clamped by hard risk caps:
  - never risk more than MAX_POSITION_PCT of equity in one position
  - never let cash go below CASH_FLOOR_PCT of equity
  - zero size for signals below the strength floor
"""
from dataclasses import dataclass

KELLY_FRACTION = 0.25        # quarter-Kelly: standard hallucination-of-edge damper
MAX_POSITION_PCT = 0.10      # hard cap: 10% of equity per position
CASH_FLOOR_PCT = 0.20        # keep 20% cash at all times
STRENGTH_FLOOR = 0.55        # ignore weak signals entirely
ASSUMED_PAYOFF = 1.5         # conservative fixed win/loss payoff ratio


@dataclass(frozen=True)
class SizingDecision:
    ticker: str
    qty: float
    notional: float
    capped_by: str           # which rule bound the size ("kelly", "position_cap", "cash_floor", "strength_floor")


def kelly_fraction(win_prob: float, payoff: float = ASSUMED_PAYOFF) -> float:
    """Classic Kelly: f* = p - (1-p)/b, floored at 0."""
    if not 0.0 <= win_prob <= 1.0:
        raise ValueError(f"win_prob out of range: {win_prob}")
    if payoff <= 0:
        raise ValueError(f"payoff must be positive: {payoff}")
    return max(0.0, win_prob - (1.0 - win_prob) / payoff)


def size_position(ticker: str, price: float, signal_strength: float,
                  equity: float, cash: float) -> SizingDecision:
    """Map a validated signal to an exact share quantity. Deterministic."""
    if price <= 0:
        raise ValueError(f"price must be positive: {price}")
    if equity <= 0 or cash < 0:
        raise ValueError("portfolio state invalid")

    if signal_strength < STRENGTH_FLOOR:
        return SizingDecision(ticker, 0.0, 0.0, "strength_floor")

    # Signal strength (0..1) is treated as a win-probability proxy; quarter-Kelly
    # then discounts the model's own confidence.
    f = kelly_fraction(signal_strength) * KELLY_FRACTION
    notional = equity * f
    capped_by = "kelly"

    cap = equity * MAX_POSITION_PCT
    if notional > cap:
        notional, capped_by = cap, "position_cap"

    spendable = cash - equity * CASH_FLOOR_PCT
    if notional > spendable:
        notional, capped_by = max(0.0, spendable), "cash_floor"

    qty = round(notional / price, 4)  # fractional shares, 4dp
    return SizingDecision(ticker, qty, round(qty * price, 2), capped_by)
