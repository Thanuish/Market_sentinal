import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.tools.sizing import (CASH_FLOOR_PCT, MAX_POSITION_PCT,
                                   kelly_fraction, size_position)


def test_kelly_basics():
    assert kelly_fraction(0.5, payoff=1.0) == 0.0          # no edge, no bet
    assert kelly_fraction(1.0) == 1.0                       # certainty -> full
    assert kelly_fraction(0.0) == 0.0                       # floored at zero
    assert 0 < kelly_fraction(0.6) < 1


def test_kelly_rejects_garbage():
    with pytest.raises(ValueError):
        kelly_fraction(1.5)
    with pytest.raises(ValueError):
        kelly_fraction(0.6, payoff=0)


def test_weak_signals_get_zero():
    d = size_position("AAPL", 100, 0.4, equity=100_000, cash=100_000)
    assert d.qty == 0 and d.capped_by == "strength_floor"


def test_position_cap_binds():
    d = size_position("AAPL", 100, 0.99, equity=100_000, cash=100_000)
    assert d.notional <= 100_000 * MAX_POSITION_PCT + 1e-6
    assert d.capped_by in ("position_cap", "kelly")


def test_cash_floor_binds():
    # equity high but almost no spendable cash above the floor
    d = size_position("AAPL", 100, 0.9, equity=100_000,
                      cash=100_000 * CASH_FLOOR_PCT + 500)
    assert d.notional <= 500 + 1e-6
    assert d.capped_by == "cash_floor"


def test_never_negative():
    d = size_position("AAPL", 100, 0.9, equity=100_000,
                      cash=100_000 * CASH_FLOOR_PCT - 1000)
    assert d.qty == 0.0


def test_deterministic():
    a = size_position("AAPL", 123.45, 0.8, 50_000, 30_000)
    b = size_position("AAPL", 123.45, 0.8, 50_000, 30_000)
    assert a == b
