"""Prove the state-machine gate: the executor is unreachable without approval,
and raises if invoked directly with a missing/negative/mismatched approval."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentinel.agents.executor import GateViolation, make_executor
from sentinel.graph import build_graph
from sentinel.state import Approval, Signal
from sentinel.tools.broker import PaperBroker
from sentinel.tools.market_data import synthetic_daily

SIG = Signal(ticker="TEST", price=100.0, rsi=70.0, macd_hist=0.5,
             reason="test", strength=0.9)


def test_executor_refuses_without_approval():
    ex = make_executor(PaperBroker())
    with pytest.raises(GateViolation):
        ex({"current": SIG, "approval": None, "log": [], "orders": []})


def test_executor_refuses_rejected_approval():
    ex = make_executor(PaperBroker())
    appr = Approval(ticker="TEST", approved=False, risk_score=0.9,
                    evidence=[], rationale="too risky")
    with pytest.raises(GateViolation):
        ex({"current": SIG, "approval": appr, "log": [], "orders": []})


def test_executor_refuses_mismatched_ticker():
    ex = make_executor(PaperBroker())
    appr = Approval(ticker="OTHER", approved=True, risk_score=0.1,
                    evidence=[], rationale="approved for a different ticker")
    with pytest.raises(GateViolation):
        ex({"current": SIG, "approval": appr, "log": [], "orders": []})


def test_executor_acts_with_valid_approval():
    broker = PaperBroker(starting_cash=100_000)
    ex = make_executor(broker)
    appr = Approval(ticker="TEST", approved=True, risk_score=0.1,
                    evidence=["[corpus] fine"], rationale="ok")
    out = ex({"current": SIG, "approval": appr, "log": [], "orders": []})
    assert out["orders"] and out["orders"][0]["status"] == "filled"
    assert broker.positions.get("TEST")


def test_full_graph_never_trades_toxic_ticker(tmp_path):
    """End to end: corpus containing fraud evidence must block execution."""
    import numpy as np

    (tmp_path / "TEST_news.txt").write_text(
        "TEST corporation faces fraud lawsuit and SEC investigation after "
        "CEO resignation; bankruptcy warning issued", encoding="utf-8")
    # force a hot signal: accelerating uptrend keeps RSI high AND the MACD
    # histogram positive (constant-rate growth would let it converge to zero)
    df = synthetic_daily("TEST", days=120, seed=1)
    rates = np.linspace(0.001, 0.03, len(df))
    df["Close"] = 100.0 * np.cumprod(1 + rates)
    broker = PaperBroker()
    graph = build_graph(lambda t: df, broker, str(tmp_path))
    final = graph.invoke({"tickers": ["TEST"], "log": [], "orders": []})
    assert final.get("approval") and final["approval"]["approved"] is False
    assert not final.get("orders")
    assert broker.positions == {}
