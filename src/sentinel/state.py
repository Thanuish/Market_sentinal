"""Shared state flowing through the LangGraph graph.

The state IS the security model: the executor node reads `approval` and refuses
to act unless the evaluator wrote an explicit, per-ticker approval. No node can
skip a stage because the graph edges only route forward through the gates.
"""
from typing import Optional, TypedDict


class Signal(TypedDict):
    ticker: str
    price: float
    rsi: float
    macd_hist: float
    reason: str          # human-readable technical justification
    strength: float      # 0..1, from technicals only (deterministic)


class Approval(TypedDict):
    ticker: str
    approved: bool
    risk_score: float    # 0 (safe) .. 1 (toxic)
    evidence: list[str]  # retrieved snippets the decision cites
    rationale: str


class OrderResult(TypedDict):
    ticker: str
    side: str
    qty: float
    fill_price: float
    status: str


class SentinelState(TypedDict, total=False):
    tickers: list[str]
    signals: list[Signal]            # written by watchdog
    current: Optional[Signal]        # signal under evaluation
    approval: Optional[Approval]     # written ONLY by evaluator
    orders: list[OrderResult]        # written ONLY by executor
    log: list[str]
