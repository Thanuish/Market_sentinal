"""Brokers. PaperBroker is a local simulated portfolio (the default and the
only one implemented). AlpacaAdapter is a stub for the planned paper-trading
integration — it raises loudly rather than pretending.
"""
import json
import time
from pathlib import Path


class PaperBroker:
    """Simulated portfolio with a JSON ledger. No real money, ever."""

    def __init__(self, starting_cash: float = 100_000.0,
                 ledger_path: str | Path | None = None):
        self.cash = starting_cash
        self.positions: dict[str, dict] = {}   # ticker -> {qty, avg_price}
        self.trades: list[dict] = []
        self.ledger_path = Path(ledger_path) if ledger_path else None

    def equity(self, marks: dict[str, float] | None = None) -> float:
        marks = marks or {}
        pos_val = sum(p["qty"] * marks.get(t, p["avg_price"])
                      for t, p in self.positions.items())
        return self.cash + pos_val

    def place_order(self, ticker: str, side: str, qty: float, price: float) -> dict:
        if qty <= 0:
            return {"ticker": ticker, "side": side, "qty": 0.0,
                    "fill_price": price, "status": "rejected_zero_qty"}
        cost = qty * price
        if side == "buy":
            if cost > self.cash:
                return {"ticker": ticker, "side": side, "qty": qty,
                        "fill_price": price, "status": "rejected_insufficient_cash"}
            self.cash -= cost
            pos = self.positions.setdefault(ticker, {"qty": 0.0, "avg_price": 0.0})
            total = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + cost) / total
            pos["qty"] = total
        elif side == "sell":
            pos = self.positions.get(ticker)
            if not pos or pos["qty"] < qty:
                return {"ticker": ticker, "side": side, "qty": qty,
                        "fill_price": price, "status": "rejected_no_position"}
            pos["qty"] -= qty
            self.cash += cost
            if pos["qty"] <= 1e-9:
                del self.positions[ticker]
        else:
            raise ValueError(f"unknown side: {side}")

        trade = {"ts": time.time(), "ticker": ticker, "side": side,
                 "qty": qty, "fill_price": price, "status": "filled"}
        self.trades.append(trade)
        self._persist()
        return trade

    def _persist(self):
        if self.ledger_path:
            self.ledger_path.write_text(json.dumps(
                {"cash": self.cash, "positions": self.positions,
                 "trades": self.trades}, indent=2), encoding="utf-8")


class AlpacaAdapter:
    """Planned: Alpaca paper-trading API adapter. Not implemented yet —
    fails loudly so nothing can silently pretend to trade."""

    def __init__(self, *_, **__):
        raise NotImplementedError(
            "AlpacaAdapter is planned but not implemented. Use PaperBroker.")
