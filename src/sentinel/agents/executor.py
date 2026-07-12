"""Agent C — Executor. The only node allowed to touch the broker, and it
refuses to act without a matching, positive Approval in state. Order size
comes exclusively from the deterministic sizing engine — never from a model.
"""
from sentinel.state import OrderResult, SentinelState
from sentinel.tools.sizing import size_position


class GateViolation(RuntimeError):
    """Raised if the executor is invoked without evaluator approval."""


def make_executor(broker):
    def executor(state: SentinelState) -> SentinelState:
        sig, appr = state.get("current"), state.get("approval")
        log = list(state.get("log", []))
        if not sig:
            return {"log": log + ["executor: no signal"]}
        if appr is None or not appr.get("approved") or appr.get("ticker") != sig["ticker"]:
            raise GateViolation(
                f"executor invoked for {sig['ticker']} without a matching approval")

        marks = {sig["ticker"]: sig["price"]}
        decision = size_position(sig["ticker"], sig["price"], sig["strength"],
                                 equity=broker.equity(marks), cash=broker.cash)
        log.append(f"executor: sizing {sig['ticker']} -> qty {decision.qty} "
                   f"(notional {decision.notional}, capped_by {decision.capped_by})")
        trade = broker.place_order(sig["ticker"], "buy", decision.qty, sig["price"])
        log.append(f"executor: order {trade['status']} — {trade}")
        orders = list(state.get("orders", []))
        orders.append(OrderResult(ticker=sig["ticker"], side="buy",
                                  qty=trade["qty"], fill_price=trade["fill_price"],
                                  status=trade["status"]))
        return {"orders": orders, "log": log}
    return executor
