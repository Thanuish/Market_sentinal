"""LangGraph wiring: watchdog -> (signal?) -> evaluator -> (approved?) -> executor.

The gates are graph edges, not conventions: there is no path to the executor
that does not pass through the evaluator's conditional edge, and the executor
additionally re-checks the approval in state (defense in depth, see
tests/test_gate.py).
"""
from langgraph.graph import END, StateGraph

from sentinel.agents.evaluator import make_evaluator
from sentinel.agents.executor import make_executor
from sentinel.agents.watchdog import make_watchdog
from sentinel.state import SentinelState


def has_signal(state: SentinelState) -> str:
    return "evaluate" if state.get("current") else "end"


def is_approved(state: SentinelState) -> str:
    appr = state.get("approval")
    return "execute" if appr and appr.get("approved") else "end"


def build_graph(data_source, broker, corpus_dir: str, use_live_news: bool = False):
    g = StateGraph(SentinelState)
    g.add_node("watchdog", make_watchdog(data_source))
    g.add_node("evaluator", make_evaluator(corpus_dir, use_live_news))
    g.add_node("executor", make_executor(broker))

    g.set_entry_point("watchdog")
    g.add_conditional_edges("watchdog", has_signal,
                            {"evaluate": "evaluator", "end": END})
    g.add_conditional_edges("evaluator", is_approved,
                            {"execute": "executor", "end": END})
    g.add_edge("executor", END)
    return g.compile()
