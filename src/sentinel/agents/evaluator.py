"""Agent B — Evaluator. Retrieval-augmented risk gate.

Retrieves evidence (local corpus + live headlines), scores risk, and writes an
explicit Approval into state. The executor is unreachable in the graph until
this node approves — and every decision cites the evidence it used.
"""
from sentinel.rag.retrieve import (CorpusRetriever, heuristic_risk,
                                   live_headlines, llm_judge)
from sentinel.state import Approval, SentinelState

RISK_CEILING = 0.5   # reject anything riskier than this


def make_evaluator(corpus_dir: str, use_live_news: bool = False):
    retriever = CorpusRetriever(corpus_dir)

    def evaluator(state: SentinelState) -> SentinelState:
        sig = state.get("current")
        log = list(state.get("log", []))
        if not sig:
            return {"approval": None, "log": log + ["evaluator: nothing to evaluate"]}

        evidence: list[str] = []
        for name, score, snippet in retriever.search(sig["ticker"], k=3):
            evidence.append(f"[corpus:{name} score={score:.2f}] {snippet}")
        if use_live_news:
            evidence += [f"[headline] {h}" for h in live_headlines(sig["ticker"])]

        judged = llm_judge(sig["ticker"], evidence)
        risk, detail = judged if judged else heuristic_risk(evidence)
        approved = risk <= RISK_CEILING
        rationale = (f"risk {risk} vs ceiling {RISK_CEILING} — {detail}"
                     + ("" if judged else " (heuristic mode)"))
        log.append(f"evaluator: {sig['ticker']} {'APPROVED' if approved else 'REJECTED'} — {rationale}")
        return {"approval": Approval(ticker=sig["ticker"], approved=approved,
                                     risk_score=risk, evidence=evidence,
                                     rationale=rationale),
                "log": log}
    return evaluator
