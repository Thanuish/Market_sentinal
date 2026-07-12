"""Evidence retrieval + risk scoring for the Evaluator.

Retrieval-augmented in the literal sense: the evaluator's decision must cite
retrieved text. Two evidence sources:
  1. a local corpus directory (news clippings / filing notes as .txt files),
     ranked by a hand-rolled TF-IDF cosine (no heavy deps, fully offline)
  2. live headlines via yfinance (best effort; skipped offline)

Scoring modes:
  - heuristic (default, keyless): keyword polarity over retrieved evidence
  - llm (optional): plug an Anthropic/Ollama call into `llm_judge` — the hook
    exists but stays OFF by default so the whole pipeline runs without keys.

Planned upgrade: embedding store (Chroma) over real SEC filings.
"""
import math
import re
from collections import Counter
from pathlib import Path

NEGATIVE = {"fraud", "lawsuit", "investigation", "sec", "probe", "resign",
            "resignation", "bankruptcy", "default", "recall", "downgrade",
            "restatement", "layoffs", "halt", "delisting", "scandal", "fine",
            "penalty", "warning", "miss", "misses", "plunge", "short-seller"}
POSITIVE = {"beat", "beats", "record", "upgrade", "raised", "guidance",
            "breakthrough", "approval", "partnership", "contract", "buyback",
            "dividend", "expansion", "profit", "growth"}

_token_re = re.compile(r"[a-z][a-z\-']+")


def _tokens(text: str) -> list[str]:
    return _token_re.findall(text.lower())


class CorpusRetriever:
    """TF-IDF cosine over .txt documents in a directory. Small and honest."""

    def __init__(self, corpus_dir: str | Path):
        self.docs: dict[str, list[str]] = {}
        p = Path(corpus_dir)
        if p.is_dir():
            for f in sorted(p.glob("*.txt")):
                self.docs[f.name] = _tokens(f.read_text(encoding="utf-8", errors="ignore"))
        self.df = Counter()
        for toks in self.docs.values():
            self.df.update(set(toks))
        self.n = max(1, len(self.docs))

    def _vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: (c / len(toks)) * math.log(1 + self.n / (1 + self.df.get(t, 0)))
                for t, c in tf.items()} if toks else {}

    def search(self, query: str, k: int = 3) -> list[tuple[str, float, str]]:
        qv = self._vec(_tokens(query))
        if not qv or not self.docs:
            return []
        scored = []
        for name, toks in self.docs.items():
            dv = self._vec(toks)
            dot = sum(qv[t] * dv.get(t, 0.0) for t in qv)
            norm = math.sqrt(sum(v * v for v in qv.values())) * \
                math.sqrt(sum(v * v for v in dv.values()) or 1)
            if dot > 0:
                snippet = " ".join(toks[:60])
                scored.append((name, dot / norm, snippet))
        return sorted(scored, key=lambda x: -x[1])[:k]


def live_headlines(ticker: str, limit: int = 8) -> list[str]:
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        titles = []
        for item in news[:limit]:
            content = item.get("content") or item
            title = content.get("title") or ""
            if title:
                titles.append(title)
        return titles
    except Exception:
        return []


def heuristic_risk(evidence: list[str]) -> tuple[float, str]:
    """Keyword-polarity risk score over evidence. 0 safe .. 1 toxic."""
    toks = _tokens(" ".join(evidence))
    neg = sum(t in NEGATIVE for t in toks)
    pos = sum(t in POSITIVE for t in toks)
    if neg == 0 and pos == 0:
        return 0.35, "no red flags found in evidence, but coverage is thin"
    risk = neg / (neg + pos + 1)
    detail = f"{neg} negative / {pos} positive markers in {len(evidence)} snippets"
    return round(min(1.0, risk), 3), detail


def llm_judge(ticker: str, evidence: list[str]) -> tuple[float, str] | None:
    """Optional LLM risk judgement. Wire an Anthropic/Ollama call here and
    return (risk_score, rationale). Returns None when unconfigured, and the
    evaluator falls back to the heuristic — the pipeline never depends on a key."""
    return None
