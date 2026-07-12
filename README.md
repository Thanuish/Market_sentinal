# Market Sentinel — Multi-Agent Paper-Trading Research System

A LangGraph-orchestrated three-agent pipeline that watches market data, validates
signals against retrieved evidence (news/filings), and executes **simulated** trades
through a deterministic guardrail layer. Built as an applied-AI engineering study:
the interesting problems here are hallucination containment, deterministic execution
boundaries, and honest evaluation — not stock picking.

> **Paper trading only.** This system never touches real money. It is a research
> and portfolio project, not investment advice or an investment product.

## Architecture

```
[ Market Data (yfinance / synthetic) ]
        │
        ▼
( Agent A: Watchdog )        deterministic technicals: RSI, MACD cross
        │  flags momentum candidates into LangGraph state
        ▼
( Agent B: Evaluator )       retrieval-augmented risk check: pulls evidence
        │  from a local corpus (news/filings notes) + headline feed,
        │  scores sentiment/risk, must APPROVE before the graph
        │  ever routes toward the executor
        ▼
( Agent C: Executor )        NO LLM MATH: position size comes from a pure-Python
        │                    fractional-Kelly + hard-cap engine; also exposed
        ▼                    as an MCP server for external agent clients
[ PaperBroker ]              local simulated portfolio (JSON ledger)
[ Backtest Harness ]         Sharpe, max drawdown, win rate vs buy-and-hold benchmark
```

### The two design rules that matter

1. **Deterministic execution guardrail.** Language models never compute order
   sizes. The executor passes `(ticker, price, signal strength, portfolio state)`
   to `sentinel/tools/sizing.py` — pure, unit-tested Python (fractional Kelly with
   hard risk caps). The same engine is exposed via `mcp_server/sizing_server.py`
   (Model Context Protocol), so any MCP client gets the math without the model
   being able to alter it.
2. **State-machine gating.** The LangGraph graph makes the `place_order` path
   structurally unreachable until the Evaluator has written an explicit approval
   into state. `tests/test_gate.py` proves the executor refuses unapproved signals.

## Status (honest ledger)

| Component | State |
|---|---|
| LangGraph 3-agent state machine with approval gating | ✅ working |
| Deterministic sizing engine (fractional Kelly + caps) + unit tests | ✅ working |
| Watchdog technicals (RSI, MACD) on yfinance daily bars | ✅ working |
| Synthetic offline data generator (reproducible dev/backtests) | ✅ working |
| Evaluator: local-corpus retrieval + keyword sentiment scoring | ✅ working (heuristic mode) |
| Evaluator: LLM judgement (Claude / Ollama pluggable) | 🔌 hook present, off by default |
| PaperBroker simulated portfolio | ✅ working |
| MCP server exposing the sizing engine | ✅ working |
| Backtest harness: Sharpe / max drawdown / win rate vs benchmark | ✅ working |
| Alpaca paper-trading adapter | 🚧 stub — planned |
| Embedding-based RAG store (Chroma) over real SEC filings | 🚧 planned |
| Intraday data + Alpha Vantage indicators | 🚧 planned |

## Quickstart

```bash
pip install -r requirements.txt
pytest                     # sizing math + gate enforcement
python main.py             # demo run (synthetic data, no keys needed)
python main.py --live-data # same pipeline on real yfinance daily bars
python backtest/run_backtest.py --days 180
```

## Repo map

```
src/sentinel/state.py            shared LangGraph state schema
src/sentinel/tools/market_data.py  OHLCV + RSI/MACD (pure pandas)
src/sentinel/tools/sizing.py       THE guardrail: pure sizing math
src/sentinel/tools/broker.py       PaperBroker + Alpaca stub
src/sentinel/rag/retrieve.py       corpus retrieval + sentiment scoring
src/sentinel/agents/{watchdog,evaluator,executor}.py
src/sentinel/graph.py              LangGraph wiring + gates
src/mcp_server/sizing_server.py    MCP interface to the sizing engine
backtest/                          metrics + walk-forward runner
tests/                             pytest suite
```
