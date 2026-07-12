"""Market Sentinel demo — one full pass of the three-agent graph.

    python main.py               # synthetic data, fully offline, no keys
    python main.py --live-data   # real yfinance daily bars + live headlines
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sentinel.graph import build_graph
from sentinel.tools.broker import PaperBroker
from sentinel.tools.market_data import fetch_daily, synthetic_daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+",
                    default=["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "INTC"])
    ap.add_argument("--live-data", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.live_data:
        def source(t):
            return fetch_daily(t, period="6mo")
    else:
        cache = {t: synthetic_daily(t, days=160, seed=args.seed + i)
                 for i, t in enumerate(args.tickers)}
        def source(t):
            return cache[t]

    broker = PaperBroker(starting_cash=100_000.0,
                         ledger_path=Path("paper_ledger.json"))
    corpus = str(Path(__file__).parent / "corpus")
    graph = build_graph(source, broker, corpus, use_live_news=args.live_data)

    print(f"Market Sentinel — {'LIVE yfinance' if args.live_data else 'synthetic'} data, "
          f"paper portfolio $100,000\n")
    final = graph.invoke({"tickers": args.tickers, "log": [], "orders": []})

    for line in final["log"]:
        print("  " + line)
    print(f"\nsignals flagged : {len(final.get('signals') or [])}")
    appr = final.get("approval")
    if appr:
        print(f"evaluator       : {appr['ticker']} "
              f"{'APPROVED' if appr['approved'] else 'REJECTED'} "
              f"(risk {appr['risk_score']})")
    print(f"orders          : {final.get('orders') or 'none'}")
    print(f"cash remaining  : ${broker.cash:,.2f}")
    print(f"positions       : {broker.positions or '{}'}")
    print("\nPaper trading only — this is an engineering study, not investment advice.")


if __name__ == "__main__":
    main()
