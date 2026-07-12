"""Walk-forward backtest: run the full agent graph day by day over historical
bars (synthetic by default; --live-data for yfinance) and compare the resulting
equity curve against buy-and-hold of a benchmark.

Usage:
    python backtest/run_backtest.py --days 180 [--live-data] [--tickers AAPL MSFT NVDA]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest_support import PROJECT_ROOT  # noqa: E402  (see below)
from sentinel.graph import build_graph  # noqa: E402
from sentinel.tools.broker import PaperBroker  # noqa: E402
from sentinel.tools.market_data import fetch_daily, synthetic_daily  # noqa: E402
from metrics import summarize  # noqa: E402

WARMUP = 40  # bars needed before indicators are meaningful


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--tickers", nargs="+",
                    default=["AAPL", "MSFT", "NVDA", "TSLA", "AMD"])
    ap.add_argument("--benchmark", default="SPY")
    ap.add_argument("--live-data", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    total = args.days + WARMUP
    if args.live_data:
        frames = {t: fetch_daily(t, period="2y").tail(total) for t in args.tickers}
        bench = fetch_daily(args.benchmark, period="2y")["Close"].tail(args.days)
    else:
        frames = {t: synthetic_daily(t, days=total, seed=args.seed + i)
                  for i, t in enumerate(args.tickers)}
        bench = synthetic_daily(args.benchmark, days=args.days, seed=args.seed - 1)["Close"]

    broker = PaperBroker(starting_cash=100_000.0)
    corpus = str(PROJECT_ROOT / "corpus")
    equity_curve, entry_prices, trade_pnls = [], {}, []

    for step in range(WARMUP, total):
        window = {t: df.iloc[: step + 1] for t, df in frames.items()}
        graph = build_graph(lambda t, w=window: w[t], broker, corpus)
        final = graph.invoke({"tickers": args.tickers, "log": [], "orders": []})
        for order in final.get("orders", []):
            if order["status"] == "filled":
                entry_prices[order["ticker"]] = order["fill_price"]

        marks = {t: float(df["Close"].iloc[step]) for t, df in frames.items()}
        # naive exit rule for the study: close positions up/down 5% vs entry
        for t in list(broker.positions):
            entry = entry_prices.get(t)
            if entry and abs(marks[t] / entry - 1) >= 0.05:
                qty = broker.positions[t]["qty"]
                broker.place_order(t, "sell", qty, marks[t])
                trade_pnls.append((marks[t] - entry) * qty)
        equity_curve.append(broker.equity(marks))

    curve = pd.Series(equity_curve)
    bench_curve = bench.tail(len(curve)).reset_index(drop=True)
    result = summarize(curve, trade_pnls, bench_curve)
    print(f"\n=== Market Sentinel backtest ({'live' if args.live_data else 'synthetic'} data, "
          f"{args.days} days, {len(args.tickers)} tickers) ===")
    for k, v in result.items():
        print(f"  {k:>24}: {v}")
    print("\nNOTE: research harness on paper trades — not investment advice, "
          "and synthetic-data numbers say nothing about markets.")


if __name__ == "__main__":
    main()
