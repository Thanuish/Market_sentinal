"""Evaluation metrics that actually matter for a trading study."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe(returns: pd.Series, risk_free_daily: float = 0.0) -> float:
    ex = returns - risk_free_daily
    sd = ex.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(ex.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def win_rate(trade_pnls: list[float]) -> float:
    closed = [p for p in trade_pnls if p != 0]
    if not closed:
        return 0.0
    return sum(p > 0 for p in closed) / len(closed)


def summarize(equity_curve: pd.Series, trade_pnls: list[float],
              benchmark_curve: pd.Series) -> dict:
    rets = equity_curve.pct_change().dropna()
    bench_rets = benchmark_curve.pct_change().dropna()
    return {
        "total_return_pct": round(100 * (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1), 2),
        "benchmark_return_pct": round(100 * (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0] - 1), 2),
        "sharpe": round(sharpe(rets), 3),
        "benchmark_sharpe": round(sharpe(bench_rets), 3),
        "max_drawdown_pct": round(100 * max_drawdown(equity_curve), 2),
        "win_rate_pct": round(100 * win_rate(trade_pnls), 1),
        "n_trades": len(trade_pnls),
    }
