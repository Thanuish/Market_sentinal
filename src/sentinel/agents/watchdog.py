"""Agent A — Watchdog. Deterministic technical screen over the ticker list.

Flags a momentum candidate when RSI leaves the neutral band while the MACD
histogram confirms direction. Signal strength is a bounded, explainable
function of the two indicators — not a model vibe.
"""
import pandas as pd

from sentinel.state import SentinelState, Signal
from sentinel.tools.market_data import with_indicators

RSI_HOT = 60.0
RSI_COLD = 40.0


def scan_frame(ticker: str, df: pd.DataFrame) -> Signal | None:
    d = with_indicators(df)
    last = d.iloc[-1]
    r, h = float(last["rsi"]), float(last["macd_hist"])
    price = float(last["Close"])
    if r >= RSI_HOT and h > 0:
        strength = min(1.0, 0.5 + (r - RSI_HOT) / 80 + min(h / price, 0.02) * 10)
        return Signal(ticker=ticker, price=price, rsi=round(r, 2),
                      macd_hist=round(h, 4), strength=round(strength, 3),
                      reason=f"RSI {r:.1f} >= {RSI_HOT} with positive MACD histogram {h:.3f}")
    return None


def make_watchdog(data_source):
    """data_source: callable ticker -> OHLCV DataFrame."""
    def watchdog(state: SentinelState) -> SentinelState:
        signals = []
        log = list(state.get("log", []))
        for t in state["tickers"]:
            try:
                sig = scan_frame(t, data_source(t))
            except Exception as e:
                log.append(f"watchdog: {t} data error: {e}")
                continue
            if sig:
                signals.append(sig)
                log.append(f"watchdog: FLAG {t} — {sig['reason']} (strength {sig['strength']})")
            else:
                log.append(f"watchdog: {t} no signal")
        return {"signals": signals, "current": signals[0] if signals else None,
                "log": log, "approval": None}
    return watchdog
