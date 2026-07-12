"""Market data + deterministic technical indicators.

Two sources behind one interface:
  - fetch_daily(ticker): real daily bars via yfinance (needs internet)
  - synthetic_daily(ticker, days, seed): reproducible random-walk bars for
    offline dev, tests, and deterministic backtests

Indicators are computed here in pure pandas — never asked of a language model.
"""
import numpy as np
import pandas as pd


def synthetic_daily(ticker: str, days: int = 250, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed if seed is not None else abs(hash(ticker)) % 2**32)
    drift, vol = 0.0003, 0.018
    rets = rng.normal(drift, vol, days)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, days))
    low = close * (1 - rng.uniform(0, 0.01, days))
    open_ = np.concatenate([[100.0], close[:-1]])
    volume = rng.integers(1e5, 5e6, days).astype(float)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                         "Close": close, "Volume": volume}, index=idx)


def fetch_daily(ticker: str, period: str = "1y") -> pd.DataFrame:
    import yfinance as yf
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(loss > 0, 100.0)                  # pure gains -> RSI 100
    out = out.where((gain > 0) | (loss > 0), 50.0)    # flat series -> neutral
    return out.fillna(50.0)


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    return macd - macd.ewm(span=signal, adjust=False).mean()


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["Close"])
    out["macd_hist"] = macd_histogram(out["Close"])
    return out
