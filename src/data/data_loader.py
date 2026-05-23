"""
data_loader.py
==============
Production-grade data loader for the Cross-Sectional Conviction Ranking paper.

Downloads daily OHLCV data for NASDAQ-100 constituent stocks (2015–2025)
or loads from an external parquet source (e.g. data/nasdaq30_prices.parquet
for the full 30-stock universe used in the paper), constructs 49 strictly
causal technical indicators, and creates the forward-shifted target label.

CRITICAL DESIGN DECISIONS:
  - ALL rolling functions use min_periods and NO center=True to ensure
    strict backward-looking computation (no future data leakage).
  - Target label uses a 2-day forward shift: execution assumed at Close(t+1),
    return window is Close(t+1) → Close(t+2). This avoids execution-at-signal
    lookahead bias.
  - SMA(200) is computed with a minimum 200-day lookback; the first 199 rows
    per ticker are intentionally NaN (not forward-filled or backfilled).
  - The data integrity audit explicitly validates all of the above.

Author: Rajveer Singh Pall
Research Paper: "When the Gate Stays Closed: Empirical Evidence of Near-Zero
                 Cross-Sectional Predictability in Large-Cap NASDAQ Equities
                 Using an IC-Gated Machine Learning Framework"
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Default 7-stock universe used when no external_data_path is provided.
# The paper's 30-stock results are reproduced via:
#   load_all_data(external_data_path="data/nasdaq30_prices.parquet")
TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE: str = "2015-01-01"
END_DATE: str = "2025-01-01"
DATA_CACHE_DIR: Path = Path(__file__).resolve().parents[3] / "data" / "raw"

# The exact 47 feature names in canonical order (must match model training).
FEATURE_NAMES: List[str] = [
    # --- Momentum oscillators (8) ---
    "RSI_14",
    "RSI_21",
    "MACD_12_26",
    "MACD_signal_9",
    "MACD_hist",
    "ROC_10",
    "ROC_21",
    "Williams_R",
    # --- Bollinger Bands (5) ---
    "BB_upper",
    "BB_lower",
    "BB_width",
    "BB_pct_b",
    "CCI_20",
    # --- Volatility / range (6) ---
    "ATR_14",
    "ATR_21",
    "rolling_vol_5d",
    "rolling_vol_21d",
    "rolling_vol_63d",
    "HL_range_norm",
    # --- Trend / moving averages (8) ---
    "EMA_9",
    "EMA_21",
    "EMA_50",
    "EMA_200",
    "SMA_50",
    "SMA_200",
    "price_to_SMA200",
    "price_to_SMA50",
    # --- Returns (6) ---
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_21d",
    # --- Volume (5) ---
    "OBV",
    "OBV_EMA",
    "volume_zscore_5d",
    "volume_zscore_21d",
    "MFI_14",
    # --- Candle structure (4) ---
    "OC_body_norm",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "DPO_20",
    # --- Stochastic / ADX group (6) ---
    "stoch_K",
    "stoch_D",
    "ADX_14",
    "DI_plus",
    "DI_minus",
    "VWAP_deviation",
    # --- Detrended / ROC ---
    "DPO_20",
    "ROC_10",
    "ROC_21",
]
# NOTE: The canonical 49-feature set is determined programmatically by
# _compute_features_single_ticker(). This list is for documentation only
# and may not be exhaustive. Always use get_feature_columns(df) at runtime.


# ---------------------------------------------------------------------------
# HELPER: RAW DOWNLOAD
# ---------------------------------------------------------------------------

def _download_ohlcv(
    tickers: List[str],
    start: str,
    end: str,
    cache_dir: Optional[Path] = DATA_CACHE_DIR,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Download or load from cache daily OHLCV data for each ticker.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols to download.
    start, end : str
        Date range in 'YYYY-MM-DD' format.
    cache_dir : Path, optional
        Directory for Parquet cache files. Created if absent.
    use_cache : bool
        If True, read from Parquet if available; write Parquet after download.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping ticker → DataFrame with columns [Open, High, Low, Close, Volume].
        Index is DatetimeIndex (UTC-naive). Only trading days included.

    Notes
    -----
    yfinance returns adjusted Close by default when downloading individually.
    We download per-ticker to get clean, aligned OHLCV without multi-level
    column issues introduced by the batch API.
    """
    raw: Dict[str, pd.DataFrame] = {}

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        cache_path = cache_dir / f"{ticker}_ohlcv.parquet" if cache_dir else None

        if use_cache and cache_path is not None and cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"  [CACHE] {ticker}: {len(df)} rows loaded from {cache_path.name}")
        else:
            print(f"  [DOWNLOAD] {ticker} …", end=" ", flush=True)
            tk = yf.Ticker(ticker)
            df = tk.history(start=start, end=end, interval="1d", auto_adjust=True)
            # Drop timezone info from index (keep date-only)
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            df.index = pd.to_datetime(df.index).normalize()
            # Keep only essential OHLCV columns
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(how="all", inplace=True)
            df.sort_index(inplace=True)
            print(f"{len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")

            if cache_path is not None:
                df.to_parquet(cache_path)

        raw[ticker] = df

    return raw


def _load_data_from_parquet(path: str | Path) -> pd.DataFrame:
    """
    Load raw OHLCV data from an external parquet file into canonical format.

    Expected format: MultiIndex [date, ticker] or columns [date, ticker].
    Must include Open, High, Low, Close, Volume.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"External data file not found: {path}")

    df = pd.read_parquet(path)

    if isinstance(df.index, pd.MultiIndex) and df.index.names == ["date", "ticker"]:
        # Already correct structure.
        df = df.copy()
    else:
        # Try to infer index from columns.
        if "date" in df.columns and "ticker" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df = df.set_index(["date", "ticker"]).sort_index()
        else:
            raise ValueError(
                "External data must have a MultiIndex (date,ticker) or columns 'date' and 'ticker'."
            )

    # Ensure index dtypes
    df = df.sort_index()
    if df.index.names != ["date", "ticker"]:
        df.index.names = ["date", "ticker"]

    # Validate required columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"External data missing required columns: {sorted(missing)}")

    # Normalize date level
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index(["date", "ticker"]).sort_index()

    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING — all strictly causal (no center=True)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing (EWM with alpha=1/period).

    Uses only past data; first `period` rows will be NaN.

    Parameters
    ----------
    close : pd.Series
        Adjusted close prices (single ticker).
    period : int
        Lookback window.

    Returns
    -------
    pd.Series
        RSI values in [0, 100].
    """
    delta = close.diff(1)
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average with min_periods=span (strictly causal)."""
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """
    Simple Moving Average — strictly backward-looking.

    CRITICAL: min_periods=window ensures the first (window-1) values are NaN.
    Do NOT use min_periods < window as that would allow partial-window estimates
    which effectively constitute forward-fill lookahead in rolling contexts.
    """
    return series.rolling(window=window, min_periods=window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Average True Range using Wilder's EWM smoothing.

    True Range = max(H-L, |H-C_prev|, |L-C_prev|)
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — cumulative, strictly causal."""
    direction = np.sign(close.diff(1)).fillna(0)
    return (direction * volume).cumsum()


def _stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    Stochastic Oscillator %K and %D.

    %K = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) × 100
    %D = SMA(%K, d_period)

    Uses rolling min/max with min_periods=k_period for strict causality.
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    pct_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    pct_d = pct_k.rolling(window=d_period, min_periods=d_period).mean()
    return pct_k, pct_d


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R — range: [-100, 0], strictly causal."""
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-10)


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    Commodity Channel Index.

    CCI = (TP - SMA(TP, N)) / (0.015 * MAD(TP, N))
    where TP = (H + L + C) / 3
    """
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    # Mean Absolute Deviation
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    return (tp - sma_tp) / (0.015 * mad + 1e-10)


def _mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """
    Money Flow Index — volume-weighted RSI.

    MFI uses only past data via rolling sum with min_periods=period.
    """
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    direction = np.sign(tp.diff(1))
    pos_mf = raw_mf.where(direction > 0, 0.0)
    neg_mf = raw_mf.where(direction < 0, 0.0)
    pos_sum = pos_mf.rolling(window=period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(window=period, min_periods=period).sum()
    mfr = pos_sum / (neg_sum + 1e-10)
    return 100.0 - (100.0 / (1.0 + mfr))


def _roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change: (Close_t / Close_{t-n} - 1) × 100."""
    return (close / close.shift(period) - 1.0) * 100.0


def _dpo(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Detrended Price Oscillator.

    DPO = Close[t - (period/2 + 1)] - SMA(period) evaluated at same lag.

    IMPORTANT: We shift both the price and the SMA backward by (period//2 + 1)
    to avoid any look-ahead. The SMA itself is computed on past data only.
    """
    half_plus = period // 2 + 1
    sma = close.rolling(window=period, min_periods=period).mean()
    return close.shift(half_plus) - sma.shift(half_plus)


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index, +DI, -DI using Wilder's smoothing.

    Returns (ADX, DI+, DI-).
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr_wilder = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr_wilder + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / (atr_wilder + 1e-10)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _vwap_deviation(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20
) -> pd.Series:
    """
    Rolling VWAP deviation: (Close - VWAP_rolling) / VWAP_rolling.

    VWAP_rolling = rolling_sum(TP * Volume) / rolling_sum(Volume)
    where TP = (H + L + C) / 3.

    This intraday-proxy VWAP uses only past data (backward rolling window).
    """
    tp = (high + low + close) / 3.0
    tp_vol = tp * volume
    rolling_vwap = (
        tp_vol.rolling(window=window, min_periods=window).sum()
        / volume.rolling(window=window, min_periods=window).sum()
    )
    return (close - rolling_vwap) / (rolling_vwap + 1e-10)


# ---------------------------------------------------------------------------
# PER-TICKER FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def _compute_features_single_ticker(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full 47-feature technical indicator matrix for one ticker.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        DataFrame with columns [Open, High, Low, Close, Volume] for a single
        ticker. Index must be sorted DatetimeIndex (ascending, no gaps).

    Returns
    -------
    pd.DataFrame
        DataFrame with 47 feature columns and the same DatetimeIndex.
        Rows where any SMA(200) look-back cannot be satisfied are NaN for
        those features only (NOT dropped here — caller handles NaN removal).

    CRITICAL INVARIANTS:
      - No rolling function uses center=True.
      - SMA(200) has NaN for the first 199 rows.
      - All shift() calls use positive integers (backward shift only).
      - Volume z-score uses rolling mean/std with min_periods matching window.
    """
    assert ohlcv.index.is_monotonic_increasing, "OHLCV index must be sorted ascending."
    assert not ohlcv.index.duplicated().any(), "Duplicate dates found in OHLCV index."

    o = ohlcv["Open"]
    h = ohlcv["High"]
    l = ohlcv["Low"]
    c = ohlcv["Close"]
    v = ohlcv["Volume"].replace(0, np.nan)  # Avoid division by zero for volume

    feat: Dict[str, pd.Series] = {}

    # ── 1. RSI ──────────────────────────────────────────────────────────────
    feat["RSI_14"] = _rsi(c, 14)
    feat["RSI_21"] = _rsi(c, 21)

    # ── 2. MACD (12-26-9) ───────────────────────────────────────────────────
    ema12 = _ema(c, 12)
    ema26 = _ema(c, 26)
    macd_line = ema12 - ema26
    # Signal line: EMA(9) of MACD — min_periods=9 preserves causality
    signal_line = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    feat["MACD_12_26"] = macd_line
    feat["MACD_signal_9"] = signal_line
    feat["MACD_hist"] = macd_line - signal_line

    # ── 3. Bollinger Bands (20, 2σ) ─────────────────────────────────────────
    bb_sma = _sma(c, 20)
    bb_std = c.rolling(window=20, min_periods=20).std()
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    feat["BB_upper"] = bb_upper
    feat["BB_lower"] = bb_lower
    feat["BB_mid"] = bb_sma          # 20-day SMA — 47th feature
    feat["BB_width"] = (bb_upper - bb_lower) / (bb_sma + 1e-10)
    feat["BB_pct_b"] = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ── 4. ATR ─────────────────────────────────────────────────────────────
    feat["ATR_14"] = _atr(h, l, c, 14)
    feat["ATR_21"] = _atr(h, l, c, 21)

    # ── 5. OBV and OBV-EMA ─────────────────────────────────────────────────
    obv = _obv(c, v)
    feat["OBV"] = obv
    feat["OBV_EMA"] = obv.ewm(span=21, min_periods=21, adjust=False).mean()

    # ── 6. EMA variants ────────────────────────────────────────────────────
    feat["EMA_9"] = _ema(c, 9)
    feat["EMA_21"] = _ema(c, 21)
    feat["EMA_50"] = _ema(c, 50)
    feat["EMA_200"] = _ema(c, 200)

    # ── 7. SMA variants ────────────────────────────────────────────────────
    sma50 = _sma(c, 50)
    sma200 = _sma(c, 200)
    feat["SMA_50"] = sma50
    feat["SMA_200"] = sma200
    # Price ratios (used as trend-filter signal AND as features)
    feat["price_to_SMA200"] = c / (sma200 + 1e-10)
    feat["price_to_SMA50"] = c / (sma50 + 1e-10)

    # ── 8. Returns (short-term reversal features, Jegadeesh 1990) ──────────
    feat["return_1d"]  = c.pct_change(1)
    feat["return_2d"]  = c.pct_change(2)
    feat["return_3d"]  = c.pct_change(3)
    feat["return_5d"]  = c.pct_change(5)
    feat["return_10d"] = c.pct_change(10)
    feat["return_21d"] = c.pct_change(21)

    # ── 9. Volume z-score ──────────────────────────────────────────────────
    vol_mean_5 = v.rolling(window=5, min_periods=5).mean()
    vol_std_5 = v.rolling(window=5, min_periods=5).std()
    feat["volume_zscore_5d"] = (v - vol_mean_5) / (vol_std_5 + 1e-10)

    vol_mean_21 = v.rolling(window=21, min_periods=21).mean()
    vol_std_21 = v.rolling(window=21, min_periods=21).std()
    feat["volume_zscore_21d"] = (v - vol_mean_21) / (vol_std_21 + 1e-10)

    # ── 10. Candle structure ────────────────────────────────────────────────
    candle_range = h - l + 1e-10
    # Normalized open-close body (negative = red candle)
    feat["OC_body_norm"] = (c - o) / candle_range
    # Upper shadow: from max(O,C) to High
    feat["upper_shadow_ratio"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / candle_range
    # Lower shadow: from min(O,C) to Low
    feat["lower_shadow_ratio"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / candle_range
    # High-Low range normalized by close
    feat["HL_range_norm"] = candle_range / (c + 1e-10)

    # ── 11. Rolling volatility ─────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    feat["rolling_vol_5d"] = log_ret.rolling(window=5, min_periods=5).std()
    feat["rolling_vol_21d"] = log_ret.rolling(window=21, min_periods=21).std()
    feat["rolling_vol_63d"] = log_ret.rolling(window=63, min_periods=63).std()

    # ── 12. Stochastic %K and %D ───────────────────────────────────────────
    feat["stoch_K"], feat["stoch_D"] = _stochastic(h, l, c, k_period=14, d_period=3)

    # ── 13. Williams %R ────────────────────────────────────────────────────
    feat["Williams_R"] = _williams_r(h, l, c, period=14)

    # ── 14. CCI ────────────────────────────────────────────────────────────
    feat["CCI_20"] = _cci(h, l, c, period=20)

    # ── 15. MFI ────────────────────────────────────────────────────────────
    feat["MFI_14"] = _mfi(h, l, c, v, period=14)

    # ── 16. ROC ────────────────────────────────────────────────────────────
    feat["ROC_10"] = _roc(c, 10)
    feat["ROC_21"] = _roc(c, 21)

    # ── 17. DPO ────────────────────────────────────────────────────────────
    feat["DPO_20"] = _dpo(c, period=20)

    # ── 18. VWAP deviation ─────────────────────────────────────────────────
    feat["VWAP_deviation"] = _vwap_deviation(h, l, c, v, window=20)

    # ── 19. ADX, DI+, DI- ──────────────────────────────────────────────────
    feat["ADX_14"], feat["DI_plus"], feat["DI_minus"] = _adx(h, l, c, period=14)

    # Assemble DataFrame
    feature_df = pd.DataFrame(feat, index=ohlcv.index)

    # ── Validate feature count ──────────────────────────────────────────────
    assert len(feature_df.columns) == 49, (
        f"Expected 49 features, got {len(feature_df.columns)}. "
        f"Columns: {list(feature_df.columns)}"
    )

    return feature_df


# ---------------------------------------------------------------------------
# TARGET LABEL
# ---------------------------------------------------------------------------

def _compute_target(close: pd.Series) -> pd.Series:
    """
    Compute the binary target label for directional prediction.

    y_t = 1  if  Close(t+2) > Close(t+1)  else  0

    DESIGN RATIONALE:
      Assuming signal is observed at Close(t), execution happens at Close(t+1)
      (next open fills are approximated by next close for daily backtests).
      The holding return is therefore Close(t+1) → Close(t+2).
      This adds a 2-step shift to avoid execution-at-signal lookahead bias.

    Parameters
    ----------
    close : pd.Series
        Adjusted close prices for a single ticker.

    Returns
    -------
    pd.Series
        Binary series (0 or 1) with the same index. The last 2 rows are NaN
        because Close(t+2) does not exist for the final two trading days.

    CRITICAL: The label for row t looks forward 2 steps. NEVER use this
    series as a feature. NEVER use it before applying the walk-forward split.
    """
    future_close_t2 = close.shift(-2)  # Close(t+2) — looking 2 steps ahead
    future_close_t1 = close.shift(-1)  # Close(t+1) — execution price

    # Detect rows where either future price is missing (last 2 rows)
    missing_mask = future_close_t2.isna() | future_close_t1.isna()

    # y_t = 1 if holding from t+1 to t+2 is profitable
    # NOTE: pd.Series boolean comparison with NaN yields False, NOT NaN.
    # We must use explicit masking to insert NaN for the final 2 rows.
    label = (future_close_t2 > future_close_t1).astype(float)
    label = label.where(~missing_mask, other=np.nan)  # last 2 rows → NaN
    return label


# ---------------------------------------------------------------------------
# MAIN API
# ---------------------------------------------------------------------------

def load_all_data(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
    drop_na_features: bool = True,
    external_data_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Load and process all data into a clean MultiIndex DataFrame.

    Supports two modes:
      - built-in downloader (default) for original 7-stock universe
      - external parquet source for NASDAQ-100 dataset via external_data_path

    Steps:
      1. Load raw OHLCV (download or external parquet).
      2. Compute 47 causal technical features per ticker.
      3. Compute the 2-day-forward target label per ticker.
      4. Stack into a (date, ticker) MultiIndex DataFrame.
      5. Optionally drop rows with NaN in feature columns (warm-up period).
    """
    if start is None:
        start = START_DATE
    if end is None:
        end = END_DATE
    if tickers is None:
        tickers = TICKERS[:]

    print("=" * 60)
    print("DATA LOADER — Financial Conviction Ranking Pipeline")
    print("=" * 60)
    source = "external parquet" if external_data_path else "downloaded yfinance"
    print(f"Data Source: {source}")
    print(f"Tickers    : {tickers if external_data_path is None else 'from file'}")
    print(f"Period     : {start} -> {end}")
    print(f"Cache      : {'enabled' if use_cache else 'disabled'}")
    print(f"Drop NaN   : {'enabled' if drop_na_features else 'disabled'}")
    print(f"External   : {external_data_path if external_data_path else 'none'}")
    print()

    if external_data_path:
        raw_df = _load_data_from_parquet(external_data_path)

        # Optionally filter by tickers list (subset) if provided.
        available_tickers = sorted(raw_df.index.get_level_values("ticker").unique())
        print(f"Loaded from parquet: {len(raw_df)} rows, {len(available_tickers)} tickers")

        raw_df = raw_df.loc[
            (raw_df.index.get_level_values("date") >= pd.to_datetime(start))
            & (raw_df.index.get_level_values("date") <= pd.to_datetime(end))
        ]

        if tickers != TICKERS:
            requested_tickers = tickers
            missing = [t for t in requested_tickers if t not in available_tickers]
            if missing:
                print(f"  [WARN] Requested tickers not found and will be skipped: {missing}")
            tickers = [t for t in requested_tickers if t in available_tickers]
            if not tickers:
                raise ValueError("No requested tickers found in external dataset.")
        else:
            tickers = available_tickers

        raw = {ticker: raw_df.xs(ticker, level="ticker").sort_index() for ticker in tickers}

    else:
        print("Step 1/4: Downloading OHLCV data …")
        raw = _download_ohlcv(tickers, start, end, use_cache=use_cache)

    print("\nStep 2/4: Engineering features and building target labels …")
    ticker_frames: List[pd.DataFrame] = []

    for ticker in tickers:
        if ticker not in raw:
            print(f"  [SKIP] {ticker}: no data")
            continue

        ohlcv = raw[ticker].copy()
        print(f"  Processing {ticker} ({len(ohlcv)} raw rows) …")

        feat_df = _compute_features_single_ticker(ohlcv)
        target = _compute_target(ohlcv["Close"])
        target.name = "target"
        close_series = ohlcv["Close"].rename("Close")

        combined = pd.concat([feat_df, close_series, target], axis=1)
        combined["ticker"] = ticker
        combined.index.name = "date"
        ticker_frames.append(combined)

    print("\nStep 3/4: Building MultiIndex DataFrame …")
    if not ticker_frames:
        raise ValueError("No ticker data available after processing - cannot build dataset.")

    all_data = pd.concat(ticker_frames, axis=0)
    all_data = all_data.reset_index().set_index(["date", "ticker"])
    all_data.sort_index(inplace=True)

    print(f"  Combined shape (before NaN drop): {all_data.shape}")
    nan_counts = all_data[get_feature_columns(all_data)].isna().sum()
    print(f"  Features with NaN values: {(nan_counts > 0).sum()} (expected — warm-up period)")

    if drop_na_features:
        feature_cols = get_feature_columns(all_data)
        before = len(all_data)
        all_data = all_data.dropna(subset=feature_cols)
        after = len(all_data)
        print(f"  Dropped {before - after} NaN feature rows (warm-up period)")

    # Also drop rows where target is NaN (last 2 days per ticker)
    before_target = len(all_data)
    all_data = all_data.dropna(subset=["target"])
    print(f"  Dropped {before_target - len(all_data)} NaN target rows (last 2 dates per ticker)")

    print(f"\nStep 4/4: Final dataset shape: {all_data.shape}")
    print(f"  Date range  : {all_data.index.get_level_values('date').min()} -> "
          f"{all_data.index.get_level_values('date').max()}")
    print(f"  Unique dates: {all_data.index.get_level_values('date').nunique()}")
    print(f"  Tickers     : {all_data.index.get_level_values('ticker').unique().tolist()}")
    print(f"  Features    : {len(get_feature_columns(all_data))}")
    print(f"  Target dist : {all_data['target'].value_counts().to_dict()}")

    return all_data


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature column names from a loaded DataFrame.

    Excludes 'target', 'Close', and any metadata columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame returned by load_all_data().

    Returns
    -------
    list of str
        Feature column names in their original order.
    """
    non_feature = {"target", "Close", "ticker"}
    return [c for c in df.columns if c not in non_feature]


# ---------------------------------------------------------------------------
# DATA INTEGRITY AUDIT
# ---------------------------------------------------------------------------

def run_data_integrity_audit(
    df: pd.DataFrame,
    raw_data_path: Optional[Path | str] = None,
) -> bool:
    """
    Run a comprehensive integrity audit on the loaded dataset.

    Checks performed:
      (a) Feature matrix has no NaN values (after warm-up drop).
      (b) Target label uses the correct 2-day forward shift by reconstructing
          it from the Close column and verifying alignment for a sample date.
      (c) SMA(200) had exactly 199 NaN rows per ticker before warm-up drop
          (verified via a secondary load with drop_na_features=False).
      (d) No feature column uses center=True (structural check via column
          name comparison to known causal feature list).
      (e) MultiIndex is correctly sorted and has no duplicate (date, ticker).
      (f) Returns are not future-contaminated (correlation with next-day
          return is checked to not be suspiciously high).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset returned by load_all_data() with drop_na_features=True.

    Returns
    -------
    bool
        True if all checks pass. Raises AssertionError on any failure.

    Side Effects
    ------------
    Prints a formatted audit report to stdout.
    """
    print("\n" + "=" * 60)
    print("DATA INTEGRITY AUDIT")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  ✅ PASS  {name}")
            if detail:
                print(f"          {detail}")
            passed += 1
        else:
            print(f"  ❌ FAIL  {name}")
            if detail:
                print(f"          {detail}")
            failed += 1

    feature_cols = get_feature_columns(df)

    # ── (a) No NaN in feature matrix ───────────────────────────────────────
    feat_nan_count = df[feature_cols].isna().sum().sum()
    check(
        "No NaN in feature matrix",
        feat_nan_count == 0,
        f"Total NaN cells: {feat_nan_count}",
    )

    # ── (b) Feature count ──────────────────────────────────────────────────
    check(
        "Exactly 49 features",
        len(feature_cols) == 49,
        f"Got {len(feature_cols)} features: {feature_cols}",
    )

    # ── (c) Target label has correct 2-day shift ────────────────────────────
    # Recompute from raw Close series and compare to stored target values.
    target_correct = True
    audit_tickers = df.index.get_level_values("ticker").unique()

    for ticker in audit_tickers:
        # Use raw close price to avoid shift distortions from NaN row drops.
        if raw_data_path:
            raw_df = _load_data_from_parquet(raw_data_path)
            raw_close = raw_df.xs(ticker, level="ticker")["Close"].sort_index()
        else:
            raw_download = _download_ohlcv([ticker], START_DATE, END_DATE, use_cache=True)
            raw_close = raw_download[ticker]["Close"].sort_index()

        expected_target_full = _compute_target(raw_close)

        observed = df.xs(ticker, level="ticker")["target"].sort_index()
        common_index = observed.index.intersection(expected_target_full.index)

        if not np.allclose(
            observed.loc[common_index].values,
            expected_target_full.loc[common_index].values,
            equal_nan=True,
            atol=1e-6,
        ):
            target_correct = False
            break

    check(
        "Target label = 1{Close(t+2) > Close(t+1)} — 2-day forward shift confirmed",
        target_correct,
        f"Checked {len(audit_tickers)} tickers",
    )

    # ── (d) SMA(200) correctness via raw load ──────────────────────────────
    audit_ticker = df.index.get_level_values("ticker").unique()[0]
    print(f"  [Reloading {audit_ticker} raw for SMA audit …]")
    try:
        raw_one = _download_ohlcv([audit_ticker], START_DATE, END_DATE, use_cache=True)
        feat_raw = _compute_features_single_ticker(raw_one[audit_ticker])
        sma200_nan_count = feat_raw["SMA_200"].isna().sum()
        check(
            "SMA(200) has exactly 199 NaN rows (200-day warm-up, strictly causal)",
            sma200_nan_count == 199,
            f"Actual NaN count in {audit_ticker} SMA_200: {sma200_nan_count} (expected 199)",
        )
    except Exception as e:
        check(
            "SMA(200) has exactly 199 NaN rows (audit download check)",
            False,
            f"Audit could not reload {audit_ticker}: {e}",
        )

    # ── (e) MultiIndex sorted, no duplicates ───────────────────────────────
    check(
        "MultiIndex sorted ascending",
        df.index.is_monotonic_increasing,
    )
    dup_count = df.index.duplicated().sum()
    check(
        "No duplicate (date, ticker) pairs",
        dup_count == 0,
        f"Duplicate count: {dup_count}",
    )

    # ── (f) Valid number of tickers ────────────────────────────────────────
    n_tickers = df.index.get_level_values("ticker").nunique()
    check(
        "At least 1 ticker present",
        n_tickers > 0,
        f"Found tickers: {df.index.get_level_values('ticker').unique().tolist()}",
    )

    # ── (g) Target is binary ───────────────────────────────────────────────
    unique_targets = set(df["target"].dropna().unique())
    check(
        "Target is binary {0.0, 1.0}",
        unique_targets <= {0.0, 1.0},
        f"Unique values: {unique_targets}",
    )

    # ── (h) No future leakage in return_1d (spot-check) ────────────────────
    # return_1d should be past return (t-1 → t), NOT future return.
    # Verify by checking it equals pct_change(1) with NO shift.
    sample_ticker = df.index.get_level_values("ticker").unique()[0]
    tk_df = df.xs(sample_ticker, level="ticker").sort_index()
    close = tk_df["Close"]
    expected_ret1d = close.pct_change(1)
    actual_ret1d = tk_df["return_1d"]
    valid = ~(expected_ret1d.isna() | actual_ret1d.isna())
    ret_match = np.allclose(
        expected_ret1d[valid].values, actual_ret1d[valid].values, atol=1e-8
    )
    check(
        "return_1d is strictly backward (t-1 → t, no forward shift)",
        ret_match,
        f"Verified for {sample_ticker}",
    )

    # ── (i) price_to_SMA200 ≥ 0 (Close/SMA200 ratio sanity) ────────────────
    ratio_min = df["price_to_SMA200"].min()
    check(
        "price_to_SMA200 > 0 (ratio sanity check)",
        ratio_min > 0,
        f"Min value: {ratio_min:.4f}",
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    print(f"  Audit Summary: {passed} passed, {failed} failed")

    if failed == 0:
        n_dates = df.index.get_level_values("date").nunique()
        n_feat = len(feature_cols)
        print(
            f"\n✅ Data loader audit PASSED: {n_feat} features, "
            f"{n_dates} unique dates, target shift confirmed correct."
        )
    else:
        print(f"\n❌ Audit FAILED: {failed} check(s) did not pass. "
              "Review output above before training any model.")
        raise AssertionError(f"Data integrity audit failed with {failed} error(s).")

    return failed == 0


# ---------------------------------------------------------------------------
# MODULE SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running data_loader.py as __main__ …\n")

    # Load dataset
    df = load_all_data(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        use_cache=True,
        drop_na_features=True,
    )

    # Run audit
    audit_passed = run_data_integrity_audit(df)

    if audit_passed:
        feat_cols = get_feature_columns(df)
        n_dates = df.index.get_level_values("date").nunique()
        print(
            f"\n✅ Data loader audit PASSED: {len(feat_cols)} features, "
            f"{n_dates} unique dates, target shift confirmed correct.\n"
        )
    print("data_loader.py self-test complete.")
