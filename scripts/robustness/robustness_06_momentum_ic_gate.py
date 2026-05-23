"""
robustness_06_momentum_ic_gate.py
===================================
ICGDF Positive Control: IC Gate Applied to Momentum Signal

This script demonstrates that the ICGDF gate OPENS for a momentum signal,
validating that the gate correctly distinguishes genuine cross-sectional
signal from noise. This is the critical positive control for the paper.

Signal : Trailing 252-day return (Jegadeesh & Titman 1993 momentum)
Gate   : Same two-stage ICGDF gate (HAC Newey-West t-test lag=9, B=1,000)
Window : Oct 2018 – Oct 2024 (same 1,512 OOS days as main experiment)
Universe: Full 30-stock NASDAQ-100 continuous-member universe
          Loaded from data/nasdaq30_prices.parquet (primary source)

Run from repo root:
    python scripts/robustness/robustness_06_momentum_ic_gate.py

Outputs:
    results/robustness/momentum_ic/momentum_ic_gate_results.csv
    results/robustness/momentum_ic/daily_ic_series.csv
    Console: formatted table — copy values into Table 10 of the manuscript.

Dependencies: numpy, pandas, scipy, yfinance (all already installed in venv)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (required for Unicode box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ──────────────────────────────────────────────────────────────
# Full 30-stock NASDAQ-100 continuous-member universe (same as main experiment)
TICKERS = [
    "AAPL", "ADBE", "ADI",  "ADP",  "AMAT", "AMGN", "AMZN", "AVGO", "BIIB",
    "CDNS", "COST", "CSCO", "GILD", "GOOGL","INTC", "INTU", "KLAC", "LRCX",
    "MCHP", "MDLZ", "META", "MSFT", "NFLX", "NVDA", "PYPL", "QCOM", "REGN",
    "SBUX", "TSLA", "TXN",
]
# Primary data source — same parquet used by the main pipeline
NASDAQ30_PARQUET = Path("data/nasdaq30_prices.parquet")

START_DATE    = "2015-01-01"   # need momentum_win days before OOS start
END_DATE      = "2024-12-31"
OOS_START     = "2018-10-01"   # same as main experiment
OOS_END       = "2024-10-31"
HAC_LAG       = 9              # same as ICGDF gate (paper Section 4.3)
PERM_B        = 1_000          # permutation replicates
ALPHA         = 0.05           # significance level
MOMENTUM_WIN  = 252            # 1-year trailing return window (trading days)

OUT_DIR = Path("results/robustness/momentum_ic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_prices() -> pd.DataFrame:
    """
    Load adjusted close prices for all 30 tickers.

    Primary source: data/nasdaq30_prices.parquet (MultiIndex date×ticker)
    Fallback:       Yahoo Finance download (slower, requires network)
    """
    # ── Primary: load from master parquet ─────────────────────────────────────
    if NASDAQ30_PARQUET.exists():
        print(f"  [PARQUET] Loading prices from {NASDAQ30_PARQUET} ...")
        raw = pd.read_parquet(NASDAQ30_PARQUET)
        # raw is MultiIndex (date, ticker) with OHLCV columns topivot to wide Close
        raw = raw.reset_index()
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        prices = raw.pivot(index="date", columns="ticker", values="Close")
        prices.index = pd.DatetimeIndex(prices.index)
        prices = prices[sorted(prices.columns)]          # canonical column order
        prices = prices.dropna(how="all")
        # Restrict to TICKERS subset (parquet may have extra symbols)
        cols_available = [t for t in TICKERS if t in prices.columns]
        missing = [t for t in TICKERS if t not in prices.columns]
        if missing:
            print(f"  [WARN] Tickers not found in parquet: {missing}")
        prices = prices[cols_available]
    else:
        # ── Fallback: Yahoo Finance download ──────────────────────────────────
        cache = OUT_DIR / "price_cache.parquet"
        if cache.exists():
            print(f"  [CACHE] Loading prices from {cache}")
            prices = pd.read_parquet(cache)
        else:
            print(f"  [DOWNLOAD] Fetching {len(TICKERS)} tickers from Yahoo Finance ...")
            raw = yf.download(
                TICKERS, start=START_DATE, end=END_DATE,
                auto_adjust=True, progress=False
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if "Close" in raw.columns else raw
            prices = prices.dropna(how="all")
            prices.to_parquet(cache)
            print(f"  [SAVED] Cache written to {cache}")

    print(f"  Price data shape: {prices.shape[0]:,} days × {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].date()} to{prices.index[-1].date()}")
    return prices


# ── IC computation ────────────────────────────────────────────────────────────
def compute_daily_momentum_ic(prices: pd.DataFrame) -> pd.Series:
    """
    For each day d in the OOS window:
      momentum_signal_d = trailing MOMENTUM_WIN-day return for each stock
      next_return_d     = 1-day forward return (next trading day)
      IC_d              = SpearmanRankCorr(momentum_ranks, next_return_ranks)

    Returns a pd.Series indexed by date.
    """
    # Trailing momentum return: prices[d] / prices[d - 252] - 1
    momentum = prices.pct_change(MOMENTUM_WIN)

    # Next-day return: prices[d+1] / prices[d] - 1  (shift(-1) to align same date)
    next_ret = prices.pct_change(1).shift(-1)

    # OOS window mask
    oos_mask = (prices.index >= OOS_START) & (prices.index <= OOS_END)
    oos_dates = prices.index[oos_mask]

    ic_records = []
    for d in oos_dates:
        mom_row = momentum.loc[d].dropna()
        nxt_row = next_ret.loc[d].reindex(mom_row.index).dropna()
        shared  = mom_row.index.intersection(nxt_row.index)

        if len(shared) < 4:   # need at least 4 stocks for meaningful rank corr
            continue

        m_vals = mom_row[shared].values.astype(float)
        n_vals = nxt_row[shared].values.astype(float)

        # Spearman rank correlation
        rho, _ = scipy.stats.spearmanr(m_vals, n_vals)
        if not np.isnan(rho):
            ic_records.append({"date": d, "ic": rho})

    ic_series = pd.DataFrame(ic_records).set_index("date")["ic"]
    return ic_series


# ── HAC Newey-West t-test ──────────────────────────────────────────────────────
def hac_ttest_onesided(ic_vals: np.ndarray, lag: int = HAC_LAG) -> tuple[float, float]:
    """
    One-sided HAC t-test: H0: mean IC <= 0,  H1: mean IC > 0.
    Uses Newey-West bandwidth with the same lag as the ICGDF gate.

    Returns
    -------
    t_stat  : float — HAC t-statistic
    p_value : float — one-sided p-value
    """
    n       = len(ic_vals)
    mean_ic = ic_vals.mean()
    resid   = ic_vals - mean_ic

    # Newey-West HAC variance
    hac_var = float(np.dot(resid, resid)) / n
    for k in range(1, lag + 1):
        w   = 1.0 - k / (lag + 1.0)
        cov = float(np.dot(resid[k:], resid[:-k])) / n
        hac_var += 2.0 * w * cov

    hac_var = max(hac_var, 1e-16)
    se      = np.sqrt(hac_var / n)
    t_stat  = mean_ic / se

    # One-sided p-value from standard normal (large-sample)
    p_value = float(scipy.stats.norm.sf(t_stat))
    return float(t_stat), p_value


# ── Permutation test ──────────────────────────────────────────────────────────
def permutation_test_onesided(ic_vals: np.ndarray, B: int = PERM_B) -> float:
    """
    Permutation test: H0: mean IC <= 0.
    Returns empirical one-sided p-value = P(permuted mean >= observed mean | H0).
    """
    obs_mean = ic_vals.mean()
    count = 0
    for _ in range(B):
        perm_mean = np.random.permutation(ic_vals).mean()
        if perm_mean >= obs_mean:
            count += 1
    return count / B


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print()
    print("=" * 65)
    print("  MOMENTUM IC GATE TEST — ICGDF POSITIVE CONTROL VALIDATION")
    print("=" * 65)

    # 1. Load prices
    print("\n[STEP 1] Loading price data ...")
    prices = load_prices()

    # 2. Compute daily IC
    print("\n[STEP 2] Computing cross-sectional momentum IC series ...")
    ic_series = compute_daily_momentum_ic(prices)
    n_days    = len(ic_series)
    print(f"  IC series: {n_days:,} daily observations")
    print(f"  OOS range: {ic_series.index[0].date()} to{ic_series.index[-1].date()}")

    # 3. Summary statistics
    ic_vals = ic_series.values
    mean_ic = float(ic_vals.mean())
    ic_std  = float(ic_vals.std(ddof=1))
    icir    = mean_ic / ic_std if ic_std > 0 else 0.0

    # AR(1) coefficient — quantifies serial dependence, justifying HAC
    ar1_coeff = float(np.corrcoef(ic_vals[:-1], ic_vals[1:])[0, 1]) if n_days > 2 else 0.0

    # 4. HAC t-test (Gate Condition A)
    print("\n[STEP 3] Applying ICGDF gate — HAC t-test (lag=9) ...")
    t_stat, p_hac = hac_ttest_onesided(ic_vals)
    gate_cond_a   = bool(t_stat > 1.645 and mean_ic > 0)

    # 5. Permutation test (Gate Condition B)
    print(f"[STEP 4] Permutation test (B={PERM_B:,}) ...")
    p_perm      = permutation_test_onesided(ic_vals)
    gate_cond_b = bool(p_perm < ALPHA)

    # 6. Final gate decision
    gate_open = gate_cond_a and gate_cond_b

    # ── Print results ──────────────────────────────────────────────────────────
    sep = "─" * 65
    print(f"\n{sep}")
    print("  IC SERIES STATISTICS")
    print(sep)
    print(f"  Signal         : Trailing 252-day return (momentum)")
    print(f"  Universe       : {len(TICKERS)} NASDAQ-100 continuous-member stocks")
    print(f"  OOS window     : {OOS_START} to{OOS_END} ({n_days:,} days)")
    print(f"  Mean IC        : {mean_ic:+.6f}")
    print(f"  IC Std Dev     : {ic_std:.6f}")
    print(f"  ICIR           : {icir:+.4f}")
    print(f"  AR(1) coeff    : {ar1_coeff:+.4f}  (autocorrelation toHAC necessary)")

    print(f"\n{sep}")
    print("  HAC T-TEST  (Newey-West, lag=9)")
    print(sep)
    print(f"  t-statistic    : {t_stat:+.4f}")
    print(f"  p-value        : {p_hac:.6f}  (one-sided, H1: mean IC > 0)")
    print(f"  Critical val.  : 1.645  (α=0.05)")
    cond_a_str = "PASS ✓" if gate_cond_a else "FAIL ✗"
    print(f"  Gate Cond. A   : {cond_a_str}  (t > 1.645 AND IC̄ > 0)")

    print(f"\n{sep}")
    print(f"  PERMUTATION TEST  (B={PERM_B:,} shuffles)")
    print(sep)
    print(f"  Empirical p    : {p_perm:.4f}")
    cond_b_str = "PASS ✓" if gate_cond_b else "FAIL ✗"
    print(f"  Gate Cond. B   : {cond_b_str}  (p_perm < 0.05)")

    gate_label = "OPEN  ← gate correctly opens for genuine signal" if gate_open else "CLOSED ✗"
    print(f"\n{sep}")
    print(f"  GATE DECISION  : {gate_label}")
    print(sep)

    # ── Comparison table for manuscript ───────────────────────────────────────
    print(f"\n{sep}")
    print("  SIGNAL COMPARISON  (for Table 10 / Section 6.6)")
    print(sep)
    header = f"  {'Signal':<28} {'Mean IC':>9} {'ICIR':>7} {'HAC t':>8} {'p-val':>8} {'Gate':>8}"
    print(header)
    print(f"  {'-'*28} {'-'*9} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Momentum (252d trailing)':<28} {mean_ic:>+9.5f} {icir:>7.3f}"
          f" {t_stat:>+8.3f} {p_hac:>8.5f} {'OPEN' if gate_open else 'CLOSED':>8}")
    print(f"  {'ML Ensemble (baseline)':<28} {-0.0005:>+9.5f} {-0.0023:>7.4f}"
          f" {-0.0901:>+8.3f} {'0.46412':>8} {'CLOSED':>8}")
    print(sep)

    # ── Save CSV ───────────────────────────────────────────────────────────────
    results = pd.DataFrame([{
        "signal":         "Momentum_252d_trailing",
        "n_days":         n_days,
        "mean_ic":        mean_ic,
        "ic_std":         ic_std,
        "icir":           icir,
        "ar1_coeff":      ar1_coeff,
        "hac_t_stat":     t_stat,
        "hac_p_value":    p_hac,
        "gate_cond_a":    gate_cond_a,
        "perm_p_value":   p_perm,
        "gate_cond_b":    gate_cond_b,
        "gate_open":      gate_open,
    }])
    results.to_csv(OUT_DIR / "momentum_ic_gate_results.csv", index=False)

    ic_series.to_frame("momentum_ic").to_csv(OUT_DIR / "daily_ic_series.csv")

    print(f"\n  [SAVED] {OUT_DIR}/momentum_ic_gate_results.csv")
    print(f"  [SAVED] {OUT_DIR}/daily_ic_series.csv")
    print()
    print("  Results saved to results/robustness/momentum_ic/")
    print("  Use momentum_ic_gate_results.csv to populate Table 10 in the manuscript.")
    print()


if __name__ == "__main__":
    main()
