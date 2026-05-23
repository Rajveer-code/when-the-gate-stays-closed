#!/usr/bin/env python3
"""
nifty50_replication.py — Task 4: Nifty 50 Cross-Market Replication
==================================================================
Downloads 20 Nifty 50 stocks, applies the identical ICGDF pipeline
(12-fold walk-forward, 49 features, CatBoost+RF+MLP ensemble, HAC+permutation gate),
and reports IC gate statistics.

Output:
  data/nifty50_prices.parquet
  results/robustness/nifty50/ic_gate_results.csv
  results/robustness/nifty50/data_integrity.csv
  results/robustness/nifty50/ticker_log.csv

Author: Rajveer Singh Pall
"""

from __future__ import annotations
import os, sys, warnings, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# Fix SSL certificate verification on Windows (curl_cffi backend)
from curl_cffi.requests import Session as _CurlSession
_orig_curl_init = _CurlSession.__init__
def _patched_curl_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_curl_init(self, *args, **kwargs)
_CurlSession.__init__ = _patched_curl_init

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf

# Resolve repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.data_loader import load_all_data, get_feature_columns
from src.training.walk_forward import generate_folds, get_fold_arrays, get_cal_arrays
from src.training.models import EnsembleModel
from src.training.calibration import (
    fit_calibrator, calibrated_predict, compute_spearman_ic,
    test_ic_significance, compute_ece
)

print("=" * 70)
print("TASK 4: Nifty 50 Cross-Market Replication")
print("=" * 70)

# ── Config ────────────────────────────────────────────────────────────────────
PRIMARY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "NTPC.NS",
]

BACKUP_TICKERS = [
    "BAJAJFINSV.NS", "HCLTECH.NS", "TECHM.NS", "DRREDDY.NS", "DIVISLAB.NS",
]

START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
TEST_START = pd.Timestamp("2018-10-01")
TEST_END = pd.Timestamp("2024-10-31")
MISSING_THRESHOLD = 0.05  # 5%
SEED = 42
HAC_LAG = 9
PERM_B = 1000

OUT_DIR = _ROOT / "results" / "robustness" / "nifty50"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = _ROOT / "data" / "nifty50_prices.parquet"

np.random.seed(SEED)


def download_nifty50_data():
    """Download OHLCV for Nifty 50 tickers, substitute failures."""
    print("\n[1/5] Downloading Nifty 50 data...")

    all_dfs = {}
    failed_tickers = []
    substitutions = []

    for ticker in PRIMARY_TICKERS:
        try:
            print(f"  Downloading {ticker}...", end=" ")
            data = yf.download(ticker, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if data is None or len(data) < 100:
                print(f"FAIL (only {len(data) if data is not None else 0} rows)")
                failed_tickers.append(ticker)
                continue

            # Check missing in test window
            test_data = data.loc[TEST_START:TEST_END]
            # Estimate expected trading days for NSE (~245/year * 6 years)
            expected_days = 245 * 6
            missing_pct = 1 - len(test_data) / expected_days if expected_days > 0 else 0

            if missing_pct > MISSING_THRESHOLD:
                print(f"FAIL ({missing_pct:.1%} missing in test window)")
                failed_tickers.append(ticker)
                continue

            all_dfs[ticker.replace(".NS", "")] = data
            print(f"OK ({len(data)} rows, test: {len(test_data)})")

        except Exception as e:
            print(f"ERROR: {e}")
            failed_tickers.append(ticker)

    # Substitute failed tickers
    for failed in failed_tickers:
        if not BACKUP_TICKERS:
            break
        backup = BACKUP_TICKERS.pop(0)
        try:
            print(f"  Substituting {failed} -> {backup}...", end=" ")
            data = yf.download(backup, start=START_DATE, end=END_DATE,
                             auto_adjust=True, progress=False)
            if data is not None and len(data) >= 100:
                all_dfs[backup.replace(".NS", "")] = data
                substitutions.append({"failed": failed, "backup": backup, "rows": len(data)})
                print(f"OK ({len(data)} rows)")
            else:
                print("FAIL (backup also failed)")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n  Final universe: {len(all_dfs)} tickers")
    print(f"  Substitutions: {len(substitutions)}")

    # Build MultiIndex DataFrame matching expected format
    frames = []
    for ticker, df in all_dfs.items():
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)

        temp = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        temp["ticker"] = ticker
        temp.index.name = "date"
        temp = temp.reset_index()
        temp["date"] = pd.to_datetime(temp["date"]).dt.normalize()
        frames.append(temp)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["date", "ticker"]).sort_index()

    # Save
    combined.to_parquet(DATA_PATH)
    print(f"  Saved: {DATA_PATH} ({len(combined)} rows)")

    # Save substitution log
    sub_df = pd.DataFrame(substitutions) if substitutions else pd.DataFrame(
        columns=["failed", "backup", "rows"])
    sub_df.to_csv(OUT_DIR / "ticker_log.csv", index=False)

    return combined, substitutions


def data_integrity_checks(df):
    """Apply 6-check data integrity protocol."""
    print("\n[2/5] Data integrity checks...")

    results = []
    tickers = df.index.get_level_values("ticker").unique()

    for ticker in tickers:
        ticker_data = df.loc[df.index.get_level_values("ticker") == ticker]
        dates = ticker_data.index.get_level_values("date")

        # Check 1: Missing values
        n_missing = ticker_data[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum()

        # Check 2: Duplicate dates
        n_dupes = dates.duplicated().sum()

        # Check 3: Stale prices (same close 5+ consecutive days)
        closes = ticker_data["Close"].values
        max_stale = 0
        current_stale = 1
        for i in range(1, len(closes)):
            if abs(closes[i] - closes[i-1]) < 1e-8:
                current_stale += 1
                max_stale = max(max_stale, current_stale)
            else:
                current_stale = 1

        # Check 4: Extreme returns (>50% in a single day)
        returns = ticker_data["Close"].pct_change().abs()
        n_extreme = (returns > 0.5).sum()

        # Check 5: Volume zeros
        n_zero_vol = (ticker_data["Volume"] == 0).sum()

        # Check 6: High < Low violations
        n_hl_violations = (ticker_data["High"] < ticker_data["Low"]).sum()

        results.append({
            "ticker": ticker,
            "n_rows": len(ticker_data),
            "missing_values": n_missing,
            "duplicate_dates": n_dupes,
            "max_stale_days": max_stale,
            "extreme_returns_50pct": n_extreme,
            "zero_volume_days": n_zero_vol,
            "high_lt_low": n_hl_violations,
            "pass": (n_missing == 0 and n_dupes == 0 and max_stale < 10
                     and n_hl_violations == 0),
        })

    integrity_df = pd.DataFrame(results)
    integrity_df.to_csv(OUT_DIR / "data_integrity.csv", index=False)

    n_pass = integrity_df["pass"].sum()
    print(f"  {n_pass}/{len(integrity_df)} tickers pass all checks")

    # Print any failures
    failures = integrity_df[~integrity_df["pass"]]
    if len(failures) > 0:
        print("  FAILURES:")
        for _, row in failures.iterrows():
            print(f"    {row['ticker']}: missing={row['missing_values']}, "
                  f"dupes={row['duplicate_dates']}, stale={row['max_stale_days']}, "
                  f"HL={row['high_lt_low']}")

    return integrity_df


def run_pipeline(df):
    """Run full ICGDF pipeline on Nifty 50 data."""
    print("\n[3/5] Running full ICGDF pipeline...")

    # Load data through the standard pipeline
    nifty_data = load_all_data(external_data_path=str(DATA_PATH))
    feature_cols = get_feature_columns(nifty_data)
    print(f"  Data: {nifty_data.shape}, features: {len(feature_cols)}")

    # Generate folds
    print("\n[4/5] Walk-forward folds...")
    folds = generate_folds(nifty_data)
    print(f"  Folds: {len(folds)}")

    # Train and collect IC per fold
    all_daily_ics = []
    fold_results = []

    for i, fold in enumerate(folds):
        fold_num = i + 1
        print(f"\n  --- Fold {fold_num}/{len(folds)} ---")

        try:
            X_train, X_test, y_train, y_test, scaler = get_fold_arrays(
                fold, nifty_data, feature_cols, scaler=None
            )
            X_cal, y_cal = get_cal_arrays(fold, nifty_data, feature_cols, scaler)

            print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

            # Train ensemble
            n_features = X_train.shape[1]
            model = EnsembleModel()
            model.fit(X_train, y_train)

            # Calibrate
            calibrator = fit_calibrator(model, X_cal, y_cal)

            # Predict
            probs = calibrated_predict(model, calibrator, X_test)

            # Compute ECE
            ece = compute_ece(probs, y_test)

            # Get test dates and tickers for IC computation
            test_dates = fold.test_dates
            test_df = nifty_data.loc[nifty_data.index.get_level_values("date").isin(test_dates)]
            test_tickers = test_df.index.get_level_values("ticker").unique()

            # Compute daily IC
            n_stocks = len(test_tickers)
            n_days = len(test_dates)

            # Reshape predictions to match test structure
            # test_df should have same order as X_test
            test_returns = test_df["target"].values  # actual returns/labels

            daily_ics = []
            for d_idx, date in enumerate(test_dates):
                start = d_idx * n_stocks
                end = start + n_stocks
                if end > len(probs):
                    break

                day_probs = probs[start:end]
                day_returns = test_returns[start:end]

                if len(day_probs) >= 3 and np.std(day_probs) > 1e-10:
                    ic = compute_spearman_ic(day_probs, day_returns)
                    if not np.isnan(ic):
                        daily_ics.append(ic)

            all_daily_ics.extend(daily_ics)

            fold_ic_mean = np.mean(daily_ics) if daily_ics else 0.0
            fold_results.append({
                "fold": fold_num,
                "n_days": len(daily_ics),
                "mean_ic": fold_ic_mean,
                "ic_std": np.std(daily_ics) if daily_ics else 0.0,
                "ece": ece,
            })
            print(f"  IC days: {len(daily_ics)}, Mean IC: {fold_ic_mean:.4f}, ECE: {ece:.4f}")

        except Exception as e:
            print(f"  ERROR in fold {fold_num}: {e}")
            fold_results.append({
                "fold": fold_num, "n_days": 0, "mean_ic": np.nan,
                "ic_std": np.nan, "ece": np.nan
            })

    return all_daily_ics, fold_results


def _newey_west_tstat(ic_array, lag):
    """Compute HAC (Newey-West) t-statistic for mean(IC) > 0."""
    T = len(ic_array)
    mean_ic = np.mean(ic_array)
    resids = ic_array - mean_ic

    # Autocovariance
    gamma = np.zeros(lag + 1)
    for h in range(lag + 1):
        gamma[h] = np.sum(resids[:T-h] * resids[h:]) / T

    # Newey-West HAC variance
    hac_var = gamma[0]
    for h in range(1, lag + 1):
        w = 1 - h / (lag + 1)  # Bartlett kernel
        hac_var += 2 * w * gamma[h]

    stderr = np.sqrt(hac_var / T)
    t_stat = mean_ic / stderr if stderr > 0 else 0.0
    return t_stat, stderr


def compute_gate_decision(daily_ics):
    """Compute IC gate statistics and decision."""
    print("\n[5/5] Computing IC gate decision...")

    ic_array = np.array(daily_ics)
    mean_ic = np.mean(ic_array)
    ic_std = np.std(ic_array)
    T = len(ic_array)
    icir = mean_ic / ic_std if ic_std > 0 else 0.0

    # HAC t-test (Newey-West, lag=9)
    t_stat, stderr = _newey_west_tstat(ic_array, HAC_LAG)

    # One-tailed p-value (upper tail: H1: IC > 0)
    p_one_tailed = 1 - scipy.stats.norm.cdf(t_stat)

    # Permutation test
    observed_mean = mean_ic
    n_above = 0
    for b in range(PERM_B):
        perm_ic = np.random.permutation(ic_array)
        if np.mean(perm_ic) >= observed_mean:
            n_above += 1
    p_perm = n_above / PERM_B

    # Gate decision
    condition_a = t_stat > 1.645 and mean_ic > 0
    condition_b = p_perm < 0.05
    gate_open = condition_a and condition_b

    results = {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "T": T,
        "hac_t_stat": t_stat,
        "hac_p_value_one_tailed": p_one_tailed,
        "permutation_p_value": p_perm,
        "gate_decision": "OPEN" if gate_open else "CLOSED",
        "condition_a_met": condition_a,
        "condition_b_met": condition_b,
    }

    print(f"\n  {'=' * 50}")
    print(f"  NIFTY 50 IC GATE RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Mean IC:        {mean_ic:.4f}")
    print(f"  IC Std Dev:     {ic_std:.4f}")
    print(f"  ICIR:           {icir:.4f}")
    print(f"  T (days):       {T}")
    print(f"  HAC t-stat:     {t_stat:.4f}")
    print(f"  p (one-tailed): {p_one_tailed:.4f}")
    print(f"  Perm p-value:   {p_perm:.4f}")
    print(f"  Gate:           {'OPEN' if gate_open else 'CLOSED'}")
    print(f"  {'=' * 50}")

    return results


def main():
    t0 = time.time()

    # Download data (use cache if available)
    if DATA_PATH.exists():
        print("\n[1/5] Loading cached Nifty 50 data...")
        raw_df = pd.read_parquet(DATA_PATH)
        print(f"  Loaded: {DATA_PATH} ({len(raw_df)} rows)")
        substitutions = []
    else:
        raw_df, substitutions = download_nifty50_data()

    # Integrity checks
    integrity = data_integrity_checks(raw_df)

    # Run pipeline
    daily_ics, fold_results = run_pipeline(raw_df)

    if not daily_ics:
        print("\nERROR: No daily IC values computed. Pipeline failed.")
        return

    # Gate decision
    gate_results = compute_gate_decision(daily_ics)

    # Save results
    gate_df = pd.DataFrame([gate_results])
    gate_df.to_csv(OUT_DIR / "ic_gate_results.csv", index=False)
    print(f"\nGate results saved: {OUT_DIR / 'ic_gate_results.csv'}")

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(OUT_DIR / "fold_results.csv", index=False)
    print(f"Fold results saved: {OUT_DIR / 'fold_results.csv'}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
