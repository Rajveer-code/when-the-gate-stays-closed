"""
calibration.py
==============
Probability calibration and Information Coefficient (IC) testing.

Probability calibration corrects the tendency of tree-based models and
uncalibrated neural networks to produce non-representative probabilities
(e.g., clustered around the mean or exhibiting extreme confidence).
Isotonic regression maps these raw probabilities to true empirical frequencies
on a hold-out calibration set.

The Information Coefficient (IC) test isolates the predictive power
of the cross-sectional ranking.

Author: Rajveer Singh Pall
Paper : "Overcoming the Transaction Cost Trap: Cross-Sectional Conviction
         Ranking in Machine Learning Equity Prediction"
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# CALIBRATION FUNCTIONS
# ---------------------------------------------------------------------------

def fit_calibrator(fitted_model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    """
    Fit an IsotonicRegression calibrator using a held-out calibration set.

    CRITICAL WARNING: Never pass test data to this function. The calibrator
    must be blind to the test distribution. Always use cal_dates from the
    WalkForwardFold — never test_dates.

    Parameters
    ----------
    fitted_model : Any
        A model object exposing a predict_proba(X) -> 1D array method.
    X_cal : np.ndarray, shape (n_cal_rows, n_features)
        Scaled calibration features.
    y_cal : np.ndarray, shape (n_cal_rows,)
        Binary calibration labels.

    Returns
    -------
    IsotonicRegression
        The fitted calibrator object.
    """
    assert len(X_cal) > 0, "Calibration set is empty"
    unique_labels = set(np.unique(y_cal))
    assert unique_labels.issubset({0.0, 1.0}), f"y_cal not binary: found {unique_labels}"
    assert len(X_cal) == len(y_cal), "X_cal / y_cal length mismatch"

    raw_probs = fitted_model.predict_proba(X_cal)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibrator.fit(raw_probs, y_cal)

    return calibrator


def calibrated_predict(fitted_model: Any, calibrator: IsotonicRegression, X: np.ndarray) -> np.ndarray:
    """
    Predict calibrated probabilities.

    Parameters
    ----------
    fitted_model : Any
        A fitted base model exposing predict_proba(X) -> 1D array method.
    calibrator : IsotonicRegression
        A fitted calibrator from fit_calibrator().
    X : np.ndarray, shape (n_rows, n_features)
        Scaled features to predict.

    Returns
    -------
    np.ndarray, shape (n_rows,)
        Calibrated probabilities in [0.0, 1.0].
    """
    raw_probs = fitted_model.predict_proba(X)
    cal_probs = calibrator.predict(raw_probs)
    cal_probs = np.clip(cal_probs, 0.0, 1.0)

    assert cal_probs.ndim == 1, f"Calibrated probs must be 1D, got {cal_probs.shape}"
    assert cal_probs.shape == (len(X),), f"Output shape {cal_probs.shape} != ({len(X)},)"
    assert (cal_probs >= 0.0).all() and (cal_probs <= 1.0).all(), "Probs out of bounds [0,1]"

    return cal_probs


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Parameters
    ----------
    probs : np.ndarray, shape (n,)
        Predicted probabilities.
    labels : np.ndarray, shape (n,)
        Binary target labels in {0.0, 1.0}.
    n_bins : int, default=10
        Number of equal-width probability bins [0, 1].

    Returns
    -------
    float
        ECE score. Lower is better.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # digitize returns 1-indexed bins
    bin_indices = np.digitize(probs, bin_edges, right=True)

    ece = 0.0
    n_total = len(probs)

    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if not np.any(mask):
            continue

        bin_probs = probs[mask]
        bin_labels = labels[mask]

        confidence = np.mean(bin_probs)
        accuracy = np.mean(bin_labels)
        weight = len(bin_probs) / n_total

        ece += weight * np.abs(accuracy - confidence)

    return float(ece)


def plot_reliability_diagram(
    fitted_model: Any,
    calibrator: IsotonicRegression,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    model_name: str,
    save_path: Path,
) -> float:
    """
    Plot and save a reliability diagram showing uncalibrated vs calibrated curves.

    Parameters
    ----------
    fitted_model : Any
    calibrator : IsotonicRegression
    X_cal : np.ndarray
    y_cal : np.ndarray
    model_name : str
    save_path : Path
        File path where the PNG will be written.

    Returns
    -------
    float
        ECE of the CALIBRATED probabilities.
    """
    raw_probs = fitted_model.predict_proba(X_cal)
    cal_probs = calibrated_predict(fitted_model, calibrator, X_cal)

    ece_raw = compute_ece(raw_probs, y_cal)
    ece_cal = compute_ece(cal_probs, y_cal)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Bin raw probs
    raw_bin_idx = np.digitize(raw_probs, bin_edges, right=True)
    raw_mean_pred = []
    raw_frac_pos = []
    for i in range(1, n_bins + 1):
        mask = raw_bin_idx == i
        if np.any(mask):
            raw_mean_pred.append(np.mean(raw_probs[mask]))
            raw_frac_pos.append(np.mean(y_cal[mask]))

    # Bin cal probs
    cal_bin_idx = np.digitize(cal_probs, bin_edges, right=True)
    cal_mean_pred = []
    cal_frac_pos = []
    for i in range(1, n_bins + 1):
        mask = cal_bin_idx == i
        if np.any(mask):
            cal_mean_pred.append(np.mean(cal_probs[mask]))
            cal_frac_pos.append(np.mean(y_cal[mask]))

    plt.figure(figsize=(8, 8))
    plt.plot(raw_mean_pred, raw_frac_pos, "r--", marker="o", label=f"Uncalibrated (ECE={ece_raw:.4f})")
    plt.plot(cal_mean_pred, cal_frac_pos, "b-", marker="s", label=f"Calibrated   (ECE={ece_cal:.4f})")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram - {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"  ECE uncalibrated: {ece_raw:.4f} | ECE calibrated: {ece_cal:.4f}")
    return ece_cal


def run_calibration_pipeline(
    fold: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    fitted_model: Any,
    scaler: Any,
    model_name: str,
    plots_dir: Path,
) -> Tuple[IsotonicRegression, float]:
    """
    Main entry point for calibration, called from the trading loop.

    Parameters
    ----------
    fold : WalkForwardFold
    df : pd.DataFrame
    feature_cols : list of str
    fitted_model : Any
    scaler : StandardScaler
        Must be pre-fitted on fold.model_train_dates.
    model_name : str
    plots_dir : Path

    Returns
    -------
    calibrator : IsotonicRegression
    ece : float
        ECE of the calibrated predictions.
    """
    # Import locally to avoid circular dependencies if placed in ___init__
    from src.training.walk_forward import get_cal_arrays

    X_cal, y_cal = get_cal_arrays(fold, df, feature_cols, scaler)

    overlap = set(fold.cal_dates) & set(fold.test_dates)
    assert not overlap, "Calibration set overlaps test set - data leakage"

    calibrator = fit_calibrator(fitted_model, X_cal, y_cal)

    save_path = plots_dir / f"{model_name}_fold{fold.fold_number:02d}_reliability.png"
    ece = plot_reliability_diagram(
        fitted_model, calibrator, X_cal, y_cal, model_name, save_path
    )

    print(
        f"  Fold {fold.fold_number} | {model_name} | Cal size: {len(y_cal)} | "
        f"ECE: {ece:.4f}"
    )

    return calibrator, ece


# ---------------------------------------------------------------------------
# INFORMATION COEFFICIENT (IC)
# ---------------------------------------------------------------------------

def compute_spearman_ic(probabilities: np.ndarray, realized_returns: np.ndarray) -> float:
    """
    Compute daily cross-sectional Spearman Rank Correlation (IC).

    Parameters
    ----------
    probabilities : np.ndarray, shape (n_tickers,)
        Predicted cross-sectional probabilities on day t.
    realized_returns : np.ndarray, shape (n_tickers,)
        Realized holding returns Close(t+2)/Close(t+1) - 1.

    Returns
    -------
    float
        Correlation in [-1.0, 1.0]. Returns 0.0 if variance is zero.
    """
    # Guard against zero variance in predictions (e.g. all 0.5)
    if np.var(probabilities) < 1e-12 or np.var(realized_returns) < 1e-12:
        return 0.0

    try:
        corr, _ = scipy.stats.spearmanr(probabilities, realized_returns)
        # NaN can happen in scipy if variance is technically > 0 but effectively 0
        if np.isnan(corr):
            return 0.0
        return float(corr)
    except Exception:
        return 0.0


def test_ic_significance(daily_ic_values: list[float]) -> Dict[str, Any]:
    """
    Test whether out-of-sample IC is significantly > 0.

    Parameters
    ----------
    daily_ic_values : list of float
        Accumulated daily cross-sectional Spearman correlations.

    Returns
    -------
    dict
        IC metrics and significance flag.
    """
    n_days = len(daily_ic_values)
    mean_ic = float(np.mean(daily_ic_values))
    ic_std = float(np.std(daily_ic_values, ddof=1))
    icir = mean_ic / ic_std if ic_std > 0 else 0.0

    # 1-sample t-test: H0: mean_ic <= 0, H1: mean_ic > 0 (upper-tail, one-sided)
    # NOTE: the main ICGDF gate uses the HAC Newey-West t-stat; this naive t-test
    # is a quick diagnostic summary only.  See run_experiments.py for gate logic.
    t_stat, _ = scipy.stats.ttest_1samp(daily_ic_values, 0)
    # Upper-tail p-value: P(T > t_stat | H0) — correct for H1: IC > 0
    p_value = float(scipy.stats.t.sf(t_stat, df=n_days - 1))

    significant = bool(p_value < 0.05 and mean_ic > 0)

    print("=" * 60)
    print("IC SIGNIFICANCE TEST")
    print("=" * 60)
    print(f"Mean IC  : {mean_ic:.4f}")
    print(f"IC Std   : {ic_std:.4f}")
    print(f"ICIR     : {icir:.4f}")
    print(f"T-stat   : {t_stat:.4f} (df = {n_days-1})")
    print(f"P-value  : {p_value:.4f} (one-tailed, H1: IC > 0)")
    print(f"N days   : {n_days}")

    if significant:
        print("Result   : [PASS] IC significantly > 0 -- cross-sectional signal confirmed")
    else:
        print("Result   : [FAIL] IC not significant -- consider repositioning paper as negative result study")
        print("\n" + "=" * 60)
        print("WARNING: Mean IC is not significantly positive.")
        print("The cross-sectional ranking contribution may be invalid.")
        print("Review feature engineering and calibration before backtesting.")
        print("=" * 60)

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_days": n_days,
        "significant": significant,
    }


# ---------------------------------------------------------------------------
# MODULE SELF-TEST (synthetic data)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    import tempfile
    import pathlib

    print("Running calibration.py self-test...")

    rng = np.random.default_rng(42)
    N = 300
    X_tr = rng.standard_normal((N, 10))
    y_tr = (X_tr[:, 0] + rng.standard_normal(N) * 0.5 > 0).astype(float)
    X_cal = rng.standard_normal((100, 10))
    y_cal_true = (X_cal[:, 0] + rng.standard_normal(100) * 0.5 > 0).astype(float)

    rf = RandomForestClassifier(n_estimators=50, random_state=0)
    rf.fit(X_tr, y_tr)

    class _Wrapper:
        def __init__(self, m):
            self.m = m
        def predict_proba(self, X):
            return self.m.predict_proba(X)[:, 1]

    wrapper = _Wrapper(rf)

    # Test 1
    calibrator = fit_calibrator(wrapper, X_cal, y_cal_true)
    assert hasattr(calibrator, "predict"), "Calibrator missing predict method"
    print("[OK] fit_calibrator")

    # Test 2
    probs = calibrated_predict(wrapper, calibrator, X_cal)
    assert probs.shape == (100,)
    assert 0.0 <= probs.min() and probs.max() <= 1.0
    print("[OK] calibrated_predict")

    # Test 3
    ece_raw = compute_ece(wrapper.predict_proba(X_cal), y_cal_true)
    ece_cal = compute_ece(probs, y_cal_true)
    assert 0.0 <= ece_raw <= 1.0
    assert 0.0 <= ece_cal <= 1.0
    print(f"[OK] compute_ece -- raw: {ece_raw:.4f} | calibrated: {ece_cal:.4f}")

    # Test 4
    with tempfile.TemporaryDirectory() as tmp:
        save_path = pathlib.Path(tmp) / "test_reliability.png"
        ece = plot_reliability_diagram(
            wrapper, calibrator, X_cal, y_cal_true, "TestModel", save_path
        )
        assert save_path.exists(), "PNG not saved"
        assert isinstance(ece, float)
    print("[OK] plot_reliability_diagram -- PNG saved and ECE returned")

    # Test 5
    probs_7 = rng.uniform(0, 1, 7)
    returns_7 = rng.standard_normal(7)
    ic = compute_spearman_ic(probs_7, returns_7)
    assert -1.0 <= ic <= 1.0
    ic_zero = compute_spearman_ic(np.ones(7), returns_7)
    assert ic_zero == 0.0, "Zero-variance case should return 0.0"
    print(f"[OK] compute_spearman_ic -- IC: {ic:.4f}, zero-var: {ic_zero}")

    # Test 6
    ic_series_sig = list(rng.normal(0.04, 0.08, 500))
    result_sig = test_ic_significance(ic_series_sig)
    assert result_sig["significant"] is True
    print("[OK] test_ic_significance -- significant case detected")

    # Test 7
    ic_series_flat = list(rng.normal(0.0, 0.1, 500))
    result_flat = test_ic_significance(ic_series_flat)
    print(f"[OK] test_ic_significance -- non-significant case: p={result_flat['p_value']:.4f}")

    print("\n[PASS] calibration.py PASSED: calibrator OK, ECE OK, reliability diagram OK, IC test OK")
