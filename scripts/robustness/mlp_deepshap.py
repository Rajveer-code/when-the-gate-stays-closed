#!/usr/bin/env python3
"""
mlp_deepshap.py — Task 3: MLP Feature Attribution
==================================================
Retrain MLP for folds 9-12, compute feature attribution via Captum
IntegratedGradients (with DeepExplainer/GradientExplainer fallbacks),
compare with existing TreeSHAP rankings.

Output:
  results/robustness/shap/mlp_attribution_results.csv
  results/robustness/shap/mlp_tree_rank_correlation.csv
  results/models/fold_{k}_mlp.pt

Author: Rajveer Singh Pall
"""

from __future__ import annotations
import os, sys, warnings, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats

# Resolve repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.data_loader import load_all_data, get_feature_columns
from src.training.walk_forward import generate_folds, get_fold_arrays, get_cal_arrays
from src.training.models import DNNModel

import torch
import torch.nn as nn

print("=" * 70)
print("TASK 3: MLP Feature Attribution (folds 9-12)")
print("=" * 70)
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:  {torch.cuda.get_device_name(0)}")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH = _ROOT / "data" / "nasdaq30_prices.parquet"
SHAP_DIR  = _ROOT / "results" / "robustness" / "shap"
MODEL_DIR = _ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TREE_SHAP_CSV = SHAP_DIR / "shap_mean_abs_by_fold.csv"
MLP_OUT_CSV   = SHAP_DIR / "mlp_attribution_results.csv"
CORR_OUT_CSV  = SHAP_DIR / "mlp_tree_rank_correlation.csv"

N_BACKGROUND = 50
N_TEST_SAMPLES = 200
TARGET_FOLDS = [9, 10, 11, 12]  # 1-indexed fold numbers
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def compute_captum_ig(model, X_test, X_background, feature_names):
    """Compute Integrated Gradients attribution using Captum."""
    from captum.attr import IntegratedGradients

    model.eval()
    device = next(model.parameters()).device

    # Use mean of background as baseline
    baseline = torch.from_numpy(X_background.mean(axis=0, keepdims=True).astype(np.float32)).to(device)
    X_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    X_t.requires_grad = True

    ig = IntegratedGradients(model)

    # Compute in batches to avoid OOM
    all_attrs = []
    batch_size = 50
    for i in range(0, len(X_t), batch_size):
        batch = X_t[i:i+batch_size]
        baseline_expanded = baseline.expand(batch.shape[0], -1)
        attr = ig.attribute(batch, baselines=baseline_expanded, n_steps=50)
        all_attrs.append(attr.detach().cpu().numpy())

    attrs = np.concatenate(all_attrs, axis=0)
    mean_abs = np.abs(attrs).mean(axis=0)

    result = pd.Series(mean_abs, index=feature_names)
    return result, "captum_ig"


def try_shap_deep(model, X_test, X_background, feature_names):
    """Try SHAP DeepExplainer first, then GradientExplainer."""
    import shap

    device = next(model.parameters()).device
    bg_t = torch.from_numpy(X_background.astype(np.float32)).to(device)
    test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)

    # Try DeepExplainer
    try:
        print("    Trying shap.DeepExplainer...")
        explainer = shap.DeepExplainer(model, bg_t)
        shap_vals = explainer.shap_values(test_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if isinstance(shap_vals, torch.Tensor):
            shap_vals = shap_vals.cpu().numpy()
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return pd.Series(mean_abs, index=feature_names), "deep_shap"
    except Exception as e:
        print(f"    DeepExplainer failed: {e}")

    # Try GradientExplainer
    try:
        print("    Trying shap.GradientExplainer...")
        explainer = shap.GradientExplainer(model, bg_t)
        shap_vals = explainer.shap_values(test_t)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if isinstance(shap_vals, torch.Tensor):
            shap_vals = shap_vals.cpu().numpy()
        mean_abs = np.abs(shap_vals).mean(axis=0)
        return pd.Series(mean_abs, index=feature_names), "gradient_shap"
    except Exception as e:
        print(f"    GradientExplainer failed: {e}")

    return None, None


def main():
    t0 = time.time()

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    df = load_all_data(external_data_path=str(DATA_PATH))
    feature_cols = get_feature_columns(df)
    print(f"  Data shape: {df.shape}, features: {len(feature_cols)}")

    # ── Generate folds ───────────────────────────────────────────────────────
    print("\n[2/4] Generating walk-forward folds...")
    folds = generate_folds(df)
    print(f"  Total folds: {len(folds)}")

    # ── Load existing TreeSHAP ───────────────────────────────────────────────
    print("\n[3/4] Loading existing TreeSHAP results...")
    tree_shap_df = pd.read_csv(TREE_SHAP_CSV, index_col=0)
    print(f"  TreeSHAP features: {len(tree_shap_df)}")

    # ── Retrain MLP + compute attribution for folds 9-12 ─────────────────────
    print("\n[4/4] Retraining MLP and computing attribution for folds 9-12...")

    mlp_results = {}
    method_used = None

    for fold_idx in TARGET_FOLDS:
        fold = folds[fold_idx - 1]  # 0-indexed
        print(f"\n  === Fold {fold_idx} ===")

        # Get arrays
        X_train, X_test, y_train, y_test, scaler = get_fold_arrays(
            fold, df, feature_cols, scaler=None
        )
        X_cal, y_cal = get_cal_arrays(fold, df, feature_cols, scaler)

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}, Cal: {X_cal.shape}")

        # Check for saved weights
        weight_path = MODEL_DIR / f"fold_{fold_idx}_mlp.pt"

        # Train MLP
        print(f"  Training MLP...")
        n_features = X_train.shape[1]
        mlp = DNNModel(in_features=n_features, random_seed=SEED)
        mlp.fit(X_train, y_train)

        # Save weights
        torch.save(mlp.model.state_dict(), weight_path)
        print(f"  Saved: {weight_path}")

        # Sample test data
        n_test = min(N_TEST_SAMPLES, len(X_test))
        rng = np.random.RandomState(SEED + fold_idx)
        test_idx = rng.choice(len(X_test), size=n_test, replace=False)
        X_test_sample = X_test[test_idx]

        # Sample background from calibration
        n_bg = min(N_BACKGROUND, len(X_cal))
        bg_idx = rng.choice(len(X_cal), size=n_bg, replace=False)
        X_bg = X_cal[bg_idx]

        print(f"  Test sample: {n_test}, Background: {n_bg}")

        # Try SHAP first, fallback to Captum IG
        result, method = try_shap_deep(
            mlp.model, X_test_sample, X_bg, feature_cols
        )

        if result is None:
            print("    Falling back to Captum IntegratedGradients...")
            result, method = compute_captum_ig(
                mlp.model, X_test_sample, X_bg, feature_cols
            )

        mlp_results[f"fold_{fold_idx}"] = result
        method_used = method
        print(f"  Method: {method}")
        print(f"  Top-5: {result.nlargest(5).to_dict()}")

    # ── Build results DataFrame ──────────────────────────────────────────────
    mlp_df = pd.DataFrame(mlp_results)
    mlp_df["mean_across_folds"] = mlp_df.mean(axis=1)
    mlp_df = mlp_df.sort_values("mean_across_folds", ascending=False)
    mlp_df.to_csv(MLP_OUT_CSV)
    print(f"\nMLP attribution saved: {MLP_OUT_CSV}")

    # ── Compute Spearman rank correlation with TreeSHAP ──────────────────────
    print("\n[Spearman Rank Correlation: MLP vs Tree]")
    corr_rows = []

    for fold_idx in TARGET_FOLDS:
        fold_col = f"fold_{fold_idx}"

        # Get MLP and Tree rankings for this fold
        mlp_vals = mlp_df[fold_col]
        tree_vals = tree_shap_df[fold_col]

        # Align features
        common_features = mlp_vals.index.intersection(tree_vals.index)
        mlp_aligned = mlp_vals.loc[common_features]
        tree_aligned = tree_vals.loc[common_features]

        # Rank
        mlp_rank = mlp_aligned.rank(ascending=False)
        tree_rank = tree_aligned.rank(ascending=False)

        # Spearman
        rho, pval = scipy.stats.spearmanr(mlp_rank, tree_rank)
        corr_rows.append({
            "fold": fold_idx,
            "spearman_rho": rho,
            "p_value": pval,
            "n_features": len(common_features),
        })
        print(f"  Fold {fold_idx}: rho = {rho:.4f} (p = {pval:.4f})")

    corr_df = pd.DataFrame(corr_rows)
    mean_rho = corr_df["spearman_rho"].mean()
    corr_df.loc[len(corr_df)] = {
        "fold": "mean",
        "spearman_rho": mean_rho,
        "p_value": np.nan,
        "n_features": corr_df["n_features"].iloc[0],
    }
    corr_df.to_csv(CORR_OUT_CSV, index=False)
    print(f"\nMean rho: {mean_rho:.4f}")
    print(f"Correlation saved: {CORR_OUT_CSV}")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"TASK 3 COMPLETE")
    print(f"  Method used: {method_used}")
    print(f"  Mean Spearman rho (MLP vs Tree): {mean_rho:.4f}")
    print(f"  Top-5 MLP features (mean):")
    top5 = mlp_df["mean_across_folds"].nlargest(5)
    for feat, val in top5.items():
        print(f"    {feat}: {val:.6f}")

    # Interpretation
    if mean_rho > 0.5:
        print(f"\n  INTERPRETATION: rho = {mean_rho:.3f} > 0.5")
        print("  MLP and tree components identify similar feature classes,")
        print("  consistent with ensemble-wide noise-fitting.")
    elif mean_rho < 0.3:
        print(f"\n  INTERPRETATION: rho = {mean_rho:.3f} < 0.3")
        print("  MLP exploits different feature combinations than tree components,")
        print("  consistent with architecturally diverse noise-fitting.")
    else:
        print(f"\n  INTERPRETATION: 0.3 <= rho = {mean_rho:.3f} <= 0.5")
        print("  Moderate agreement; neither confirms nor rules out")
        print("  architecture-specific artifacts.")

    print(f"\n  Runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
