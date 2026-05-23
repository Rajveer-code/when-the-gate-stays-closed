"""
run_experiments.py
==================
Main pipeline entry point for the ICGDF paper.

To reproduce paper results (30-stock NASDAQ-100 universe):
    python scripts/run_experiments.py --data-path data/nasdaq30_prices.parquet

Without --data-path, the script falls back to downloading 7-stock data
from Yahoo Finance (development / quick-check mode only).

Flags:
  --data-path PATH         External OHLCV parquet (MultiIndex date x ticker)
  --model {ensemble,catboost,rf}
  --uncalibrated
  --exclude-ticker TICKER

Author: Rajveer Singh Pall
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import scipy.stats
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import load_all_data, get_feature_columns
from src.training.walk_forward import (
    WalkForwardFold, generate_folds, get_fold_arrays, get_cal_arrays, print_fold_summary
)
from src.training.models import CatBoostModel, RandomForestModel, EnsembleModel
from src.training.calibration import (
    fit_calibrator, calibrated_predict, compute_spearman_ic,
    test_ic_significance, plot_reliability_diagram
)
from src.backtesting.backtester import (
    STRATEGY_CONFIGS, run_backtest, run_all_strategies,
    run_subperiod_analysis, run_cost_sensitivity, run_spy_buyhold
)

OUTPUT_DIR = Path("results")


# ---------------------------------------------------------------------------
# MODEL FACTORY
# ---------------------------------------------------------------------------

def make_model(model_type: str) -> Any:
    if model_type == "catboost":
        print("  [MODEL] Using CatBoostModel (solo)")
        return CatBoostModel()
    elif model_type == "rf":
        print("  [MODEL] Using RandomForestModel (solo)")
        return RandomForestModel()
    else:
        print("  [MODEL] Using EnsembleModel (CatBoost + RF [+ DNN if available])")
        return EnsembleModel()


def get_checkpoint_suffix(model_type: str, use_calibration: bool,
                          exclude_ticker: Optional[str] = None) -> str:
    """
    Build checkpoint filename suffix so no run ever overwrites another.

    Examples:
      ensemble, calibrated, no exclusion  ->  ''               (fold_01.parquet)
      catboost, calibrated, no exclusion  ->  '_catboost'
      rf,       calibrated, no exclusion  ->  '_rf'
      ensemble, uncalibrated              ->  '_uncal'
      ensemble, calibrated, NVDA excluded ->  '_nonvda'
      catboost, uncalibrated              ->  '_catboost_uncal'
    """
    suffix = "" if model_type == "ensemble" else f"_{model_type}"
    if not use_calibration:
        suffix += "_uncal"
    if exclude_ticker:
        suffix += f"_no{exclude_ticker.lower()}"
    return suffix


# ---------------------------------------------------------------------------
# ORCHESTRATION PIPELINE
# ---------------------------------------------------------------------------

def build_predictions_df(
    fold: WalkForwardFold,
    df: pd.DataFrame,
    actual_returns_series: pd.Series,
    feature_cols: List[str],
    fitted_model: Any,
    calibrator,
    scaler: StandardScaler,
    model_name: str,
    use_calibration: bool = True,
) -> pd.DataFrame:
    dates_level = df.index.get_level_values('date')
    test_mask = dates_level.isin(set(fold.test_dates))
    test_df = df.loc[test_mask].copy()

    assert hasattr(scaler, 'mean_') and scaler.mean_ is not None, "Scaler unfitted"
    X_test = scaler.transform(test_df[feature_cols].values)

    if use_calibration and calibrator is not None:
        probs = calibrated_predict(fitted_model, calibrator, X_test)
    else:
        probs = fitted_model.predict_proba(X_test)
        probs = np.clip(probs, 0.0, 1.0)

    assert probs.shape == (len(test_df),), "Shape misalignment"

    result = pd.DataFrame({
        'prob': probs,
        'Close': test_df['Close'].values,
        'SMA_200': test_df['SMA_200'].values,
        'target': test_df['target'].values,
        'trailing_return_21d': test_df['return_21d'].values,
    }, index=test_df.index)

    aligned_actuals = actual_returns_series.loc[test_df.index]
    result['actual_return'] = aligned_actuals.values
    result['model'] = model_name
    result = result.dropna(subset=['actual_return'])

    assert not result[['prob', 'Close', 'SMA_200', 'target']].isna().any().any(), \
        "Structural NaN in metrics"

    return result


def run_walk_forward_loop(
    df: pd.DataFrame,
    feature_cols: List[str],
    folds: List[WalkForwardFold],
    use_ensemble: bool = True,
    model_type: str = "ensemble",
    use_calibration: bool = True,
    exclude_ticker: Optional[str] = None,          # ← FIXED: now a proper parameter
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:

    all_predictions = []
    all_fold_ic = []

    actual_returns_dict = {}
    tickers = df.index.get_level_values('ticker').unique()
    for ticker in tickers:
        close = df.xs(ticker, level='ticker')['Close']
        actual_returns_dict[ticker] = close.shift(-2) / close.shift(-1) - 1.0

    actual_returns_df = pd.concat(
        {t: s for t, s in actual_returns_dict.items()}, axis=1
    ).stack()
    actual_returns_df.index.names = ['date', 'ticker']

    # ── Checkpoint suffix — unique per run, no overwriting ──────────────
    ckpt_suffix = get_checkpoint_suffix(model_type, use_calibration, exclude_ticker)

    print("\n" + "=" * 50)
    print(f"Pipeline | model={model_type} | calibration={use_calibration} | "
          f"exclude={exclude_ticker} | suffix='{ckpt_suffix}'")
    print("=" * 50)

    plots_dir = OUTPUT_DIR / "plots" / "reliability_diagrams"
    preds_dir = OUTPUT_DIR / "predictions"

    for fold in folds:
        print("\n" + "=" * 50)
        print(f"FOLD {fold.fold_number}/{len(folds)}")
        print(f"  Train: {fold.train_start.date()} to {fold.train_end.date()}")
        print(f"  Test : {fold.test_start.date()} to {fold.test_end.date()}")
        print(f"  Train rows: {len(fold.model_train_dates) * len(tickers)}")
        print(f"  Test rows : {len(fold.test_dates) * len(tickers)}")

        checkpoint_path = preds_dir / f"fold_{fold.fold_number:02d}{ckpt_suffix}.parquet"
        if checkpoint_path.exists():
            print(f"  [CHECKPOINT] Fold {fold.fold_number} loaded ({checkpoint_path.name})")
            fold_preds = pd.read_parquet(checkpoint_path)
            all_predictions.append(fold_preds)
            for date in fold_preds.index.get_level_values('date').unique():
                day_slice = fold_preds.loc[date]
                if (np.var(day_slice['prob'].values) > 1e-12 and
                        np.var(day_slice['actual_return'].values) > 1e-12):
                    corr, _ = spearmanr(
                        day_slice['prob'].values, day_slice['actual_return'].values
                    )
                    all_fold_ic.append(0.0 if np.isnan(corr) else float(corr))
                else:
                    all_fold_ic.append(0.0)
            continue

        X_train, X_test, y_train, y_test, scaler = get_fold_arrays(fold, df, feature_cols)
        print(f"  [1/5] Arrays extracted. Train={X_train.shape}")

        model = make_model(model_type)
        model.fit(X_train, y_train)
        print(f"  [2/5] Model trained.")

        calibrator = None
        if use_calibration:
            X_cal, y_cal = get_cal_arrays(fold, df, feature_cols, scaler)
            calibrator = fit_calibrator(model, X_cal, y_cal)
            model_label = model_type.capitalize()
            plot_reliability_diagram(
                model, calibrator, X_cal, y_cal,
                f"{model_label}_Fold{fold.fold_number:02d}",
                plots_dir / f"{model_label}_fold{fold.fold_number:02d}{ckpt_suffix}.png"
            )
            print(f"  [3/5] Calibration complete.")
        else:
            print(f"  [3/5] Calibration SKIPPED (--uncalibrated).")

        model_label = model_type.capitalize()
        if not use_calibration:
            model_label += "_Uncal"

        fold_preds = build_predictions_df(
            fold, df, actual_returns_df, feature_cols,
            model, calibrator, scaler,
            model_name=model_label,
            use_calibration=use_calibration,
        )
        print(f"  [4/5] Predictions built: {len(fold_preds)} rows.")

        fold_ic_values = []
        for date, day_slice in fold_preds.groupby(level='date'):
            ic = compute_spearman_ic(
                day_slice['prob'].values, day_slice['actual_return'].values
            )
            fold_ic_values.append(ic)

        f_mean = np.mean(fold_ic_values)
        f_std = np.std(fold_ic_values) if np.std(fold_ic_values) > 0 else 1.0
        print(f"  [5/5] Fold IC: mean={f_mean:.4f}, ICIR={f_mean/f_std:.4f}")

        all_predictions.append(fold_preds)
        all_fold_ic.extend(fold_ic_values)
        fold_preds.to_parquet(checkpoint_path)

    print("\nAggregating predictions...")
    predictions_df = pd.concat(all_predictions).sort_index()
    assert not predictions_df.index.duplicated().any(), "Duplicate index"

    ic_results = test_ic_significance(all_fold_ic)
    pd.DataFrame([ic_results]).to_csv(
        OUTPUT_DIR / "metrics" / "ic_test_results.csv", index=False
    )

    base_file_name = f"predictions_{model_type}{ckpt_suffix}.parquet"
    predictions_df.to_parquet(preds_dir / base_file_name)

    print("-" * 60)
    print(f"Walk-forward complete.")
    print(f"Total OOS rows : {len(predictions_df)}")
    print(f"Overall mean IC: {np.mean(all_fold_ic):.4f}")
    print(f"Saved to: {preds_dir / base_file_name}")

    return predictions_df, pd.DataFrame([ic_results]), all_fold_ic


# ---------------------------------------------------------------------------
# SPY
# ---------------------------------------------------------------------------

def load_spy_returns(start: str, end: str) -> pd.Series:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "SPY_ohlcv.parquet"

    if cache_path.exists():
        spy_df = pd.read_parquet(cache_path)
    else:
        spy_df = yf.download("SPY", start=start, end=end)
        spy_df.to_parquet(cache_path)

    if "Close" in spy_df.columns:
        close_col = (spy_df['Close']['SPY']
                     if isinstance(spy_df.columns, pd.MultiIndex)
                     else spy_df['Close'])
    else:
        close_col = spy_df.iloc[:, 0]

    spy_returns = close_col.pct_change(1).dropna()
    spy_returns.index = spy_returns.index.tz_localize(None)
    spy_returns.name = 'SPY'
    return spy_returns


# ---------------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------------

def run_all_experiments(predictions_df: pd.DataFrame, spy_returns: pd.Series,
                        model_type: str = "ensemble",
                        exclude_ticker: Optional[str] = None) -> None:

    metrics_dir = OUTPUT_DIR / "metrics"
    perm_dir = OUTPUT_DIR / "permutation"

    sfx = f"_{model_type}" if model_type != "ensemble" else ""
    if exclude_ticker:
        sfx += f"_no{exclude_ticker.lower()}"

    # Experiment 1 — Strategy Comparison
    print("\nRunning Experiment 1: Strategy Comparison...")
    results_df = run_all_strategies(predictions_df, spy_returns)
    results_df.to_csv(metrics_dir / f"strategy_comparison{sfx}.csv", index=False)

    topk1_row = results_df[results_df['strategy_name'] == 'TopK1']
    if len(topk1_row) > 0:
        sharpe = topk1_row.iloc[0]['sharpe_ratio']
        ann_ret = topk1_row.iloc[0]['annual_return']
        print("\n" + "=" * 60)
        print(f"  *** TopK1 Sharpe  = {sharpe:.6f}  ***")
        print(f"  *** TopK1 AnnRet  = {ann_ret:.6f}  ***")
        label = exclude_ticker if exclude_ticker else model_type
        print(f"  (Record this — run={label})")
        print("=" * 60)

    print("\nTop 3 by Sharpe:")
    print(results_df[['strategy_name', 'annual_return', 'sharpe_ratio']].head(3).to_string(index=False))

    # Experiment 2 — Permutation test (main ensemble only, no exclusions)
    if model_type == "ensemble" and not exclude_ticker:
        print("\nRunning Experiment 2: Permutation Test (parallel)...")
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from parallel_permutation import run_parallel_permutation_test
            topk1_config = [c for c in STRATEGY_CONFIGS if c.name == 'TopK1'][0]
            perm_results = run_parallel_permutation_test(
                predictions_df, topk1_config,
                n_permutations=1000, n_jobs=12,
            )
            pd.Series(perm_results['null_sharpes'], name='null_sharpe').to_csv(
                perm_dir / "permutation_topk1.csv", index=False
            )
            sum_dict = {k: v for k, v in perm_results.items() if k != 'null_sharpes'}
            pd.DataFrame([sum_dict]).to_csv(
                perm_dir / "permutation_topk1_summary.csv", index=False
            )
        except Exception as e:
            print(f"  [WARN] Permutation test failed: {e}")

    # Experiment 3 — Sub-period (main ensemble only)
    if model_type == "ensemble" and not exclude_ticker:
        print("\nRunning Experiment 3: Sub-period Analysis...")
        key_configs = [c for c in STRATEGY_CONFIGS
                       if c.name in ['TopK1', 'TopK1_Trend', 'Equal_Weight', 'Random_Top1']]
        sub_df = run_subperiod_analysis(predictions_df, key_configs)

        periods = {
            'Period 1 - ZIRP Bull':    ('2015-10-16', '2018-12-31'),
            'Period 2 - COVID/Growth': ('2019-01-01', '2021-12-31'),
            'Period 3 - Rate Shock':   ('2022-01-01', '2024-12-31'),
        }
        spy_recs = []
        spy_cfg = [c for c in STRATEGY_CONFIGS if c.name == 'BuyHold_SPY'][0]
        for p_name, (s, e) in periods.items():
            mask = (
                (predictions_df.index.get_level_values('date') >= pd.to_datetime(s)) &
                (predictions_df.index.get_level_values('date') <= pd.to_datetime(e))
            )
            dates_in = predictions_df.loc[mask].index.get_level_values('date').unique().tolist()
            if dates_in:
                res = run_spy_buyhold(spy_returns, dates_in)
                spy_recs.append({
                    'strategy_name': spy_cfg.name, 'sub_period': p_name,
                    'annual_return': res['annual_return'],
                    'sharpe_ratio': res['sharpe_ratio'],
                    'max_drawdown': res['max_drawdown'],
                })
        if spy_recs:
            sub_df = pd.concat([
                sub_df,
                pd.DataFrame(spy_recs).set_index(['strategy_name', 'sub_period'])
            ])

        sub_df.to_csv(metrics_dir / "subperiod_analysis.csv")
        print(sub_df.to_string())

        try:
            topk1_sub = sub_df.xs('TopK1', level='strategy_name')
            print("\n" + "=" * 60)
            print("TopK1 sub-period Sharpes (for paper A4):")
            print(topk1_sub[['sharpe_ratio']].to_string())
            print("=" * 60)
        except Exception:
            pass

    # Experiment 4 — Cost Sensitivity
    print("\nRunning Experiment 4: Cost Sensitivity...")
    try:
        topk1_cfg = [c for c in STRATEGY_CONFIGS if c.name == 'TopK1'][0]
        cost_df = run_cost_sensitivity(
            predictions_df, topk1_cfg,
            cost_levels_bps=[0, 5, 10, 15, 20, 30, 50]
        )
        cost_df.to_csv(metrics_dir / f"cost_sensitivity_topk1{sfx}.csv")
        print(cost_df[['cost_bps', 'annual_return', 'sharpe_ratio']].to_string())
    except Exception as e:
        print(f"  [WARN] Cost Sensitivity failed: {e}")

    # Experiment 5 — K Sensitivity (main ensemble only)
    if model_type == "ensemble" and not exclude_ticker:
        print("\nRunning Experiment 5: K Sensitivity...")
        k_configs = [c for c in STRATEGY_CONFIGS if c.name in ['TopK1', 'TopK2', 'TopK3']]
        k_results = [run_backtest(predictions_df, c) for c in k_configs]
        k_df = pd.DataFrame(k_results)[
            ['strategy_name', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'n_trades', 'mean_ic']
        ]
        k_df.to_csv(metrics_dir / "k_sensitivity.csv", index=False)
        print("\n" + k_df.to_string(index=False))

    print("\n" + "=" * 60)
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Results saved with suffix: '{sfx}'")
    print("=" * 60)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end experiments.")
    parser.add_argument("--data-path",  type=str, default=None)
    parser.add_argument("--start",      type=str, default=None)
    parser.add_argument("--end",        type=str, default=None)
    parser.add_argument("--use-cache",  action="store_true", default=False)
    parser.add_argument(
        "--model", type=str, default="ensemble",
        choices=["ensemble", "catboost", "rf"],
    )
    parser.add_argument("--uncalibrated", action="store_true", default=False)
    parser.add_argument("--exclude-ticker", type=str, default=None, metavar="TICKER")
    args = parser.parse_args()

    print("=" * 60)
    print("RUN CONFIGURATION")
    print(f"  --model          : {args.model}")
    print(f"  --uncalibrated   : {args.uncalibrated}")
    print(f"  --exclude-ticker : {args.exclude_ticker}")
    print("=" * 60)

    if args.model == "catboost" and not args.uncalibrated:
        print(">> Computes: A2_cb (CatBoost solo TopK1 Sharpe)")
    elif args.model == "rf" and not args.uncalibrated:
        print(">> Computes: A2_rf (RF solo TopK1 Sharpe)")
    elif args.model == "ensemble" and args.uncalibrated:
        print(">> Computes: A3 (Uncalibrated ensemble TopK1 Sharpe)")
    elif args.exclude_ticker == "NVDA":
        print(">> Computes: A5 (NVDA-excluded TopK1 Sharpe)")
    else:
        print(">> Standard ensemble (main paper result)")
    print()

    for d in ["predictions", "metrics", "plots/reliability_diagrams", "permutation"]:
        (OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)

    print("Step 1/5: Loading data...")
    df = load_all_data(
        start=args.start, end=args.end,
        use_cache=args.use_cache, external_data_path=args.data_path,
    )
    feature_cols = get_feature_columns(df)
    print(f"  Loaded: {len(df)} rows, {len(feature_cols)} features")
    print(f"  Universe: {sorted(df.index.get_level_values('ticker').unique().tolist())}")

    if args.exclude_ticker:
        original = df.index.get_level_values('ticker').unique().tolist()
        df = df[df.index.get_level_values('ticker') != args.exclude_ticker]
        after = df.index.get_level_values('ticker').unique().tolist()
        print(f"\n  [EXCLUDE] '{args.exclude_ticker}' removed.")
        print(f"  Before: {sorted(original)}")
        print(f"  After : {sorted(after)}")
        print(f"  Rows  : {len(df)}")
        if args.exclude_ticker not in original:
            print(f"  [WARNING] Ticker not found — nothing removed.")

    print("\nStep 2/5: Generating folds...")
    folds = generate_folds(df)
    print_fold_summary(folds)

    print("\nStep 3/5: Loading SPY...")
    spy_returns = load_spy_returns("2015-01-01", "2025-01-01")
    print(f"  SPY: {len(spy_returns)} days")

    print("\nStep 4/5: Walk-forward loop...")
    predictions_df, ic_results, all_fold_ic = run_walk_forward_loop(
        df, feature_cols, folds,
        use_ensemble=(args.model == "ensemble"),
        model_type=args.model,
        use_calibration=not args.uncalibrated,
        exclude_ticker=args.exclude_ticker,      # ← passed correctly now
    )

    if not ic_results.iloc[0]['significant']:
        print("\n[WARNING] IC not significant.")
    else:
        print("\n[PASS] IC significant.")

    print("\nStep 5/5: Experiments...")
    run_all_experiments(
        predictions_df, spy_returns,
        model_type=args.model,
        exclude_ticker=args.exclude_ticker,      # ← passed correctly now
    )

    print("\nDone. All results in results/")