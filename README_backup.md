# When the Gate Stays Closed

### Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework

**Rajveer Singh Pall** — Independent Researcher &nbsp;·&nbsp; rajveerpall04@gmail.com

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Status: Under Review](https://img.shields.io/badge/status-under%20review-f59e0b)]()
[![Reproducible](https://img.shields.io/badge/reproducible-yes-2A9D8F)]()

---

## What This Paper Is About

Most machine learning research in finance asks: *can we predict the market?* This paper asks the harder question: *how do we know when a model is ready to trade — and what happens when it isn't?*

The answer is the **IC-Gated Deployment Framework (ICGDF)**: a two-stage statistical filter that requires a model to demonstrate genuine cross-sectional predictive skill (measured by Information Coefficient with HAC-corrected inference) before any capital is deployed. Both stages must pass simultaneously; if either fails, the system takes no position.

I applied ICGDF to a three-model ensemble (CatBoost + Random Forest + MLP) on 30 NASDAQ-100 stocks across **1,512 consecutive out-of-sample trading days** (October 2018 – October 2024). The gate never opened — not once in twelve walk-forward folds. Mean IC = −0.0005, HAC t = −0.09, p = 0.464.

That is the finding. And it is worth reporting carefully.

> **The model is well-calibrated (ECE < 0.025) but produces zero exploitable cross-sectional discrimination.** This demonstrates that calibration quality and predictive content are orthogonal properties — a subtle but important distinction for practitioners who use probability estimates as a deployment readiness signal.

A momentum positive control (252-day trailing return) achieves Sharpe = 0.57 over the same window, confirming that cross-sectional structure *does* exist in this universe. Momentum's IC is directionally positive but statistically insufficient under HAC-corrected inference (p = 0.276), suggesting its return comes from multi-week trend persistence rather than daily rank discrimination — a mechanistically distinct channel that the IC gate is not designed to screen.

---

## Key Results at a Glance

| Test | Value | Threshold | Decision |
|---|---|---|---|
| Mean IC | −0.0005 | > 0 | — |
| IC Std Dev | 0.2204 | — | — |
| HAC t-statistic | −0.09 | > 1.645 | Not significant |
| p-value (one-tailed) | 0.464 | < 0.05 | **Gate CLOSED** |
| Permutation p-value | 0.742 | < 0.05 | **Gate CLOSED** |
| Gate-open folds | 0 / 12 | ≥ 1 | Never opened |

**The benchmark convergence signature** — Sharpe increasing monotonically from TopK1 (−0.16) to TopK2 (−0.01) to TopK3 (+0.12) toward the equal-weight limit (0.96) — is the mathematical diagnostic of a cross-sectional ranker with zero information content.

---

## Repository Structure

```
when-the-gate-stays-closed/
│
├── paper/
│   └── when_the_gate_stays_closed_FINAL.docx     # Manuscript (submission version)
│
├── src/                                            # Core pipeline modules
│   ├── data/
│   │   └── data_loader.py                          # Yahoo Finance data acquisition
│   ├── training/
│   │   ├── models.py                               # CatBoost, Random Forest, MLP ensemble
│   │   ├── calibration.py                          # Isotonic probability calibration
│   │   └── walk_forward.py                         # 12-fold expanding-window validator
│   └── backtesting/
│       └── backtester.py                           # Vectorised portfolio backtest engine
│
├── scripts/
│   ├── run_experiments.py                          # Main pipeline entry point
│   ├── factor_regression.py                        # Fama-French factor regressions
│   ├── parallel_permutation.py                     # Permutation test (parallelised, B=1,000)
│   ├── robustness/
│   │   ├── robustness_01_expanded_universe.py      # R1: N=100 universe check
│   │   ├── robustness_02_shap_analysis.py          # R2: SHAP feature attribution & stability
│   │   ├── robustness_03_04_05_dm_vix_bootstrap.py # R3–R5: DM test, VIX IC, bootstrap CIs
│   │   ├── robustness_06_momentum_ic_gate.py       # Momentum positive control (IC gate)
│   │   └── robustness_07_ablation.py               # Gate component ablation study
│   └── revision_audit/                             # Pre-submission statistical audit scripts
│       ├── power_analysis.py                       # MDE and power curves for null result
│       ├── hac_lag_sensitivity.py                  # NW bandwidth robustness (L=1..20)
│       ├── permutation_test_clarification.py       # Type A vs B permutation comparison
│       ├── survivorship_bias_quantification.py     # Bias magnitude estimation
│       ├── mlp_calibration_window_check.py         # Val/calibration overlap diagnosis
│       ├── one_tailed_pvalue_audit.py              # Full p-value consistency audit
│       └── yfinance_data_validation_report.py      # Data quality protocol & sensitivity
│
├── results/
│   ├── figures/
│   │   ├── pub/                                    # Publication-ready figures (PNG + PDF)
│   │   └── revision_audit/                         # Figures from audit scripts
│   ├── metrics/                                    # IC statistics, strategy metrics (CSV)
│   ├── permutation/                                # Permutation null distributions (CSV)
│   ├── plots/reliability_diagrams/                 # Per-fold calibration diagrams
│   └── robustness/
│       ├── expanded_universe/                      # R1 outputs
│       ├── shap/                                   # R2 outputs
│       ├── dm_test/                                # R3 outputs
│       ├── vix_ic/                                 # R4 outputs
│       ├── bootstrap/                              # R5 outputs
│       ├── momentum_ic/                            # Momentum IC gate results
│       └── ablation/                               # Ablation study results
│
├── data/
│   └── nasdaq30_prices.parquet                     # Adjusted OHLCV, 30 stocks, 2015–2024
│
├── generate_figures.py                             # Publication figure generation
├── build_manuscript_v2.py                          # Manuscript builder (python-docx)
├── requirements.txt
└── .gitignore
```

---

## Methodology

### The ICGDF Algorithm

The gate applies two conditions before every deployment decision. Both must hold; if either fails, no position is taken.

**Input:** Daily OHLCV panel for N stocks over T trading days · α = 0.05 · HAC lag L = 9 · permutation replicates B = 1,000.

**Stage 1 — Training and Calibration (per fold k)**
1. Construct expanding training window with 2-calendar-day embargo to prevent lookahead.
2. Engineer 49 strictly causal OHLCV features; no future-referencing windows.
3. Fit CatBoost, Random Forest, and MLP independently; combine by equal probability averaging.
4. Calibrate via isotonic regression on the final 20% of training data (frozen before test begins).

**Stage 2 — IC Gate (before each deployment decision)**

5. Compute IC for the full test fold: IC_d = SpearmanRankCorr(p̂_d, r_{d+1}).
6. **Condition A** — Newey-West HAC t-test (one-tailed, L = 9):

   &nbsp;&nbsp;&nbsp;&nbsp;`t_HAC = IC̄ / √(V̂_HAC / N) > 1.645   AND   IC̄ > 0`

7. **Condition B** — Permutation test (temporal shuffle, B = 1,000):

   &nbsp;&nbsp;&nbsp;&nbsp;`p_perm < 0.05`

8. Gate opens ⟺ Condition A **AND** Condition B.
9. Gate closed → no position. Gate open → equal-weight the K highest-conviction stocks, 5 bps round-trip cost.

The gate is **model-agnostic**: any base learner producing a cross-sectional conviction ranking can replace Steps 3–4 without modifying the gate logic.

### Walk-Forward Design

| Parameter | Value |
|---|---|
| Out-of-sample period | October 2018 – October 2024 |
| Total OOS trading days | 1,512 |
| Folds | 12 expanding windows (6-month increments) |
| Embargo | 2 calendar days |
| Calibration window | Last 20% of each training period |
| HAC lag | 9 days (rule-of-thumb for T ≈ 126: L = floor(4·(T/100)^{2/9}) = 7; L=9 is conservative) |
| Permutation replicates | B = 1,000 |
| Universe | 30 NASDAQ-100 continuous members, survivorship-bias-mitigated |
| Features | 49 strictly causal OHLCV technical indicators |

### Ensemble Configuration

| Component | Configuration |
|---|---|
| CatBoost | 500 trees · depth 6 · lr 0.05 · l2_leaf_reg 3.0 |
| Random Forest | 500 trees · max_depth 10 · min_samples_leaf 20 |
| MLP | [256→128→64] · dropout 0.3 · early stopping (patience 10) |
| Combination | Equal probability averaging |
| Calibration | Isotonic regression (per fold, fitted on training tail, frozen before test) |
| Random seed | 42 (all components) |

---

## Full Results

### Strategy Performance (October 2018 – October 2024)

| Strategy | Ann. Return | Sharpe | Sortino | Max Drawdown | Trades |
|---|---|---|---|---|---|
| Equal-Weight Benchmark | +25.0% | **0.96** | 1.28 | −32.4% | 1 |
| SPY Buy & Hold | +14.9% | 0.74 | 0.91 | −33.7% | 0 |
| Momentum Top-1 | +26.4% | 0.57 | 0.79 | −62.7% | 407 |
| TopK3 (ML) | +3.5% | 0.12 | 0.16 | −38.2% | 1,248 |
| TopK2 (ML) | −0.3% | −0.01 | −0.01 | −53.9% | 1,134 |
| Random Top-1 | −4.6% | −0.12 | −0.15 | −65.6% | 1,461 |
| **TopK1 (ML, gate closed)** | **−5.9%** | **−0.16** | **−0.21** | **−67.0%** | 833 |

### Ablation Study: Why Both Gate Components Are Necessary

Simulated AR(1) null IC process (φ = 0.30, N = 126 days/trial, 500 trials):

| Gate Variant | Null False Positive Rate | Assessment |
|---|---|---|
| Naive t-test only | **11.8%** | Severely inflated (2× nominal α) |
| HAC t-test only | 7.6% | Reduced but above α |
| **Full ICGDF** | **0.0%** | Correct under simulated null |

### Robustness Checks (All Confirm the Null)

| Check | Key Result | Interpretation |
|---|---|---|
| R1: Expanded Universe (N=100) | IC = −0.006, p = 0.947 | Not universe-specific |
| R2: SHAP Feature Attribution | Inter-fold rank ρ = 0.13–0.40 | No stable signal; noise-fitting confirmed |
| R3: Diebold-Mariano Test | DM = 0.42, p = 0.672 vs. Random Top-1 | ML indistinguishable from random selection |
| R4: VIX-Conditioned IC | Min p = 0.136 across all regimes | Gate closed in all volatility environments |
| R5: Block Bootstrap CIs | 0 / 12 folds exclude zero | All fold CIs consistent with null IC |

### Momentum Positive Control

| Signal | Mean IC | IC Std | HAC t | p-value | Gate |
|---|---|---|---|---|---|
| ML Ensemble | −0.0005 | 0.2204 | −0.09 | 0.464 | CLOSED |
| Momentum (252-day) | +0.0071 | 0.4747 | +0.60 | 0.276 | CLOSED |

Momentum's Sharpe advantage (0.57) arises from multi-week trend persistence, not daily IC significance — a mechanistically distinct predictive channel.

### Calibration vs. Discrimination

| Metric | Value | Interpretation |
|---|---|---|
| Mean ECE (12 folds) | < 0.025 | Excellent calibration |
| Mean IC | −0.0005 | Zero discriminative content |

A well-calibrated model with zero IC is a real, practically important failure mode — not an artefact.

---

## Reproducing Results

### 1. Clone and install

```bash
git clone https://github.com/Rajveer-code/transaction-cost-trap.git
cd transaction-cost-trap
pip install -r requirements.txt
```

> **Data note:** `data/nasdaq30_prices.parquet` (3.2 MB) is included and contains adjusted OHLCV for all 30 stocks from January 2015 through December 2024. Scripts use this cached file by default; delete it to force a fresh Yahoo Finance download.

### 2. Run the main pipeline (~30 min)

```bash
python scripts/run_experiments.py          # Walk-forward training, IC gate, backtest
python scripts/factor_regression.py        # CAPM / FF3 / FF5 / FF5+MOM regressions
python scripts/parallel_permutation.py     # Permutation null (B=1,000, parallelised)
```

### 3. Run robustness checks (in order)

```bash
python scripts/robustness/robustness_01_expanded_universe.py      # ~15 min
python scripts/robustness/robustness_02_shap_analysis.py          # ~5 min
python scripts/robustness/robustness_03_04_05_dm_vix_bootstrap.py # ~10 min
python scripts/robustness/robustness_06_momentum_ic_gate.py       # ~3 min
python scripts/robustness/robustness_07_ablation.py               # ~2 min
```

All write outputs to `results/robustness/`. Run from the repository root.

### 4. Generate publication figures

```bash
python generate_figures.py
# → results/figures/pub/fig01_*.png through fig12_*.png
```

### 5. Run revision audit scripts

These self-contained scripts address statistical methodology questions raised during peer review (power analysis, HAC bandwidth justification, permutation test interpretation, survivorship bias quantification, MLP calibration overlap, p-value consistency, and data validation).

```bash
cd scripts/revision_audit

python power_analysis.py                    # MDE table + figure_power_analysis.png
python hac_lag_sensitivity.py               # HAC sensitivity table + figure_hac_sensitivity.png
python permutation_test_clarification.py    # Type A vs B + figure_permutation_comparison.png
python survivorship_bias_quantification.py  # Bias quantification table
python mlp_calibration_window_check.py      # Val/calibration overlap diagnosis
python one_tailed_pvalue_audit.py           # Full p-value audit table
python yfinance_data_validation_report.py   # Data quality protocol

# Figures are saved to results/figures/revision_audit/
```

Each script prints results directly to stdout in LaTeX-compatible format for manuscript insertion, and requires only `numpy`, `scipy`, and `matplotlib`.

### 6. Build the manuscript document

```bash
python build_manuscript_v2.py
# → paper/when_the_gate_stays_closed_FINAL.docx
```

---

## Requirements

Tested on Python 3.10 / 3.11 / 3.12, Windows 11 and Ubuntu 22.04.

```
catboost>=1.2
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.11
yfinance>=0.2.28
shap>=0.44
python-docx>=1.1
matplotlib>=3.7
seaborn>=0.13
statsmodels>=0.14
pyarrow>=14.0
```

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{pall2025icgdf,
  title   = {When the Gate Stays Closed: Empirical Evidence of Near-Zero
             Cross-Sectional Predictability in Large-Cap {NASDAQ} Equities
             Using an {IC}-Gated Machine Learning Framework},
  author  = {Pall, Rajveer Singh},
  year    = {2025},
  note    = {Working paper. Available at \url{https://github.com/Rajveer-code/transaction-cost-trap}}
}
```

---

## License

MIT. See [LICENSE](LICENSE) for full terms.

---

## Contact

**Rajveer Singh Pall** &nbsp;·&nbsp; rajveerpall04@gmail.com &nbsp;·&nbsp; [github.com/Rajveer-code](https://github.com/Rajveer-code)
