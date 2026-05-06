"""
factor_regression.py
====================
Fama-French 5-Factor + Momentum (6-Factor) regression analysis
for the Cross-Sectional Conviction Ranking paper.

Downloads Fama-French factor data directly from Ken French's website (FREE).
Reconstructs daily strategy returns from the predictions parquet.
Runs OLS regressions for all key strategies.
Outputs a publication-ready results table.

Run from research_clean/ directory:
    python scripts/factor_regression.py

Author: Rajveer Singh Pall
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

# ── Configuration ─────────────────────────────────────────────────────────────

PARQUET_PATH   = Path("results/predictions/predictions_ensemble.parquet")
OUTPUT_DIR     = Path("results/metrics")
COST_BPS       = 5.0        # one-way transaction cost in basis points
# Full 30-stock NASDAQ-100 continuous-member universe (same as main pipeline).
# NOTE: this constant documents the expected universe; the actual tickers are
# derived at runtime from the predictions parquet via get_level_values('ticker').
TICKERS = [
    "AAPL", "ADBE", "ADI",  "ADP",  "AMAT", "AMGN", "AMZN", "AVGO", "BIIB",
    "CDNS", "COST", "CSCO", "GILD", "GOOGL","INTC", "INTU", "KLAC", "LRCX",
    "MCHP", "MDLZ", "META", "MSFT", "NFLX", "NVDA", "PYPL", "QCOM", "REGN",
    "SBUX", "TSLA", "TXN",
]

# Ken French data URLs (free, no login required)
FF5_URL  = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL  = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
            "F-F_Momentum_Factor_daily_CSV.zip")


# ── Step 1: Reconstruct daily strategy returns ────────────────────────────────

def reconstruct_strategy_returns(
    df: pd.DataFrame,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """
    Reconstruct daily net returns for key strategies from the predictions DataFrame.

    Strategies reconstructed:
        TopK1           — top-1 by calibrated prob, daily
        TopK1_LongShort — long top-1, short bottom-1 (dollar-neutral)
        Baseline_P50    — equal-weight all tickers with prob > 0.50
        Equal_Weight    — equal-weight all tickers in the universe always
        Random_Top1     — mean of 50 random-selection simulations

    Parameters
    ----------
    df : pd.DataFrame
        MultiIndex (date, ticker) DataFrame from predictions_ensemble.parquet.
    cost_bps : float
        One-way transaction cost in basis points.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with one column per strategy, daily net returns.
    """
    cost = cost_bps / 10_000.0
    sorted_dates = sorted(df.index.get_level_values("date").unique())
    all_tickers  = sorted(df.index.get_level_values("ticker").unique())

    records = {s: [] for s in [
        "TopK1", "TopK1_LongShort", "Baseline_P50",
        "Equal_Weight", "Random_Top1"
    ]}
    dates_out = []

    # Previous weights per strategy (for turnover cost calculation)
    prev = {s: pd.Series(0.0, index=all_tickers) for s in records}

    rng = np.random.default_rng(seed=42)

    for date in sorted_dates:
        day = df.xs(date, level="date")           # 7-row slice, index = ticker
        probs  = day["prob"]
        ret    = day["actual_return"]

        # ── TopK1 ──────────────────────────────────────────────────────────
        top1_ticker = probs.idxmax()
        w_top1 = pd.Series(0.0, index=all_tickers)
        w_top1[top1_ticker] = 1.0

        # ── TopK1 Long-Short (dollar-neutral) ──────────────────────────────
        top1_t   = probs.idxmax()
        bot1_t   = probs.idxmin()
        w_ls = pd.Series(0.0, index=all_tickers)
        w_ls[top1_t]  =  0.5
        w_ls[bot1_t]  = -0.5     # short leg

        # ── Baseline_P50 ────────────────────────────────────────────────────
        eligible = probs[probs > 0.50].index.tolist()
        w_p50 = pd.Series(0.0, index=all_tickers)
        if eligible:
            w_p50[eligible] = 1.0 / len(eligible)

        # ── Equal_Weight ────────────────────────────────────────────────────
        w_ew = pd.Series(1.0 / len(all_tickers), index=all_tickers)

        # ── Random_Top1 (mean of 50 simulations) ───────────────────────────
        rand_rets = []
        for _ in range(50):
            rand_t = rng.choice(all_tickers)
            w_rand = pd.Series(0.0, index=all_tickers)
            w_rand[rand_t] = 1.0
            chg = (w_rand - prev["Random_Top1"]).abs().sum()
            rand_rets.append((ret[rand_t] if rand_t in ret.index else 0.0) - chg * cost)
        rand_net = float(np.mean(rand_rets))

        # ── Compute net returns (gross - turnover cost) ─────────────────────
        def net_ret(w_new, w_old, strategy_name):
            aligned_old, aligned_new = w_old.align(w_new, fill_value=0.0)
            turnover = (aligned_new - aligned_old).abs().sum()
            # For long-short, cost applies separately to each leg
            if strategy_name == "TopK1_LongShort":
                turnover = (aligned_new.abs() - aligned_old.abs()).abs().sum()
            # Gross return (handles negative weights for short leg)
            gross = (w_new * ret.reindex(w_new.index, fill_value=0.0)).sum()
            return float(gross - turnover * cost)

        records["TopK1"].append(net_ret(w_top1, prev["TopK1"], "TopK1"))
        records["TopK1_LongShort"].append(net_ret(w_ls, prev["TopK1_LongShort"], "TopK1_LongShort"))
        records["Baseline_P50"].append(net_ret(w_p50, prev["Baseline_P50"], "Baseline_P50"))
        records["Equal_Weight"].append(net_ret(w_ew, prev["Equal_Weight"], "Equal_Weight"))
        records["Random_Top1"].append(rand_net)

        # ── Update previous weights ─────────────────────────────────────────
        prev["TopK1"]           = w_top1.copy()
        prev["TopK1_LongShort"] = w_ls.copy()
        prev["Baseline_P50"]    = w_p50.copy()
        prev["Equal_Weight"]    = w_ew.copy()
        # Random_Top1 prev not tracked (random each day)

        dates_out.append(date)

    return pd.DataFrame(records, index=pd.DatetimeIndex(dates_out))


# ── Step 2: Download Fama-French factors ──────────────────────────────────────

import re
from io import StringIO
import pandas as pd

def _parse_ff_csv(raw_text: str) -> pd.DataFrame:
    """
    Parse Ken French daily factor CSV text robustly.
    Keeps only rows that start with YYYYMMDD.
    """
    lines = raw_text.splitlines()

    data_lines = []
    started = False

    for line in lines:
        s = line.strip()

        # Keep only actual daily data rows like:
        # 20180102,0.12,-0.34,...
        if re.match(r"^\d{8},", s):
            data_lines.append(s)
            started = True
            continue

        # Stop once data has started and we hit the next section / footer
        if started and (not s or s.lower().startswith("annual") or s.lower().startswith("copyright")):
            break

    if not data_lines:
        raise ValueError("No daily factor rows found in Ken French file.")

    df = pd.read_csv(StringIO("\n".join(data_lines)), header=None, sep=r"\s*,\s*", engine="python")

    # Extract date column safely
    dates = pd.to_datetime(df.iloc[:, 0].astype(str), format="%Y%m%d")

    # Drop original column and set index
    df = df.drop(columns=df.columns[0])
    df.index = dates
    df.index.name = "date"

    df = df.apply(pd.to_numeric, errors="coerce") / 100.0
    df.dropna(how="all", inplace=True)
    return df


def download_ff_factors() -> pd.DataFrame:
    """
    Download Fama-French 5 factors + Momentum factor (daily) from Ken French's
    website. Merges into a single DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
        Index: DatetimeIndex (daily)
    """
    print("Downloading Fama-French 5-Factor data from Ken French website...")
    r5 = requests.get(FF5_URL, timeout=60)
    z5 = zipfile.ZipFile(io.BytesIO(r5.content))
    fname5 = [f for f in z5.namelist() if f.upper().endswith(".CSV")][0]
    raw5 = z5.read(fname5).decode("latin-1")
    ff5 = _parse_ff_csv(raw5)
    # Standard FF5 columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    ff5.columns = ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"]

    print("Downloading Momentum factor data...")
    rm = requests.get(MOM_URL, timeout=60)
    zm = zipfile.ZipFile(io.BytesIO(rm.content))
    fnamem = [f for f in zm.namelist() if f.upper().endswith(".CSV")][0]
    rawm = zm.read(fnamem).decode("latin-1")
    mom = _parse_ff_csv(rawm)
    mom.columns = ["Mom"]

    factors = ff5.join(mom, how="left")
    print(f"  Factor data loaded: {len(factors)} days "
          f"({factors.index.min().date()} to {factors.index.max().date()})")
    return factors


# ── Step 3: Run OLS regressions ───────────────────────────────────────────────

def run_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    model_name: str = "CAPM",
) -> Dict:
    """
    Run a time-series OLS regression of strategy excess returns on factors.

    Models:
        CAPM   — 1-factor: Mkt_RF
        FF3    — 3-factor: Mkt_RF, SMB, HML
        FF5    — 5-factor: Mkt_RF, SMB, HML, RMW, CMA
        FF5+MOM — 6-factor: Mkt_RF, SMB, HML, RMW, CMA, Mom

    Parameters
    ----------
    strategy_returns : pd.Series
        Daily net returns for one strategy.
    factors : pd.DataFrame
        Factor returns including RF column.
    model_name : str
        One of 'CAPM', 'FF3', 'FF5', 'FF5+MOM'.

    Returns
    -------
    dict
        Regression results: alpha (annualised), t-stat, p-value, R2,
        factor loadings with t-stats, N observations.
    """
    factor_sets = {
        "CAPM":    ["Mkt_RF"],
        "FF3":     ["Mkt_RF", "SMB", "HML"],
        "FF5":     ["Mkt_RF", "SMB", "HML", "RMW", "CMA"],
        "FF5+MOM": ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "Mom"],
    }
    selected_factors = factor_sets[model_name]

    # Align on shared dates
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), factors],
        axis=1,
        join="inner",
    ).dropna()

    # Excess return = strategy return - risk-free rate
    y = aligned["strategy"] - aligned["RF"]
    X = sm.add_constant(aligned[selected_factors])

    # Newey-West HAC standard errors (lag=5 trading days)
    result = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})

    # Annualise alpha (daily alpha * 252)
    alpha_daily   = result.params["const"]
    alpha_annual  = alpha_daily * 252
    alpha_t       = result.tvalues["const"]
    alpha_p       = result.pvalues["const"]

    output = {
        "model":         model_name,
        "alpha_daily":   alpha_daily,
        "alpha_annual":  alpha_annual,
        "alpha_t":       alpha_t,
        "alpha_p":       alpha_p,
        "R2_adj":        result.rsquared_adj,
        "N":             int(result.nobs),
    }

    # Factor loadings
    for f in selected_factors:
        output[f"beta_{f}"]    = result.params[f]
        output[f"t_{f}"]       = result.tvalues[f]
        output[f"p_{f}"]       = result.pvalues[f]

    return output


def run_all_regressions(
    strategy_ret_df: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run FF5+MOM regression for every strategy, plus all 4 model specifications
    for the TopK1 strategy specifically.

    Returns
    -------
    pd.DataFrame
        Summary table suitable for the paper.
    """
    results = []

    # ── Primary table: all strategies under FF5+MOM ───────────────────────
    print("\nRunning FF5+MOM regression for all strategies...")
    for strat in strategy_ret_df.columns:
        r = run_regression(strategy_ret_df[strat], factors, "FF5+MOM")
        r["strategy"] = strat
        results.append(r)
        sig = "***" if r["alpha_p"] < 0.01 else "**" if r["alpha_p"] < 0.05 else "*" if r["alpha_p"] < 0.10 else ""
        print(f"  {strat:<22} alpha={r['alpha_annual']*100:+.2f}% p.a.  "
              f"t={r['alpha_t']:+.3f}  p={r['alpha_p']:.4f} {sig}  "
              f"R2={r['R2_adj']:.3f}")

    summary = pd.DataFrame(results).set_index("strategy")

    # ── Model specification comparison for TopK1 ─────────────────────────
    print("\nRunning model specification comparison for TopK1...")
    topk1_specs = []
    for model in ["CAPM", "FF3", "FF5", "FF5+MOM"]:
        r = run_regression(strategy_ret_df["TopK1"], factors, model)
        r["model_spec"] = model
        topk1_specs.append(r)
        sig = "***" if r["alpha_p"] < 0.01 else "**" if r["alpha_p"] < 0.05 else "*" if r["alpha_p"] < 0.10 else ""
        print(f"  {model:<10} alpha={r['alpha_annual']*100:+.2f}% p.a.  "
              f"t={r['alpha_t']:+.3f}  p={r['alpha_p']:.4f} {sig}  "
              f"Mkt_beta={r.get('beta_Mkt_RF', float('nan')):.3f}")

    spec_df = pd.DataFrame(topk1_specs)

    return summary, spec_df


# ── Step 4: Print publication-ready results ───────────────────────────────────

def print_paper_table(summary: pd.DataFrame, spec_df: pd.DataFrame) -> None:
    """Print the key numbers formatted for the paper's results section."""

    def sig_stars(p):
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""

    print("\n" + "=" * 70)
    print("TABLE 3 — FAMA-FRENCH 6-FACTOR REGRESSION (FF5 + MOMENTUM)")
    print("All strategies | HAC Newey-West standard errors (lag=5)")
    print("=" * 70)
    print(f"{'Strategy':<22} {'Ann. Alpha':>12} {'t-stat':>8} {'p-value':>8} "
          f"{'Mkt beta':>10} {'Mom beta':>10} {'Adj R2':>8}")
    print("-" * 70)

    for strat, row in summary.iterrows():
        stars = sig_stars(row["alpha_p"])
        mkt_b = row.get("beta_Mkt_RF", float("nan"))
        mom_b = row.get("beta_Mom", float("nan"))
        print(f"{strat:<22} "
              f"{row['alpha_annual']*100:>+10.2f}%{stars:<2} "
              f"{row['alpha_t']:>8.3f} "
              f"{row['alpha_p']:>8.4f} "
              f"{mkt_b:>10.3f} "
              f"{mom_b:>10.3f} "
              f"{row['R2_adj']:>8.3f}")

    print("-" * 70)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("Note: Alpha is annualised (daily alpha * 252). Returns net of 5 bps "
          "one-way costs.")

    print("\n" + "=" * 70)
    print("TABLE 4 — MODEL SPECIFICATION COMPARISON: TopK1 STRATEGY")
    print("=" * 70)
    print(f"{'Model':<12} {'Ann. Alpha':>12} {'t-stat':>8} {'p-value':>8} "
          f"{'Mkt beta':>10} {'Adj R2':>8} {'N obs':>7}")
    print("-" * 70)

    for _, row in spec_df.iterrows():
        stars = sig_stars(row["alpha_p"])
        mkt_b = row.get("beta_Mkt_RF", float("nan"))
        print(f"{row['model_spec']:<12} "
              f"{row['alpha_annual']*100:>+10.2f}%{stars:<2} "
              f"{row['alpha_t']:>8.3f} "
              f"{row['alpha_p']:>8.4f} "
              f"{mkt_b:>10.3f} "
              f"{row['R2_adj']:>8.3f} "
              f"{int(row['N']):>7}")

    print("-" * 70)
    print("Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print("HAC Newey-West standard errors with lag=5 throughout.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load predictions
    print("Loading predictions parquet...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df)} rows | {df.index.get_level_values('date').nunique()} dates")

    # 2. Reconstruct strategy returns
    print("\nReconstructing daily strategy returns...")
    strategy_ret = reconstruct_strategy_returns(df)
    print(f"  Returns reconstructed: {strategy_ret.shape}")
    print(f"  Date range: {strategy_ret.index.min().date()} "
          f"to {strategy_ret.index.max().date()}")

    # Quick sanity check — annualised returns should match paper
    n = len(strategy_ret)
    for col in strategy_ret.columns:
        cum = (1 + strategy_ret[col]).prod()
        ann = cum ** (252 / n) - 1
        print(f"  {col:<22} Ann. return = {ann*100:+.1f}%")

    # 3. Download FF factors
    factors = download_ff_factors()

    # 4. Run regressions
    summary, spec_df = run_all_regressions(strategy_ret, factors)

    # 5. Print publication table
    print_paper_table(summary, spec_df)

    # 6. Save to CSV
    summary.to_csv(OUTPUT_DIR / "factor_regression_all_strategies.csv")
    spec_df.to_csv(OUTPUT_DIR / "factor_regression_topk1_specs.csv", index=False)
    strategy_ret.to_csv(OUTPUT_DIR / "daily_strategy_returns.csv")
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("  factor_regression_all_strategies.csv")
    print("  factor_regression_topk1_specs.csv")
    print("  daily_strategy_returns.csv")
    print("\n[DONE]")