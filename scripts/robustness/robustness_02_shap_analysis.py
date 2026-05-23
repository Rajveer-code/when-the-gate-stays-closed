"""
robustness_02_shap_analysis.py  [FIXED v2]
==========================================
SHAP feature importance analysis across walk-forward folds.

FIXES APPLIED:
  1. load_pipeline_data() detects raw OHLCV parquet (no 'target' col, <10 cols)
     and runs full feature engineering before returning — no more KeyError.
  2. Fold generator is fully self-contained (no cross-script imports).
  3. Path resolution works whether script is in src/, repo root, or anywhere.
  4. All NaN/inf in feature arrays are cleaned before SHAP computation.

Run from repo root OR from src/:
  python robustness_02_shap_analysis.py
  python src/robustness_02_shap_analysis.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.stats
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── SHAP ─────────────────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
    print("[OK] shap available")
except ImportError:
    SHAP_AVAILABLE = False
    print("[WARN] shap not installed — pip install shap. Using RF importances fallback.")

# ── Resolve repo root (works from any sub-directory) ─────────────────────────
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
for _ in range(5):
    if (_ROOT / "requirements.txt").exists() or (_ROOT / "data").exists():
        break
    _ROOT = _ROOT.parent
print(f"[PATH] Repo root resolved: {_ROOT}")
sys.path.insert(0, str(_ROOT))

# ── Try src/ modules (optional — full fallback if absent) ─────────────────────
MODULES_AVAILABLE = False
try:
    from src.data.data_loader import load_all_data, get_feature_columns
    from src.training.walk_forward import generate_folds, get_fold_arrays
    from src.training.models import CatBoostModel
    MODULES_AVAILABLE = True
    print("[OK] src/ pipeline modules imported")
except ImportError as e:
    print(f"[INFO] src/ not available ({e}). Fully self-contained mode.")

OUT_DIR = _ROOT / "results" / "robustness" / "shap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SHAP_FOLDS   = 4
SHAP_SAMPLE_N  = 500
TOP_N_FEATURES = 20

PAPER_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    "AVGO","QCOM","TXN","AMAT","LRCX","KLAC","ADI","INTC","MCHP",
    "ADBE","CSCO","INTU","ADP","CDNS",
    "AMGN","GILD","BIIB","REGN",
    "NFLX","COST","SBUX","MDLZ","PYPL",
]


# ═════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED FEATURE ENGINEERING  (mirrors data_loader.py exactly)
# ═════════════════════════════════════════════════════════════════════════════

def _rsi(c, n):
    d = c.diff(1)
    g = d.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + g / (l + 1e-10))

def _ema(s, n): return s.ewm(span=n, min_periods=n, adjust=False).mean()
def _sma(s, n): return s.rolling(n, min_periods=n).mean()

def _atr(h, l, c, n):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, min_periods=n, adjust=False).mean()


def compute_features_for_ticker(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """49 causal technical features — identical column names to data_loader.py."""
    o = ohlcv["Open"]
    h = ohlcv["High"]
    l = ohlcv["Low"]
    c = ohlcv["Close"]
    v = ohlcv["Volume"].replace(0, np.nan)
    f = {}

    f["RSI_14"] = _rsi(c, 14);  f["RSI_21"] = _rsi(c, 21)
    m12 = _ema(c, 12);  m26 = _ema(c, 26);  macd = m12 - m26
    sig = macd.ewm(span=9, min_periods=9, adjust=False).mean()
    f["MACD_12_26"] = macd;  f["MACD_signal_9"] = sig;  f["MACD_hist"] = macd - sig

    bs = _sma(c, 20);  bsd = c.rolling(20, min_periods=20).std()
    bu = bs + 2*bsd;   bl  = bs - 2*bsd
    f["BB_upper"] = bu;  f["BB_lower"] = bl;  f["BB_mid"] = bs
    f["BB_width"] = (bu - bl)/(bs+1e-10);  f["BB_pct_b"] = (c-bl)/(bu-bl+1e-10)

    f["ATR_14"] = _atr(h, l, c, 14);  f["ATR_21"] = _atr(h, l, c, 21)
    obv = (np.sign(c.diff(1)).fillna(0)*v).cumsum()
    f["OBV"] = obv;  f["OBV_EMA"] = obv.ewm(span=21, min_periods=21, adjust=False).mean()
    f["EMA_9"]=_ema(c,9); f["EMA_21"]=_ema(c,21); f["EMA_50"]=_ema(c,50); f["EMA_200"]=_ema(c,200)
    s50=_sma(c,50); s200=_sma(c,200)
    f["SMA_50"]=s50; f["SMA_200"]=s200
    f["price_to_SMA200"]=c/(s200+1e-10); f["price_to_SMA50"]=c/(s50+1e-10)
    for lag in [1,2,3,5,10,21]: f[f"return_{lag}d"] = c.pct_change(lag)
    for w in [5,21]:
        vm=v.rolling(w,min_periods=w).mean(); vs=v.rolling(w,min_periods=w).std()
        f[f"volume_zscore_{w}d"] = (v-vm)/(vs+1e-10)
    cr = h-l+1e-10
    f["OC_body_norm"]      = (c-o)/cr
    f["upper_shadow_ratio"] = (h - pd.concat([o,c],axis=1).max(axis=1))/cr
    f["lower_shadow_ratio"] = (pd.concat([o,c],axis=1).min(axis=1)-l)/cr
    f["HL_range_norm"] = cr/(c+1e-10)
    lr = np.log(c/c.shift(1))
    f["rolling_vol_5d"]=lr.rolling(5,min_periods=5).std()
    f["rolling_vol_21d"]=lr.rolling(21,min_periods=21).std()
    f["rolling_vol_63d"]=lr.rolling(63,min_periods=63).std()
    tp=(h+l+c)/3
    mad=tp.rolling(20,min_periods=20).apply(lambda x:np.mean(np.abs(x-np.mean(x))),raw=True)
    f["CCI_20"]=(tp-_sma(tp,20))/(0.015*mad+1e-10)
    raw_mf=tp*v; sgn=np.sign(tp.diff(1))
    f["MFI_14"]=100-100/(1+(raw_mf.where(sgn>0,0).rolling(14,min_periods=14).sum()/
                            (raw_mf.where(sgn<0,0).rolling(14,min_periods=14).sum()+1e-10)))
    f["ROC_10"]=(c/c.shift(10)-1)*100;  f["ROC_21"]=(c/c.shift(21)-1)*100
    half=20//2+1;  f["DPO_20"]=c.shift(half)-_sma(c,20).shift(half)
    tv=(tp*v).rolling(20,min_periods=20).sum(); vs20=v.rolling(20,min_periods=20).sum()
    f["VWAP_deviation"]=(c-(tv/(vs20+1e-10)))/(tv/(vs20+1e-10)+1e-10)
    ll14=l.rolling(14,min_periods=14).min(); hh14=h.rolling(14,min_periods=14).max()
    sk=100*(c-ll14)/(hh14-ll14+1e-10)
    f["stoch_K"]=sk;  f["stoch_D"]=sk.rolling(3,min_periods=3).mean()
    f["Williams_R"]=-100*(hh14-c)/(hh14-ll14+1e-10)
    pch=h.shift(1); pcl=l.shift(1); pcc=c.shift(1)
    up=(h-pch).where((h-pch)>(pcl-l).clip(lower=0),0).clip(lower=0)
    dn=(pcl-l).where((pcl-l)>(h-pch).clip(lower=0),0).clip(lower=0)
    tr2=pd.concat([h-l,(h-pcc).abs(),(l-pcc).abs()],axis=1).max(axis=1)
    a14=tr2.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    dip=100*up.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/(a14+1e-10)
    dim=100*dn.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/(a14+1e-10)
    dx=100*(dip-dim).abs()/(dip+dim+1e-10)
    f["ADX_14"]=dx.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    f["DI_plus"]=dip; f["DI_minus"]=dim
    return pd.DataFrame(f, index=ohlcv.index)


def compute_target(close: pd.Series) -> pd.Series:
    f2=close.shift(-2); f1=close.shift(-1)
    lbl=(f2>f1).astype(float)
    return lbl.where(~(f2.isna()|f1.isna()), other=np.nan)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LOADING  (detects raw vs processed, builds features if needed)
# ═════════════════════════════════════════════════════════════════════════════

def _is_raw_ohlcv(df: pd.DataFrame) -> bool:
    """Return True if df looks like raw OHLCV (few columns, no target)."""
    if "target" in df.columns:
        return False
    ohlcv_names = {"open","high","low","close","volume",
                   "Open","High","Low","Close","Volume"}
    col_set = set(df.columns)
    # Raw if most columns are OHLCV names
    return len(col_set & ohlcv_names) >= 3


def _build_from_raw(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Given raw OHLCV (MultiIndex or with ticker/date columns), run
    feature engineering and return a processed DataFrame + feature_cols.
    """
    # Normalise column names
    raw_df = raw_df.copy()
    col_map = {c: c.capitalize() for c in raw_df.columns
               if c.lower() in ("open","high","low","close","volume")}
    raw_df.rename(columns=col_map, inplace=True)

    # Reset MultiIndex if needed
    if isinstance(raw_df.index, pd.MultiIndex):
        raw_df = raw_df.reset_index()

    # Identify date and ticker columns
    date_col   = next((c for c in raw_df.columns if c.lower() in ("date","datetime","index")), None)
    ticker_col = next((c for c in raw_df.columns if c.lower() in ("ticker","symbol")), None)

    if date_col is None:
        # Date might be the unnamed index
        raw_df = raw_df.reset_index()
        raw_df.rename(columns={raw_df.columns[0]: "date"}, inplace=True)
        date_col = "date"

    raw_df[date_col] = pd.to_datetime(raw_df[date_col])

    required = {"Open","High","Low","Close","Volume"}
    missing  = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Cannot find OHLCV columns. Missing: {missing}. "
                         f"Available: {list(raw_df.columns)}")

    frames = []

    if ticker_col:
        tickers = raw_df[ticker_col].unique()
        print(f"[FEATURES] Building features for {len(tickers)} tickers …")
        for tk in tickers:
            sub = raw_df[raw_df[ticker_col] == tk].set_index(date_col).sort_index()
            sub.index.name = "date"
            if len(sub) < 300:
                print(f"  [SKIP] {tk}: {len(sub)} rows")
                continue
            try:
                feats  = compute_features_for_ticker(sub)
                target = compute_target(sub["Close"])
                feats["Close"]  = sub["Close"]
                feats["target"] = target
                feats["ticker"] = tk
                frames.append(feats.reset_index())
                print(f"  [OK] {tk}: {len(feats)} rows")
            except Exception as exc:
                print(f"  [ERR] {tk}: {exc}")
    else:
        print("[FEATURES] Single-ticker dataset.")
        sub = raw_df.set_index(date_col).sort_index()
        sub.index.name = "date"
        feats  = compute_features_for_ticker(sub)
        target = compute_target(sub["Close"])
        feats["Close"]  = sub["Close"]
        feats["target"] = target
        feats["ticker"] = "SINGLE"
        frames.append(feats.reset_index())

    if not frames:
        raise RuntimeError("No tickers processed. Check OHLCV column names.")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.set_index(["date","ticker"]).sort_index()

    feature_cols = [c for c in combined.columns if c not in {"Close","target"}]
    before = len(combined)
    combined = combined.dropna(subset=feature_cols + ["target"])
    print(f"  Dropped {before - len(combined)} NaN rows (warmup + target)")
    print(f"  Final: {len(combined):,} rows, {len(feature_cols)} features, "
          f"{combined.index.get_level_values('ticker').nunique()} tickers")
    return combined, feature_cols


def load_pipeline_data() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the processed feature matrix (with 'target' column).

    Priority:
      1. results/robustness/shap/feature_cache.parquet  (pre-built, fastest)
      2. src/ modules (load_all_data)
      3. data/nasdaq30_prices.parquet → detect raw vs processed → build if raw
      4. data/raw/*_ohlcv.parquet    → individual ticker files → build features
      5. yfinance download of paper's 30 tickers → build features
    """
    feature_cache = OUT_DIR / "feature_cache.parquet"

    # 0. Pre-built cache
    if feature_cache.exists():
        print(f"[CACHE] {feature_cache}")
        df = pd.read_parquet(feature_cache)
        fc = [c for c in df.columns if c not in {"Close","target"}]
        print(f"  {len(df):,} rows | {len(fc)} features | "
              f"{df.index.get_level_values('ticker').nunique()} tickers")
        return df, fc

    # 1. src/ modules
    if MODULES_AVAILABLE:
        raw_path = _ROOT / "data" / "nasdaq30_prices.parquet"
        df = (load_all_data(external_data_path=raw_path, use_cache=True)
              if raw_path.exists()
              else load_all_data(use_cache=True))
        fc = get_feature_columns(df)
        df.to_parquet(feature_cache);  print(f"[SAVED] {feature_cache}")
        return df, fc

    # 2. nasdaq30_prices.parquet
    main_parquet = _ROOT / "data" / "nasdaq30_prices.parquet"
    if main_parquet.exists():
        print(f"[DATA] Loading {main_parquet}")
        raw = pd.read_parquet(main_parquet)
        print(f"  Shape: {raw.shape} | Columns: {list(raw.columns)[:8]}")

        if _is_raw_ohlcv(raw):
            print("  → Raw OHLCV detected. Running feature engineering …")
            df, fc = _build_from_raw(raw)
        else:
            # Already processed
            if not isinstance(raw.index, pd.MultiIndex):
                raw = raw.reset_index()
                dc = next(c for c in raw.columns if c.lower() == "date")
                tc = next((c for c in raw.columns if c.lower() == "ticker"), None)
                if tc:
                    raw = raw.set_index([dc, tc])
                else:
                    raw["ticker"] = "UNKNOWN"
                    raw = raw.set_index([dc, "ticker"])
                raw.index.names = ["date","ticker"]
            fc = [c for c in raw.columns if c not in {"Close","target"}]
            df = raw
        df.to_parquet(feature_cache);  print(f"[SAVED] {feature_cache}")
        return df, fc

    # 3. Per-ticker parquets in data/raw/
    raw_dir = _ROOT / "data" / "raw"
    ticker_files = list(raw_dir.glob("*_ohlcv.parquet")) if raw_dir.exists() else []
    if ticker_files:
        print(f"[DATA] Found {len(ticker_files)} per-ticker parquets in {raw_dir}")
        frames = []
        for p in ticker_files:
            tk = p.stem.replace("_ohlcv","")
            t  = pd.read_parquet(p)
            t.index = pd.to_datetime(t.index).tz_localize(None).normalize()
            t.index.name = "date"
            col_map = {c: c.capitalize() for c in t.columns
                       if c.lower() in ("open","high","low","close","volume")}
            t.rename(columns=col_map, inplace=True)
            if len(t) < 300: continue
            try:
                feats  = compute_features_for_ticker(t)
                target = compute_target(t["Close"])
                feats["Close"] = t["Close"];  feats["target"] = target;  feats["ticker"] = tk
                frames.append(feats.reset_index())
            except Exception as exc:
                print(f"  [ERR] {tk}: {exc}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"])
            combined = combined.set_index(["date","ticker"]).sort_index()
            fc = [c for c in combined.columns if c not in {"Close","target"}]
            combined = combined.dropna(subset=fc+["target"])
            combined.to_parquet(feature_cache)
            return combined, fc

    # 4. Download
    print("[DATA] Downloading 30 tickers from yfinance …")
    frames = []
    for tk in PAPER_TICKERS:
        try:
            t = yf.Ticker(tk)
            raw = t.history(start="2015-01-01", end="2025-01-01",
                            interval="1d", auto_adjust=True)
            raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
            raw.index.name = "date"
            raw = raw[["Open","High","Low","Close","Volume"]].dropna()
            if len(raw) < 500: continue
            feats  = compute_features_for_ticker(raw)
            target = compute_target(raw["Close"])
            feats["Close"] = raw["Close"];  feats["target"] = target;  feats["ticker"] = tk
            frames.append(feats.reset_index())
            print(f"  [OK] {tk}: {len(raw)} rows")
        except Exception as exc:
            print(f"  [ERR] {tk}: {exc}")
    if not frames:
        raise RuntimeError("Could not load data from any source.")
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined.set_index(["date","ticker"]).sort_index()
    fc = [c for c in combined.columns if c not in {"Close","target"}]
    combined = combined.dropna(subset=fc+["target"])
    combined.to_parquet(feature_cache)
    print(f"[SAVED] {feature_cache}")
    return combined, fc


# ═════════════════════════════════════════════════════════════════════════════
#  SELF-CONTAINED FOLD GENERATOR  (matches walk_forward.py exactly)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SimpleFold:
    fold_number: int
    train_dates: list
    cal_dates: list
    test_dates: list
    model_train_dates: list
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds_self(df: pd.DataFrame) -> List[SimpleFold]:
    unique_dates = sorted(df.index.get_level_values("date").unique())
    n = len(unique_dates)
    MIN_TRAIN = 756;  TEST = 126;  STEP = 126;  EMBARGO = 2
    folds, i, fn = [], 0, 0
    while True:
        tr_end   = MIN_TRAIN + i * STEP
        te_start = tr_end + EMBARGO
        te_end   = te_start + TEST
        if te_end > n: break
        tr   = unique_dates[:tr_end]
        te   = unique_dates[te_start:te_end]
        cal  = tr[int(len(tr)*0.8):]
        mtr  = tr[:int(len(tr)*0.8)]
        fn  += 1
        folds.append(SimpleFold(fn, tr, cal, te, mtr, te[0], te[-1]))
        i += 1
    return folds


# ═════════════════════════════════════════════════════════════════════════════
#  SHAP COMPUTATION PER FOLD
# ═════════════════════════════════════════════════════════════════════════════

def compute_shap_for_fold(fold, df, feature_cols, fold_idx, total_folds):
    print(f"\n── Fold {fold.fold_number}/{total_folds} "
          f"[{fold.test_start.date()} → {fold.test_end.date()}] ──")

    from sklearn.preprocessing import StandardScaler

    date_level = df.index.get_level_values("date")
    tr_mask = date_level.isin(set(fold.model_train_dates))
    te_mask = date_level.isin(set(fold.test_dates))

    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        print(f"  [SKIP] Empty train or test mask")
        return None

    X_tr = df.loc[tr_mask, feature_cols].values.astype(np.float64)
    y_tr = df.loc[tr_mask, "target"].values.astype(np.float64)
    X_te = df.loc[te_mask, feature_cols].values.astype(np.float64)

    # Clean NaN / inf
    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr)
    X_te   = scaler.transform(X_te)

    print(f"  Train: {X_tr.shape} | Test: {X_te.shape}")

    # Train CatBoost
    try:
        from catboost import CatBoostClassifier
        vi  = int(0.85 * len(X_tr))
        cb  = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
            random_seed=42, thread_count=1, verbose=0,
            early_stopping_rounds=50, eval_metric="AUC"
        )
        cb.fit(X_tr[:vi], y_tr[:vi],
               eval_set=(X_tr[vi:], y_tr[vi:]), verbose=False)
        print(f"  CatBoost best_iter={cb.get_best_iteration()}")
        USE_CB = True
    except Exception as exc:
        print(f"  [WARN] CatBoost: {exc} — using RF")
        USE_CB = False

    if USE_CB and SHAP_AVAILABLE:
        try:
            bg  = X_tr[np.random.choice(len(X_tr), min(SHAP_SAMPLE_N, len(X_tr)), replace=False)]
            exp = shap.TreeExplainer(cb, data=bg, feature_perturbation="interventional")
            sv  = exp.shap_values(X_te)
            if isinstance(sv, list): sv = sv[1]
            print(f"  SHAP: {sv.shape}")
            return sv, X_te
        except Exception as exc:
            print(f"  [WARN] TreeExplainer: {exc} — using RF importances")

    # Fallback: RF importances
    from sklearn.ensemble import RandomForestClassifier
    rf  = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  min_samples_leaf=20, random_state=42, n_jobs=-1)
    model = cb if USE_CB else rf
    if not USE_CB:
        rf.fit(X_tr, y_tr)
        model = rf
    importance = (model.feature_importances_ if hasattr(model, "feature_importances_")
                  else np.ones(len(feature_cols)) / len(feature_cols))
    sv_proxy = importance.reshape(1, -1).repeat(len(X_te), axis=0)
    print(f"  RF importance proxy: {sv_proxy.shape}")
    return sv_proxy, X_te


# ═════════════════════════════════════════════════════════════════════════════
#  RANK STABILITY
# ═════════════════════════════════════════════════════════════════════════════

def compute_rank_stability(mean_abs_by_fold, feature_cols):
    fold_nums = sorted(mean_abs_by_fold.keys())
    n = len(fold_nums)
    mat = np.zeros((n, n))
    for i, fi in enumerate(fold_nums):
        for j, fj in enumerate(fold_nums):
            if i == j: mat[i,j] = 1.0
            else:
                ri = scipy.stats.rankdata(-mean_abs_by_fold[fi])
                rj = scipy.stats.rankdata(-mean_abs_by_fold[fj])
                rho, _ = scipy.stats.spearmanr(ri, rj)
                mat[i,j] = float(rho) if not np.isnan(rho) else 0.0
    df_corr = pd.DataFrame(mat,
                           index=[f"Fold {f}" for f in fold_nums],
                           columns=[f"Fold {f}" for f in fold_nums])
    mean_off = (mat.sum() - n) / max(n*(n-1), 1)
    print(f"\n[RANK STABILITY] Mean off-diagonal rho = {mean_off:.4f}")
    msg = ("LOW → noise-fitting confirmed at feature level"   if mean_off < 0.3
           else "MODERATE → some features consistently ranked" if mean_off < 0.6
           else "HIGH → consistent ranking despite absent IC")
    print(f"  → {msg}")
    return df_corr, mean_off


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def plot_beeswarm(sv, feature_cols, fold_number, out_path):
    mean_abs = np.abs(sv).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N_FEATURES]
    top_f    = [feature_cols[i] for i in top_idx]
    top_v    = mean_abs[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    clrs = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(top_f)))
    ax.barh(range(len(top_f)), top_v[::-1], color=clrs[::-1], alpha=0.85)
    ax.set_yticks(range(len(top_f)));  ax.set_yticklabels(top_f[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP|", fontsize=10)
    ax.set_title(f"Top {TOP_N_FEATURES} Features — Fold {fold_number}\n"
                 f"({len(sv):,} test observations)", fontsize=10, fontweight="bold")
    ax.axvline(top_v.mean(), color="red", ls="--", lw=1.5,
               label=f"Mean: {top_v.mean():.4f}")
    ax.legend(fontsize=8);  ax.grid(True, alpha=0.3, axis="x")

    ax2 = axes[1]
    cats = {
        "Momentum":   ["RSI_14","RSI_21","MACD_12_26","MACD_signal_9","MACD_hist","ROC_10","ROC_21","Williams_R"],
        "Bollinger":  ["BB_upper","BB_lower","BB_mid","BB_width","BB_pct_b","CCI_20"],
        "Volatility": ["ATR_14","ATR_21","rolling_vol_5d","rolling_vol_21d","rolling_vol_63d","HL_range_norm"],
        "Trend/MA":   ["EMA_9","EMA_21","EMA_50","EMA_200","SMA_50","SMA_200","price_to_SMA200","price_to_SMA50"],
        "Returns":    ["return_1d","return_2d","return_3d","return_5d","return_10d","return_21d"],
        "Volume":     ["OBV","OBV_EMA","volume_zscore_5d","volume_zscore_21d","MFI_14"],
        "Candle":     ["OC_body_norm","upper_shadow_ratio","lower_shadow_ratio","DPO_20"],
        "Directional":["stoch_K","stoch_D","ADX_14","DI_plus","DI_minus","VWAP_deviation"],
    }
    cat_v = {cat: float(np.abs(sv[:, [i for i,f in enumerate(feature_cols) if f in flist]]).mean())
             for cat, flist in cats.items()}
    tot = sum(cat_v.values()) + 1e-10
    ax2.pie([v/tot*100 for v in cat_v.values()],
            labels=list(cat_v.keys()),
            colors=["#2171b5","#238b45","#d94801","#6a3d9a","#e31a1c","#1d91c0","#fd8d3c","#807dba"],
            autopct="%1.1f%%", startangle=90, pctdistance=0.8,
            textprops={"fontsize": 8})
    ax2.set_title(f"Attribution by Category — Fold {fold_number}", fontsize=10, fontweight="bold")
    plt.suptitle("SHAP Feature Importance — When the Gate Stays Closed", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight");  plt.close()
    print(f"[SAVED] {out_path}")


def plot_heatmap(corr_df, mean_off, out_path):
    fig, ax = plt.subplots(figsize=(9, 7))
    mat  = corr_df.values;  n = len(corr_df)
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im   = ax.imshow(mat, cmap=plt.cm.RdBu_r, norm=norm, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman rho", shrink=0.8)
    ax.set_xticks(range(n));  ax.set_xticklabels(corr_df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n));  ax.set_yticklabels(corr_df.index, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(mat[i,j]) > 0.6 else "black")
    ax.set_title(f"SHAP Feature Rank Stability Across Folds\n"
                 f"Mean off-diagonal rho = {mean_off:.4f} "
                 f"({'low → noise-fitting' if mean_off < 0.3 else 'moderate/high → some consistency'})",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight");  plt.close()
    print(f"[SAVED] {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("ROBUSTNESS CHECK 2: SHAP FEATURE IMPORTANCE [FIXED v2]")
    print("=" * 60)

    df, feature_cols = load_pipeline_data()
    assert "target" in df.columns, \
        f"'target' missing after load. Columns: {list(df.columns)[:10]}"
    print(f"\n[DATA] {len(df):,} rows | {len(feature_cols)} features | "
          f"{df.index.get_level_values('ticker').nunique()} tickers")

    # Folds
    all_folds = (generate_folds(df) if MODULES_AVAILABLE
                 else generate_folds_self(df))
    selected  = all_folds[-N_SHAP_FOLDS:]
    print(f"[FOLDS] {len(all_folds)} total → SHAP on folds "
          f"{selected[0].fold_number}–{selected[-1].fold_number}")

    # Compute SHAP
    mean_abs_by_fold: Dict[int, np.ndarray] = {}
    shap_last = X_last = None

    for fold in selected:
        res = compute_shap_for_fold(fold, df, feature_cols,
                                    fold.fold_number, len(all_folds))
        if res is None: continue
        sv, X_te = res
        mean_abs_by_fold[fold.fold_number] = np.abs(sv).mean(axis=0)
        if fold is selected[-1]:
            shap_last = sv;  X_last = X_te

    if not mean_abs_by_fold:
        print("[ERROR] No SHAP values computed.")
        return

    # Save CSVs
    shap_df = pd.DataFrame({f"fold_{k}": v for k, v in mean_abs_by_fold.items()},
                            index=feature_cols)
    shap_df["mean_across_folds"] = shap_df.mean(axis=1)
    shap_df = shap_df.sort_values("mean_across_folds", ascending=False)
    shap_df.to_csv(OUT_DIR / "shap_mean_abs_by_fold.csv")
    print(f"\n[SAVED] {OUT_DIR}/shap_mean_abs_by_fold.csv")
    print(f"\n[TOP 15 FEATURES]")
    print(shap_df["mean_across_folds"].head(15).to_string())

    corr_df, mean_off = compute_rank_stability(mean_abs_by_fold, feature_cols)
    corr_df.to_csv(OUT_DIR / "shap_fold_rank_correlation.csv")
    print(f"[SAVED] {OUT_DIR}/shap_fold_rank_correlation.csv")

    if shap_last is not None:
        plot_beeswarm(shap_last, feature_cols, selected[-1].fold_number,
                      OUT_DIR / "fig_shap_beeswarm.png")
    plot_heatmap(corr_df, mean_off, OUT_DIR / "fig_shap_rank_stability.png")

    top5 = shap_df["mean_across_folds"].head(5).index.tolist()
    print(f"\n[DONE] Top-5 features: {', '.join(top5)}")
    print(f"       Mean rank stability rho: {mean_off:.4f}")
    print(f"       See results/robustness/shap/ for CSV outputs to use in manuscript.")


if __name__ == "__main__":
    main()