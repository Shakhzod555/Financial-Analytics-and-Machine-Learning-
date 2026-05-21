"""
Full-paper replication of Christensen, Siggaard & Veliyev (2023, JFE).
Implements the M_HAR contest at the scope of the paper's Section 1:

  - 22 models: 6 HAR-family, 5 regularised linear, 3 trees, 8 NN
  - Diebold-Mariano (HLN-corrected) and Model Confidence Set (Hansen-Lunde-Nason)
  - Accumulated Local Effect variable importance (Apley-Zhu 2020)
  - Value-at-Risk forecast via filtered historical simulation with Kupiec
    (1995) and Christoffersen (1998) coverage tests
  - Train/val/test split = 70/10/20 chronological, horizons h in {1, 5, 22}

Skips by design (paper components not feasible or out-of-scope here):
  - M_ALL information set (option-implied volatility, EPU, ADS, etc. not in data)
  - Rolling-window day-by-day refit (fixed window only; paper does both)
  - Short-training-set robustness (Appendix A.1)

Outputs:
  results_full.csv        — 22 models x 3 tickers x 3 horizons
  mcs_results.csv         — MCS membership at 75% and 90% per (ticker, horizon)
  ale_importance.csv      — variable importance from ALE
  var_results.csv         — VaR loss, Kupiec p, Christoffersen p for each model
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

N_JOBS = max(1, os.cpu_count() or 1)
INNER_JOBS = 1     # avoid oversubscription when outer loop is parallel

DATA_DIR = Path(".")
TICKERS = ["AAPL", "AMZN", "JPM"]
RV_FREQ_MIN = 5
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
HORIZONS = {"h=1": 1, "h=5": 5, "h=22": 22}
HOR_LABEL = {"h=1": "1-day", "h=5": "1-week", "h=22": "1-month"}

# Paper's 22-model list (Table 2 columns)
MODELS = [
    "HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ",
    "Ridge", "Lasso", "ElasticNet", "AdaLasso", "PostLasso",
    "Bagging", "RandomForest", "GBoost",
    "NN1_1", "NN1_10", "NN2_1", "NN2_10",
    "NN3_1", "NN3_10", "NN4_1", "NN4_10",
]
SHORT = {
    "HAR": "HAR", "HAR-X": "HAR-X", "LogHAR": "LogHAR", "LevHAR": "LevHAR",
    "SHAR": "SHAR", "HARQ": "HARQ",
    "Ridge": "RR", "Lasso": "LA", "ElasticNet": "EN",
    "AdaLasso": "A-LA", "PostLasso": "P-LA",
    "Bagging": "BG", "RandomForest": "RF", "GBoost": "GB",
    "NN1_1": "NN1^1", "NN1_10": "NN1^10",
    "NN2_1": "NN2^1", "NN2_10": "NN2^10",
    "NN3_1": "NN3^1", "NN3_10": "NN3^10",
    "NN4_1": "NN4^1", "NN4_10": "NN4^10",
}

NN_ARCHS = {"NN1": (2,), "NN2": (4, 2), "NN3": (8, 4, 2), "NN4": (16, 8, 4, 2)}
NN_SEEDS_ENSEMBLE = 100  # paper-exact: top-10 of 100 seeds for "_10" ensemble variant
NN_TOP_K = 10
NN_MAX_ITER = 500

HAR_FEATS = ["RVD", "RVW", "RVM"]
LEVHAR_FEATS = ["RVD", "RVW", "RVM", "rN_D", "rN_W", "rN_M"]
SHAR_FEATS = ["RVp_d", "RVn_d", "RVW", "RVM"]

# M_ALL covariates (8 macro features available from public data; IV is
# proxied by VIX so we don't include a separate IV column).
MACRO_FEATS = ["EA", "VIX", "HSI", "ADS", "US3M", "EPU", "M1W", "DVOL"]
M_ALL_FEATS    = HAR_FEATS    + MACRO_FEATS                # 3 + 8 = 11
LEVHAR_M_ALL   = LEVHAR_FEATS + MACRO_FEATS                # 6 + 8 = 14
SHAR_M_ALL     = SHAR_FEATS   + MACRO_FEATS                # 4 + 8 = 12

MACRO_PATH = Path("data/macro.csv")
INFO_SETS = ("M_HAR", "M_ALL")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_minute_bars(path):
    df = pd.read_csv(path, header=None,
                     names=["date", "time", "open", "high", "low", "close", "vol"],
                     dtype={"date": str, "time": str})
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"],
                              format="%m/%d/%Y %H:%M")
    return df.sort_values("dt").reset_index(drop=True)


def compute_features(df, freq=5):
    df_idx = df.set_index("dt")
    closes = (df_idx["close"]
              .resample(f"{freq}min", label="right", closed="right")
              .last().dropna())
    bars = closes.to_frame("c")
    bars["day"] = bars.index.date
    bars["r"] = np.log(bars["c"]).groupby(bars["day"]).diff()
    intra = bars.dropna(subset=["r"])
    g = intra.groupby("day")["r"]
    out = pd.DataFrame({
        "RV":  g.apply(lambda r: (r ** 2).sum()),
        "RQ":  g.apply(lambda r: (len(r) / 3.0) * (r ** 4).sum()),
        "RVp": g.apply(lambda r: (r[r > 0] ** 2).sum()),
        "RVn": g.apply(lambda r: (r[r < 0] ** 2).sum()),
        "r_daily": g.sum(),
        "n": g.size(),
    })
    out = out[out["n"] >= 10]
    out.index = pd.to_datetime(out.index)
    # Daily dollar volume (for M_ALL's DVOL feature) — from 1-min bars
    df_idx = df_idx.copy()
    df_idx["dollar_vol_min"] = df_idx["close"] * df_idx["vol"]
    daily_dv = df_idx.groupby(df_idx.index.date)["dollar_vol_min"].sum()
    daily_dv.index = pd.to_datetime(daily_dv.index)
    out["DOLLAR_VOL"] = daily_dv.reindex(out.index).fillna(method="ffill")
    return out


def build_har_design(feat, h=1, macro=None, ticker=None, dollar_volume=None):
    """HAR design matrix. If `macro` is a DataFrame indexed by date and
    `ticker` is provided, also append the M_ALL covariates (VIX, HSI, ADS,
    US3M, EPU, EA, M1W, DVOL).  All extras are lagged by 1 day.
    """
    df = feat.copy()
    df["RVD"] = df["RV"].shift(1)
    df["RVW"] = df["RV"].shift(1).rolling(5).mean()
    df["RVM"] = df["RV"].shift(1).rolling(22).mean()
    df["RQ_d"] = np.sqrt(df["RQ"]).shift(1)
    df["RVp_d"] = df["RVp"].shift(1)
    df["RVn_d"] = df["RVn"].shift(1)
    rn = np.minimum(0, df["r_daily"])
    df["rN_D"] = rn.shift(1)
    df["rN_W"] = rn.shift(1).rolling(5).mean()
    df["rN_M"] = rn.shift(1).rolling(22).mean()
    # Target = mean(RV_t, ..., RV_{t+h-1}); features at row t use info up to t-1
    # (matches paper Section 1.1: RV_t = beta0 + beta'·Z_{t-1} + u_t, generalised
    # to multi-step by averaging RV over [t, t+h-1]).
    df["y"] = df["RV"].rolling(h).mean().shift(-(h - 1))

    if macro is not None and ticker is not None:
        # Locally computed M_ALL features
        df["M1W"] = df["r_daily"].shift(1).rolling(5).sum()
        if dollar_volume is not None:
            dv = dollar_volume.reindex(df.index)
            df["DVOL"] = np.log(dv).diff().shift(1)
        else:
            df["DVOL"] = np.log(df["RV"]).diff().shift(1) * 0  # zero fallback
        # External M_ALL features (aligned and forward-filled across non-trading days)
        macro_aligned = macro.reindex(df.index).ffill(limit=5)
        df["VIX"]  = macro_aligned["VIX"].shift(1)
        df["HSI"]  = macro_aligned["HSI"].shift(1)
        df["ADS"]  = macro_aligned["ADS"].shift(1)
        # Paper first-differences US3M
        df["US3M"] = macro_aligned["US3M"].diff().shift(1)
        df["EPU"]  = macro_aligned["EPU"].shift(1)
        # Earnings dummy is contemporaneous (announcement on the forecast day)
        ea_col = f"EA_{ticker}"
        df["EA"] = macro_aligned[ea_col].astype(float) if ea_col in macro_aligned.columns else 0.0
    return df.dropna()


def _harq_design(df):
    return pd.DataFrame({
        "RVD": df["RVD"],
        "RVD_RQ": df["RVD"] * df["RQ_d"],
        "RVW": df["RVW"],
        "RVM": df["RVM"],
    }, index=df.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tune_alpha_holdout(estimator_fn, alpha_grid, X_tr, y_tr, X_va, y_va):
    """Pick alpha by held-out validation MSE."""
    best_a, best_mse = alpha_grid[0], np.inf
    for a in alpha_grid:
        m = estimator_fn(a)
        m.fit(X_tr, y_tr)
        mse = mean_squared_error(y_va, m.predict(X_va))
        if mse < best_mse:
            best_a, best_mse = a, mse
    return best_a


def _scale(train, val, test, feats):
    sc = StandardScaler().fit(train[feats])
    return (sc.transform(train[feats]),
            sc.transform(val[feats]),
            sc.transform(test[feats]),
            sc)


def _refit_scaled(tv, test, feats):
    sc = StandardScaler().fit(tv[feats])
    return sc.transform(tv[feats]), sc.transform(test[feats]), sc


def _nn_arch_predict(arch, train, val, test, n_seeds, top_k, feats=HAR_FEATS):
    """Run n_seeds NN fits with the given architecture; return
    (single_test_pred, ensemble_test_pred, single_train_pred, ensemble_train_pred).
    Single = seed 0 only.  Ensemble = mean of top-k by validation MSE.
    """
    sx = StandardScaler().fit(train[feats])
    sy = StandardScaler().fit(train["y"].values.reshape(-1, 1))
    Xtr = sx.transform(train[feats])
    Xva = sx.transform(val[feats])
    Xte = sx.transform(test[feats])
    ytr_s = sy.transform(train["y"].values.reshape(-1, 1)).ravel()

    val_mses, te_preds, tr_preds = [], [], []
    inv = lambda p: sy.inverse_transform(p.reshape(-1, 1)).ravel()
    for seed in range(n_seeds):
        net = MLPRegressor(hidden_layer_sizes=arch, activation="relu",
                           solver="adam", learning_rate_init=0.001,
                           max_iter=NN_MAX_ITER, random_state=seed)
        net.fit(Xtr, ytr_s)
        vp = inv(net.predict(Xva))
        tp = inv(net.predict(Xte))
        ip = inv(net.predict(Xtr))
        val_mses.append(mean_squared_error(val["y"], vp))
        te_preds.append(tp); tr_preds.append(ip)
    val_mses = np.array(val_mses)
    top = np.argsort(val_mses)[:top_k]
    return (te_preds[0],
            np.mean([te_preds[i] for i in top], axis=0),
            tr_preds[0],
            np.mean([tr_preds[i] for i in top], axis=0))


# ---------------------------------------------------------------------------
# 22-model fit_predict
# ---------------------------------------------------------------------------
def _features_for(name, info_set):
    """Resolve the feature column list for a given (model, info_set) combo."""
    if info_set == "M_HAR":
        if name in ("LevHAR",):
            return LEVHAR_FEATS
        if name in ("SHAR",):
            return SHAR_FEATS
        return HAR_FEATS
    # M_ALL
    if name in ("LevHAR",):
        return LEVHAR_M_ALL
    if name in ("SHAR",):
        return SHAR_M_ALL
    return M_ALL_FEATS


def fit_predict(name, train, test, val=None, nn_cache=None, info_set="M_HAR"):
    """Return (test_pred, in_sample_pred).  NN cache: dict keyed by arch name to
    avoid refitting the same architecture twice (NN_x^1 and NN_x^10 share seeds).
    `info_set` selects feature set: "M_HAR" or "M_ALL".
    """
    y_tr = train["y"].values
    feats = _features_for(name, info_set)
    use_mall = (info_set == "M_ALL")

    # ----- HAR family (OLS) -----
    if name == "HAR":
        m = LinearRegression().fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "HAR-X":
        # On M_HAR, HAR-X coincides with HAR; on M_ALL it adds the macro extras.
        m = LinearRegression().fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "LogHAR":
        # Log-transform RV-based features (and VIX on M_ALL); leave macros raw.
        log_cols = [c for c in ("RVD", "RVW", "RVM", "VIX") if c in feats]
        other_cols = [c for c in feats if c not in log_cols]
        Xtr = train[feats].copy()
        Xte = test[feats].copy()
        for c in log_cols:
            Xtr[c] = np.log(np.maximum(Xtr[c], 1e-12))
            Xte[c] = np.log(np.maximum(Xte[c], 1e-12))
        ytr = np.log(y_tr)
        m = LinearRegression().fit(Xtr.values, ytr)
        sig = np.var(ytr - m.predict(Xtr.values), ddof=1)
        return (np.exp(m.predict(Xte.values) + 0.5 * sig),
                np.exp(m.predict(Xtr.values) + 0.5 * sig))

    if name == "LevHAR":
        m = LinearRegression().fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "SHAR":
        m = LinearRegression().fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "HARQ":
        Xtr_base = _harq_design(train); Xte_base = _harq_design(test)
        if use_mall:
            # Append macro covariates after the HARQ interaction columns
            extras = [c for c in MACRO_FEATS if c in train.columns]
            Xtr = pd.concat([Xtr_base, train[extras]], axis=1).values
            Xte = pd.concat([Xte_base, test[extras]], axis=1).values
        else:
            Xtr = Xtr_base.values; Xte = Xte_base.values
        m = LinearRegression().fit(Xtr, y_tr)
        return m.predict(Xte), m.predict(Xtr)

    alpha_grid = np.logspace(-5, 2, 100)
    if val is not None:
        y_va = val["y"].values

    # ----- Regularised linear -----
    if name == "Ridge":
        assert val is not None, "Ridge needs val set"
        Xtr, Xva, Xte, _ = _scale(train, val, test, feats)
        a = _tune_alpha_holdout(lambda a: Ridge(alpha=a, random_state=42),
                                alpha_grid, Xtr, y_tr, Xva, y_va)
        tv = pd.concat([train, val])
        Xtv, Xte2, _ = _refit_scaled(tv, test, feats)
        m = Ridge(alpha=a, random_state=42).fit(Xtv, tv["y"].values)
        return m.predict(Xte2), m.predict(Xtv)

    if name == "Lasso":
        assert val is not None, "Lasso needs val set"
        Xtr, Xva, Xte, _ = _scale(train, val, test, feats)
        a = _tune_alpha_holdout(
            lambda a: Lasso(alpha=a, max_iter=10000, random_state=42),
            alpha_grid, Xtr, y_tr, Xva, y_va)
        tv = pd.concat([train, val])
        Xtv, Xte2, _ = _refit_scaled(tv, test, feats)
        m = Lasso(alpha=a, max_iter=10000, random_state=42).fit(Xtv, tv["y"].values)
        return m.predict(Xte2), m.predict(Xtv)

    if name == "ElasticNet":
        assert val is not None, "ElasticNet needs val set"
        sc = StandardScaler().fit(train[feats])
        Xtr = sc.transform(train[feats])
        Xva = sc.transform(val[feats])
        y_va = val["y"].values
        l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        best_mse, best = np.inf, (alpha_grid[0], l1_grid[0])
        for l1r in l1_grid:
            for a in alpha_grid:
                m = ElasticNet(alpha=a, l1_ratio=l1r,
                               max_iter=10000, random_state=42)
                m.fit(Xtr, y_tr)
                mse = mean_squared_error(y_va, m.predict(Xva))
                if mse < best_mse:
                    best_mse, best = mse, (a, l1r)
        tv = pd.concat([train, val])
        Xtv, Xte2, _ = _refit_scaled(tv, test, feats)
        m = ElasticNet(alpha=best[0], l1_ratio=best[1],
                       max_iter=10000, random_state=42).fit(Xtv, tv["y"].values)
        return m.predict(Xte2), m.predict(Xtv)

    if name == "AdaLasso":
        assert val is not None, "AdaLasso needs val set"
        Xtr, Xva, Xte, _ = _scale(train, val, test, feats)
        ols = LinearRegression().fit(Xtr, y_tr)
        w = 1.0 / (np.abs(ols.coef_) + 1e-6)
        Xtr_w = Xtr / w
        Xva_w = Xva / w
        a = _tune_alpha_holdout(
            lambda a: Lasso(alpha=a, max_iter=10000, random_state=42),
            alpha_grid, Xtr_w, y_tr, Xva_w, y_va)
        tv = pd.concat([train, val])
        Xtv, Xte2, _ = _refit_scaled(tv, test, feats)
        ols2 = LinearRegression().fit(Xtv, tv["y"].values)
        w2 = 1.0 / (np.abs(ols2.coef_) + 1e-6)
        Xtv_w = Xtv / w2; Xte2_w = Xte2 / w2
        m = Lasso(alpha=a, max_iter=10000, random_state=42).fit(Xtv_w, tv["y"].values)
        return m.predict(Xte2_w), m.predict(Xtv_w)

    if name == "PostLasso":
        assert val is not None, "PostLasso needs val set"
        Xtr, Xva, Xte, _ = _scale(train, val, test, feats)
        a = _tune_alpha_holdout(
            lambda a: Lasso(alpha=a, max_iter=10000, random_state=42),
            alpha_grid, Xtr, y_tr, Xva, y_va)
        tv = pd.concat([train, val])
        Xtv, Xte2, _ = _refit_scaled(tv, test, feats)
        first = Lasso(alpha=a, max_iter=10000, random_state=42).fit(Xtv, tv["y"].values)
        sel = np.where(np.abs(first.coef_) > 1e-10)[0]
        if len(sel) == 0:
            return np.full(len(test), tv["y"].mean()), np.full(len(tv), tv["y"].mean())
        m = LinearRegression().fit(Xtv[:, sel], tv["y"].values)
        return m.predict(Xte2[:, sel]), m.predict(Xtv[:, sel])

    # ----- Tree ensembles -----
    if name == "Bagging":
        base = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
        m = BaggingRegressor(estimator=base, n_estimators=500,
                             n_jobs=INNER_JOBS, random_state=42)
        m.fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "RandomForest":
        m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                  max_features="sqrt",
                                  n_jobs=INNER_JOBS, random_state=42)
        m.fit(train[feats], y_tr)
        return m.predict(test[feats]), m.predict(train[feats])

    if name == "GBoost":
        assert val is not None, "GBoost needs val set"
        y_va = val["y"].values
        gb_grid = [(n, lr, d) for n in (100, 300, 500)
                                  for lr in (0.01, 0.05, 0.1)
                                  for d in (3, 5)]
        best_mse, best = np.inf, gb_grid[0]
        for n_est, lr, depth in gb_grid:
            m = GradientBoostingRegressor(
                n_estimators=n_est, learning_rate=lr, max_depth=depth,
                min_samples_leaf=5, subsample=0.8, random_state=42)
            m.fit(train[feats], y_tr)
            mse = mean_squared_error(y_va, m.predict(val[feats]))
            if mse < best_mse:
                best_mse, best = mse, (n_est, lr, depth)
        tv = pd.concat([train, val])
        final = GradientBoostingRegressor(
            n_estimators=best[0], learning_rate=best[1], max_depth=best[2],
            min_samples_leaf=5, subsample=0.8, random_state=42)
        final.fit(tv[feats], tv["y"].values)
        return final.predict(test[feats]), final.predict(tv[feats])

    # ----- Neural networks -----
    if name.startswith("NN"):
        assert val is not None, f"{name} needs val set"
        arch_name = name.split("_")[0]      # "NN1" .. "NN4"
        is_ensemble = name.endswith("_10")
        arch = NN_ARCHS[arch_name]
        # Cache key includes info_set so M_HAR and M_ALL caches don't collide.
        cache_key = f"{arch_name}__{info_set}"
        if nn_cache is not None and cache_key in nn_cache:
            single_te, ens_te, single_tr, ens_tr = nn_cache[cache_key]
        else:
            single_te, ens_te, single_tr, ens_tr = _nn_arch_predict(
                arch, train, val, test,
                n_seeds=NN_SEEDS_ENSEMBLE, top_k=NN_TOP_K, feats=feats)
            if nn_cache is not None:
                nn_cache[cache_key] = (single_te, ens_te, single_tr, ens_tr)
        if is_ensemble:
            return ens_te, ens_tr
        return single_te, single_tr

    raise ValueError(f"Unknown model: {name}")


def insanity_filter(pred, train_y):
    return np.clip(pred, max(train_y.min(), 1e-12), train_y.max() * 2.0)


def rolling_har_predict(name, data, cut, h, info_set="M_HAR"):
    """Daily-rolling re-estimation for the HAR family. At each test day t, fit
    on data[:t] then predict day t. `info_set` selects M_HAR or M_ALL features.
    Returns (test_pred, in_sample_pred_on_initial_tv).
    """
    feats = _features_for(name, info_set)
    use_mall = (info_set == "M_ALL")
    log_cols = [c for c in ("RVD", "RVW", "RVM", "VIX") if c in feats]
    test_idx = np.arange(cut, len(data))
    preds = np.zeros(len(test_idx))
    for k, t_idx in enumerate(test_idx):
        train_block = data.iloc[:t_idx]
        test_row = data.iloc[t_idx:t_idx + 1]
        y_tr = train_block["y"].values
        if name in ("HAR", "HAR-X"):
            m = LinearRegression().fit(train_block[feats], y_tr)
            preds[k] = m.predict(test_row[feats])[0]
        elif name == "LogHAR":
            Xtr = train_block[feats].copy()
            Xte = test_row[feats].copy()
            for c in log_cols:
                Xtr[c] = np.log(np.maximum(Xtr[c], 1e-12))
                Xte[c] = np.log(np.maximum(Xte[c], 1e-12))
            ytr = np.log(y_tr)
            m = LinearRegression().fit(Xtr.values, ytr)
            sig = np.var(ytr - m.predict(Xtr.values), ddof=1)
            preds[k] = np.exp(m.predict(Xte.values)[0] + 0.5 * sig)
        elif name == "LevHAR":
            m = LinearRegression().fit(train_block[feats], y_tr)
            preds[k] = m.predict(test_row[feats])[0]
        elif name == "SHAR":
            m = LinearRegression().fit(train_block[feats], y_tr)
            preds[k] = m.predict(test_row[feats])[0]
        elif name == "HARQ":
            Xtr_base = _harq_design(train_block)
            Xte_base = _harq_design(test_row)
            if use_mall:
                extras = [c for c in MACRO_FEATS if c in train_block.columns]
                Xtr = pd.concat([Xtr_base, train_block[extras]], axis=1).values
                Xte = pd.concat([Xte_base, test_row[extras]], axis=1).values
            else:
                Xtr = Xtr_base.values; Xte = Xte_base.values
            m = LinearRegression().fit(Xtr, y_tr)
            preds[k] = m.predict(Xte)[0]
        else:
            raise ValueError(f"rolling_har_predict: unknown model {name}")
    # In-sample fit on initial train+val for ACF / diagnostic use
    tv = data.iloc[:cut]
    y_tv = tv["y"].values
    if name == "LogHAR":
        Xtv = tv[feats].copy()
        for c in log_cols:
            Xtv[c] = np.log(np.maximum(Xtv[c], 1e-12))
        ytv_log = np.log(y_tv)
        m = LinearRegression().fit(Xtv.values, ytv_log)
        sig = np.var(ytv_log - m.predict(Xtv.values), ddof=1)
        in_sample = np.exp(m.predict(Xtv.values) + 0.5 * sig)
    elif name == "HARQ":
        Xtv_base = _harq_design(tv)
        if use_mall:
            extras = [c for c in MACRO_FEATS if c in tv.columns]
            Xtv = pd.concat([Xtv_base, tv[extras]], axis=1).values
        else:
            Xtv = Xtv_base.values
        in_sample = LinearRegression().fit(Xtv, y_tv).predict(Xtv)
    else:
        in_sample = LinearRegression().fit(tv[feats], y_tv).predict(tv[feats])
    return preds, in_sample


# ---------------------------------------------------------------------------
# Diebold-Mariano + Model Confidence Set
# ---------------------------------------------------------------------------
def _dm_from_diff(d, h=1):
    T = len(d)
    if T < 5:
        return np.nan, np.nan
    dbar = d.mean()
    g0 = np.mean((d - dbar) ** 2)
    var = g0
    for k in range(1, h):
        if k >= T:
            break
        gk = np.mean((d[:-k] - dbar) * (d[k:] - dbar))
        var += 2 * (1 - k / h) * gk
    var = max(var, 1e-20)
    DM = dbar / np.sqrt(var / T)
    corr = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if corr <= 0:
        return np.nan, np.nan
    DM_adj = DM * np.sqrt(corr)
    p_one = 1 - stats.norm.cdf(DM_adj)
    return DM_adj, p_one


def dm_test(y, p_bench, p_alt, h=1):
    return _dm_from_diff((y - p_bench) ** 2 - (y - p_alt) ** 2, h=h)


def model_confidence_set(loss_matrix, alpha=0.10, B=1000, h=1):
    """Hansen-Lunde-Nason (2011) MCS with moving-block (circular) bootstrap.
    Block length follows the Politis-White (2004) rough cube-root rule
    L ≈ T^(1/3), with a floor that respects the forecast-overlap horizon h.
    `loss_matrix` shape (T, M); returns list of surviving model indices.
    """
    from arch.bootstrap import MCS
    T = len(loss_matrix)
    bs_window = max(int(np.ceil(T ** (1.0 / 3.0))), h, 2)
    mcs = MCS(loss_matrix, size=alpha,
              reps=B, block_size=bs_window,
              bootstrap="circular",
              method="max", seed=42)
    mcs.compute()
    return list(mcs.included)


# ---------------------------------------------------------------------------
# Accumulated Local Effect (Apley-Zhu 2020)
# ---------------------------------------------------------------------------
def ale_1d(predict_fn, X, feature_idx, n_quantiles=20):
    """Compute centred 1-D ALE on a numpy array `X` for a single feature index.
    `predict_fn` accepts an (n, p) array and returns (n,)."""
    X = np.asarray(X)
    n, p = X.shape
    z = np.quantile(X[:, feature_idx], np.linspace(0, 1, n_quantiles + 1))
    z = np.unique(z)
    K = len(z) - 1
    f = np.zeros(K)
    for k in range(K):
        lo, hi = z[k], z[k + 1]
        if k == K - 1:
            mask = (X[:, feature_idx] >= lo) & (X[:, feature_idx] <= hi)
        else:
            mask = (X[:, feature_idx] >= lo) & (X[:, feature_idx] < hi)
        if mask.sum() < 2:
            continue
        Xl = X[mask].copy(); Xl[:, feature_idx] = lo
        Xh = X[mask].copy(); Xh[:, feature_idx] = hi
        f[k] = np.mean(predict_fn(Xh) - predict_fn(Xl))
    cum = np.cumsum(f)
    # centre: subtract weighted mean over the empirical distribution of X[:,j]
    edges = z
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = np.histogram(X[:, feature_idx], bins=edges)[0]
    if weights.sum() == 0:
        return centres, cum
    cum -= np.average(cum, weights=weights)
    return centres, cum


def ale_importance(centres_list, ale_list):
    """Importance = stdev of the centred ALE over the empirical distribution of
    the feature (we approximate by stdev of equispaced ALE values).
    """
    return [np.std(a) for a in ale_list]


# ---------------------------------------------------------------------------
# VaR application (filtered historical simulation + Kupiec + Christoffersen)
# ---------------------------------------------------------------------------
def fhs_var_forecast(sigma_pred, r_history, alpha=0.05):
    """One-day-ahead VaR at quantile `alpha` from filtered historical simulation.
    For each test day t, scale historical standardised residuals by sigma_pred_t.
    Returns array of VaR_t (negative number; VaR^a s.t. P(r_{t+1} < VaR) = a).
    """
    sigma_pred = np.asarray(sigma_pred)
    r_history = np.asarray(r_history)
    sigma_history = pd.Series(r_history).rolling(22).std().bfill().values
    z = r_history / np.where(sigma_history > 0, sigma_history, np.nan)
    z = z[~np.isnan(z)]
    quant = np.quantile(z, alpha)
    return sigma_pred * quant


def kupiec_lr(hits, alpha):
    """Likelihood-ratio test for unconditional coverage."""
    T = len(hits)
    x = int(hits.sum())
    if x == 0 or x == T:
        return np.nan, np.nan
    p_hat = x / T
    LR_uc = -2 * (
        x * np.log(alpha) + (T - x) * np.log(1 - alpha)
        - x * np.log(p_hat) - (T - x) * np.log(1 - p_hat)
    )
    p_value = 1 - stats.chi2.cdf(LR_uc, df=1)
    return LR_uc, p_value


def christoffersen_lr(hits):
    """Likelihood-ratio test for independence (Markov-chain transitions)."""
    h = np.asarray(hits).astype(int)
    n00 = ((h[:-1] == 0) & (h[1:] == 0)).sum()
    n01 = ((h[:-1] == 0) & (h[1:] == 1)).sum()
    n10 = ((h[:-1] == 1) & (h[1:] == 0)).sum()
    n11 = ((h[:-1] == 1) & (h[1:] == 1)).sum()
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan, np.nan
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p_unc = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    eps = 1e-20
    def safe_log(x): return np.log(x + eps)
    LL_null = (n01 + n11) * safe_log(p_unc) + (n00 + n10) * safe_log(1 - p_unc)
    LL_alt = (n01 * safe_log(p01) + n00 * safe_log(1 - p01)
              + n11 * safe_log(p11) + n10 * safe_log(1 - p11))
    LR_ind = -2 * (LL_null - LL_alt)
    p_value = 1 - stats.chi2.cdf(LR_ind, df=1)
    return LR_ind, p_value


def quantile_loss(r_true, var_pred, alpha=0.05):
    """Asymmetric tick loss for quantile forecasts (Koenker-Bassett)."""
    d = (r_true < var_pred).astype(int)
    return ((alpha - d) * (r_true - var_pred)).mean()


# ---------------------------------------------------------------------------
# Per-cell worker (runs in parallel across (ticker, horizon, info_set) combos)
# ---------------------------------------------------------------------------
def process_cell(t, h_lab, h, feat_t, info_set="M_HAR", macro=None):
    """Process one (ticker, horizon, info_set) cell: fit 22 models, compute
    MSE/DM, return rows, loss matrix, forecasts. Stateless for joblib.
    """
    if info_set == "M_ALL":
        dv = feat_t["DOLLAR_VOL"] if "DOLLAR_VOL" in feat_t.columns else None
        data = build_har_design(feat_t, h=h, macro=macro, ticker=t,
                                 dollar_volume=dv)
    else:
        data = build_har_design(feat_t, h=h)
    n = len(data)
    n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    train = data.iloc[:n_tr]
    val = data.iloc[n_tr:n_tr + n_va]
    test = data.iloc[n_tr + n_va:]
    tv = pd.concat([train, val])
    y_test = test["y"].values

    preds = {}
    nn_cache = {}
    cut = n_tr + n_va
    for m in MODELS:
        if m in ("HAR", "HAR-X", "LogHAR", "LevHAR", "SHAR", "HARQ"):
            pte, _ = rolling_har_predict(m, data, cut=cut, h=h,
                                          info_set=info_set)
        elif m in ("Bagging", "RandomForest"):
            pte, _ = fit_predict(m, tv, test, info_set=info_set)
        else:
            pte, _ = fit_predict(m, train, test, val=val,
                                  nn_cache=nn_cache, info_set=info_set)
        pte = insanity_filter(pte, tv["y"])
        preds[m] = pte

    mse_har = mean_squared_error(y_test, preds["HAR"])
    T_test = len(y_test)
    loss_mat = np.zeros((T_test, len(MODELS)))
    rows = []
    for j, m in enumerate(MODELS):
        mse = mean_squared_error(y_test, preds[m])
        if m == "HAR":
            DM, pv = np.nan, np.nan
        else:
            DM, pv = dm_test(y_test, preds["HAR"], preds[m], h=h)
        rows.append({"ticker": t, "horizon": h_lab, "model": m,
                     "info_set": info_set,
                     "mse": mse, "rel_mse": mse / mse_har,
                     "DM": DM, "p_value": pv})
        loss_mat[:, j] = (y_test - preds[m]) ** 2

    forecasts = {"y": y_test, "p": preds, "dates": test.index}
    log_msg = (f"  [{t} | {h_lab} | {info_set}] done — train={len(train)} "
               f"val={len(val)} test={len(test)}")
    return t, h_lab, info_set, rows, loss_mat, forecasts, log_msg


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run():
    print("=" * 70)
    print(f"FULL-PAPER REPLICATION: 22-model contest, M_HAR + M_ALL  "
          f"[parallel: n_jobs={N_JOBS}]")
    print("=" * 70)

    feat = {}
    rv_panel = {}
    daily_ret = {}
    print("\nLoading 1-min data and computing 5-min RV + daily volume...")
    for t in TICKERS:
        df = load_minute_bars(DATA_DIR / f"{t}.txt")
        f = compute_features(df, freq=RV_FREQ_MIN)
        feat[t] = f
        rv_panel[t] = f["RV"]
        daily_ret[t] = f["r_daily"]
        print(f"  {t}: {len(f)} sessions; mean RV={f['RV'].mean():.3e}")

    # Load macro covariates (for M_ALL)
    if MACRO_PATH.exists():
        macro = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)
        macro.index = pd.to_datetime(macro.index).normalize()
        print(f"\nLoaded macro: {macro.shape}, "
              f"{macro.index.min().date()} → {macro.index.max().date()}")
    else:
        macro = None
        print("\n(no data/macro.csv — running M_HAR only)")

    info_sets_to_run = INFO_SETS if macro is not None else ("M_HAR",)
    cells = [(t, h_lab, h, info_set)
             for info_set in info_sets_to_run
             for t in TICKERS
             for h_lab, h in HORIZONS.items()]
    print(f"\nRunning {len(cells)} cells in parallel on {N_JOBS} cores...")
    results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(process_cell)(t, h_lab, h, feat[t],
                              info_set=info_set, macro=macro)
        for (t, h_lab, h, info_set) in cells
    )

    rows = []
    losses_panel = {}
    forecasts_panel = {iset: {t: {} for t in TICKERS}
                       for iset in info_sets_to_run}
    for t, h_lab, info_set, cell_rows, loss_mat, forecasts, log_msg in results:
        rows.extend(cell_rows)
        losses_panel[(t, h_lab, info_set)] = loss_mat
        forecasts_panel[info_set][t][h_lab] = forecasts
        print(log_msg)

    results_df = pd.DataFrame(rows)
    results_df.to_csv("results_full.csv", index=False)
    print(f"\nResults saved: results_full.csv  ({len(results_df)} rows)")

    # --- Model Confidence Set (parallel over cells × info_sets) ---
    print("\n" + "=" * 70)
    print(f"MODEL CONFIDENCE SET (Hansen-Lunde-Nason 2011) "
          f"[parallel: n_jobs={N_JOBS}]")
    print("=" * 70)

    def _mcs_one(t, h_lab, info_set, L, h):
        try:
            in_75 = set(model_confidence_set(L, alpha=0.25, B=1000, h=h))
            in_90 = set(model_confidence_set(L, alpha=0.10, B=1000, h=h))
            return t, h_lab, info_set, in_75, in_90, None
        except Exception as e:
            return t, h_lab, info_set, set(), set(), str(e)

    mcs_results = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_mcs_one)(t, h_lab, info_set,
                          losses_panel[(t, h_lab, info_set)],
                          HORIZONS[h_lab])
        for (t, h_lab, info_set) in losses_panel.keys()
    )
    mcs_rows = []
    for t, h_lab, info_set, in_75, in_90, err in mcs_results:
        if err:
            print(f"  [{t} | {h_lab} | {info_set}] MCS failed: {err}")
        else:
            print(f"  [{t} | {h_lab} | {info_set}] MCS_90 size: {len(in_90)}  "
                  f"MCS_75 size: {len(in_75)}")
        for j, m in enumerate(MODELS):
            mcs_rows.append({"ticker": t, "horizon": h_lab,
                             "info_set": info_set, "model": m,
                             "in_MCS_75": j in in_75,
                             "in_MCS_90": j in in_90})
    pd.DataFrame(mcs_rows).to_csv("mcs_results.csv", index=False)
    print("MCS saved: mcs_results.csv")

    # --- ALE variable importance (parallel over cells) ---
    print("\n" + "=" * 70)
    print(f"ACCUMULATED LOCAL EFFECTS (Apley-Zhu 2020) [parallel: n_jobs={N_JOBS}]")
    print("=" * 70)
    target_models = ["HAR", "LogHAR", "ElasticNet", "RandomForest", "NN2_10"]

    def _ale_one(t, h_lab, h, feat_t):
        data = build_har_design(feat_t, h=h)
        n = len(data)
        n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
        train = data.iloc[:n_tr]
        val = data.iloc[n_tr:n_tr + n_va]
        test = data.iloc[n_tr + n_va:]
        tv = pd.concat([train, val])
        X_in = tv[HAR_FEATS].values
        local_rows = []
        for mname in target_models:
            if mname in ("HAR", "HAR-X"):
                m = LinearRegression().fit(tv[HAR_FEATS].values, tv["y"].values)
                predict_fn = lambda X, m=m: m.predict(X)
            elif mname == "LogHAR":
                m = LinearRegression().fit(np.log(tv[HAR_FEATS].values),
                                            np.log(tv["y"].values))
                sig = np.var(np.log(tv["y"].values) -
                              m.predict(np.log(tv[HAR_FEATS].values)),
                              ddof=1)
                predict_fn = lambda X, m=m, sig=sig: np.exp(
                    m.predict(np.log(np.maximum(X, 1e-12))) + 0.5 * sig)
            elif mname == "ElasticNet":
                # Val-tune EN exactly as in fit_predict (avoids zero-coef collapse).
                sc_tr = StandardScaler().fit(train[HAR_FEATS].values)
                Xtr = sc_tr.transform(train[HAR_FEATS].values)
                Xva = sc_tr.transform(val[HAR_FEATS].values)
                y_tr = train["y"].values
                y_va = val["y"].values
                l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
                alpha_grid = np.logspace(-5, 2, 100)
                best_mse, best = np.inf, (alpha_grid[0], l1_grid[0])
                for l1r in l1_grid:
                    for a in alpha_grid:
                        m = ElasticNet(alpha=a, l1_ratio=l1r,
                                       max_iter=10000, random_state=42)
                        m.fit(Xtr, y_tr)
                        mse = mean_squared_error(y_va, m.predict(Xva))
                        if mse < best_mse:
                            best_mse, best = mse, (a, l1r)
                sc = StandardScaler().fit(tv[HAR_FEATS].values)
                Xtv = sc.transform(tv[HAR_FEATS].values)
                en = ElasticNet(alpha=best[0], l1_ratio=best[1],
                                max_iter=10000, random_state=42)
                en.fit(Xtv, tv["y"].values)
                predict_fn = lambda X, sc=sc, en=en: en.predict(sc.transform(X))
            elif mname == "RandomForest":
                rf = RandomForestRegressor(n_estimators=500,
                                           min_samples_leaf=5,
                                           max_features="sqrt",
                                           n_jobs=INNER_JOBS, random_state=42)
                rf.fit(tv[HAR_FEATS].values, tv["y"].values)
                predict_fn = lambda X, rf=rf: rf.predict(X)
            elif mname == "NN2_10":
                sx = StandardScaler().fit(train[HAR_FEATS].values)
                sy = StandardScaler().fit(train["y"].values.reshape(-1, 1))
                Xtr = sx.transform(train[HAR_FEATS].values)
                ytr_s = sy.transform(train["y"].values.reshape(-1, 1)).ravel()
                seeds_pred = []
                val_mses = []
                for seed in range(NN_SEEDS_ENSEMBLE):
                    net = MLPRegressor(hidden_layer_sizes=(4, 2),
                                       activation="relu", solver="adam",
                                       learning_rate_init=0.001,
                                       max_iter=NN_MAX_ITER,
                                       random_state=seed)
                    net.fit(Xtr, ytr_s)
                    val_mses.append(
                        mean_squared_error(val["y"].values,
                                           sy.inverse_transform(
                                               net.predict(sx.transform(
                                                   val[HAR_FEATS].values)
                                                   ).reshape(-1, 1)).ravel()))
                    seeds_pred.append(net)
                top = np.argsort(val_mses)[:NN_TOP_K]
                def predict_fn(X, sx=sx, sy=sy, top=top,
                               seeds_pred=seeds_pred):
                    Xs = sx.transform(X)
                    return np.mean([
                        sy.inverse_transform(
                            seeds_pred[i].predict(Xs).reshape(-1, 1)).ravel()
                        for i in top
                    ], axis=0)
            else:
                continue
            imps = []
            for j_feat, fname in enumerate(HAR_FEATS):
                _, cum = ale_1d(predict_fn, X_in, j_feat)
                imps.append(np.std(cum))
            total = sum(imps) + 1e-20
            for j_feat, fname in enumerate(HAR_FEATS):
                local_rows.append({"ticker": t, "horizon": h_lab,
                                    "model": mname, "feature": fname,
                                    "importance": imps[j_feat] / total})
        return t, h_lab, local_rows

    ale_out = Parallel(n_jobs=N_JOBS, verbose=10)(
        delayed(_ale_one)(t, h_lab, HORIZONS[h_lab], feat[t])
        for t in TICKERS for h_lab in HORIZONS
    )
    ale_rows = []
    for t, h_lab, lr in ale_out:
        ale_rows.extend(lr)
        print(f"  [{t} | {h_lab}] ALE done")
    pd.DataFrame(ale_rows).to_csv("ale_importance.csv", index=False)
    print("ALE saved: ale_importance.csv")

    # --- VaR application (h = 1, both info sets) ---
    print("\n" + "=" * 70)
    print("VALUE-AT-RISK APPLICATION (h = 1, alpha = 5%) — both info sets")
    print("=" * 70)
    var_rows = []
    alpha = 0.05
    for info_set in info_sets_to_run:
        for t in TICKERS:
            if "h=1" not in forecasts_panel[info_set][t]:
                continue
            d = forecasts_panel[info_set][t]["h=1"]
            dates = d["dates"]
            r_test = daily_ret[t].reindex(dates).values
            r_hist = daily_ret[t].loc[:dates[0]].iloc[:-1].dropna().values
            for m in MODELS:
                sigma_pred = np.sqrt(np.clip(d["p"][m], 1e-20, None))
                var_pred = fhs_var_forecast(sigma_pred, r_hist, alpha=alpha)
                hits = (r_test < var_pred).astype(int)
                ql = quantile_loss(r_test, var_pred, alpha=alpha)
                _, p_kupiec = kupiec_lr(hits, alpha)
                _, p_chr = christoffersen_lr(hits)
                var_rows.append({"ticker": t, "info_set": info_set, "model": m,
                                 "quantile_loss": ql,
                                 "hit_rate": hits.mean(),
                                 "expected_rate": alpha,
                                 "p_kupiec": p_kupiec,
                                 "p_christoffersen": p_chr})
    pd.DataFrame(var_rows).to_csv("var_results.csv", index=False)
    print(f"VaR saved: var_results.csv  ({len(var_rows)} rows)")
    print("\nDone.")
    return results_df


if __name__ == "__main__":
    run()
