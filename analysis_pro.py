"""
Replication of Christensen, Siggaard & Veliyev (2023, JFE)
"A Machine Learning Approach to Volatility Forecasting"

Replicates:
  - Table 2 / 4 / 6 (relative MSE, M_HAR information set, horizons 1d/1w/1m)
  - Diebold-Mariano test of equal predictive accuracy
  - Figure 5 (forecast accuracy by RV decile)
  - Figure 8 (ACF of in-sample fitted RV)
  - Rolling-window robustness at h=22

Produces publication-quality figures + tables for a 3-page report.
"""

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = Path(".")
TICKERS = ["AAPL", "AMZN", "JPM"]
SECTOR = {"AAPL": "Technology", "AMZN": "Consumer", "JPM": "Financials"}
RV_FREQ_MIN = 5
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10
HORIZONS = {"h=1": 1, "h=5": 5, "h=22": 22}
HOR_LABEL = {"h=1": "1-day", "h=5": "1-week", "h=22": "1-month"}

MODELS = ["HAR", "LogHAR", "LevHAR", "SHAR", "HARQ",
          "ElasticNet", "RandomForest", "NN(4,2)"]
SHORT = {"HAR": "HAR", "LogHAR": "LogHAR", "LevHAR": "LevHAR", "SHAR": "SHAR",
         "HARQ": "HARQ", "ElasticNet": "EN", "RandomForest": "RF",
         "NN(4,2)": "NN"}
ML_MODELS = ["ElasticNet", "RandomForest", "NN(4,2)"]

COLORS = {
    "HAR":          "#34495e",
    "LogHAR":       "#3498db",
    "LevHAR":       "#8e44ad",
    "SHAR":         "#16a085",
    "HARQ":         "#d35400",
    "ElasticNet":   "#27ae60",
    "RandomForest": "#c0392b",
    "NN(4,2)":      "#e67e22",
}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
})


# =============================================================================
# Data loading & feature engineering
# =============================================================================
def load_minute_bars(path):
    df = pd.read_csv(path, header=None,
                     names=["date", "time", "open", "high", "low", "close", "vol"],
                     dtype={"date": str, "time": str})
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"],
                              format="%m/%d/%Y %H:%M")
    return df.sort_values("dt").reset_index(drop=True)


def compute_features(df, freq=5):
    """Daily RV, RQ, RV+, RV-, daily return — from 1-min closes."""
    df = df.set_index("dt")
    closes = (df["close"]
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
    return out


def build_har_design(feat, h=1):
    """HAR-X-style design matrix. Target y = mean(RV_{t+1..t+h})."""
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
    return df.dropna()


# =============================================================================
# Models
# =============================================================================
HAR_FEATS = ["RVD", "RVW", "RVM"]
LEVHAR_FEATS = ["RVD", "RVW", "RVM", "rN_D", "rN_W", "rN_M"]
SHAR_FEATS = ["RVp_d", "RVn_d", "RVW", "RVM"]


def _harq_design(df):
    return pd.DataFrame({
        "RVD": df["RVD"],
        "RVD_RQ": df["RVD"] * df["RQ_d"],
        "RVW": df["RVW"],
        "RVM": df["RVM"],
    }, index=df.index)


def fit_predict(name, train, test, val=None):
    """Return (test_pred, in_sample_pred)."""
    y_tr = train["y"].values

    if name == "HAR":
        m = LinearRegression().fit(train[HAR_FEATS], y_tr)
        return m.predict(test[HAR_FEATS]), m.predict(train[HAR_FEATS])

    if name == "LogHAR":
        Xtr = np.log(train[HAR_FEATS]); ytr = np.log(y_tr)
        m = LinearRegression().fit(Xtr, ytr)
        sig = np.var(ytr - m.predict(Xtr), ddof=1)
        Xte = np.log(test[HAR_FEATS])
        return (np.exp(m.predict(Xte) + 0.5 * sig),
                np.exp(m.predict(Xtr) + 0.5 * sig))

    if name == "LevHAR":
        m = LinearRegression().fit(train[LEVHAR_FEATS], y_tr)
        return m.predict(test[LEVHAR_FEATS]), m.predict(train[LEVHAR_FEATS])

    if name == "SHAR":
        m = LinearRegression().fit(train[SHAR_FEATS], y_tr)
        return m.predict(test[SHAR_FEATS]), m.predict(train[SHAR_FEATS])

    if name == "HARQ":
        Xtr = _harq_design(train); Xte = _harq_design(test)
        m = LinearRegression().fit(Xtr, y_tr)
        return m.predict(Xte), m.predict(Xtr)

    if name == "ElasticNet":
        # Hyperparameters tuned on a held-out validation set (matches paper
        # Appendix A.4 — avoids k-fold CV's lookahead leak on autocorrelated
        # volatility data). Final forecast uses train+val with chosen (alpha, l1_ratio).
        assert val is not None
        sc = StandardScaler().fit(train[HAR_FEATS])
        Xtr = sc.transform(train[HAR_FEATS])
        Xva = sc.transform(val[HAR_FEATS])
        y_va = val["y"].values

        l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
        alpha_grid = np.logspace(-5, 2, 100)
        best_mse, best = np.inf, (alpha_grid[0], l1_grid[0])
        for l1r in l1_grid:
            for alpha in alpha_grid:
                m = ElasticNet(alpha=alpha, l1_ratio=l1r,
                               max_iter=10000, random_state=42)
                m.fit(Xtr, y_tr)
                mse_va = mean_squared_error(y_va, m.predict(Xva))
                if mse_va < best_mse:
                    best_mse, best = mse_va, (alpha, l1r)

        # Refit on train+val with the validated hyperparameters
        tv = pd.concat([train, val])
        sc2 = StandardScaler().fit(tv[HAR_FEATS])
        Xtv = sc2.transform(tv[HAR_FEATS])
        Xte = sc2.transform(test[HAR_FEATS])
        final = ElasticNet(alpha=best[0], l1_ratio=best[1],
                           max_iter=10000, random_state=42)
        final.fit(Xtv, tv["y"].values)
        return final.predict(Xte), final.predict(Xtv)

    if name == "RandomForest":
        m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                                  max_features="sqrt", n_jobs=-1,
                                  random_state=42)
        m.fit(train[HAR_FEATS], y_tr)
        return m.predict(test[HAR_FEATS]), m.predict(train[HAR_FEATS])

    if name == "NN(4,2)":
        assert val is not None
        sx = StandardScaler().fit(train[HAR_FEATS])
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Xtr = sx.transform(train[HAR_FEATS])
        Xva = sx.transform(val[HAR_FEATS])
        Xte = sx.transform(test[HAR_FEATS])
        ytr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()

        val_mses, te_preds, tr_preds = [], [], []
        for seed in range(10):
            net = MLPRegressor(hidden_layer_sizes=(4, 2),
                               activation="relu", solver="adam",
                               learning_rate_init=0.001, max_iter=500,
                               random_state=seed)
            net.fit(Xtr, ytr_s)
            inv = lambda p: sy.inverse_transform(p.reshape(-1, 1)).ravel()
            vp = inv(net.predict(Xva))
            tp = inv(net.predict(Xte))
            ip = inv(net.predict(Xtr))
            val_mses.append(mean_squared_error(val["y"], vp))
            te_preds.append(tp); tr_preds.append(ip)
        top = np.argsort(val_mses)[:3]
        return (np.mean([te_preds[i] for i in top], axis=0),
                np.mean([tr_preds[i] for i in top], axis=0))

    raise ValueError(name)


def insanity_filter(pred, train_y):
    """Floor at training-set min RV; ceiling at 2x max to avoid wild forecasts."""
    return np.clip(pred, max(train_y.min(), 1e-12), train_y.max() * 2.0)


# =============================================================================
# Diebold-Mariano with HLN small-sample correction (Bartlett-Newey-West variance)
# =============================================================================
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
    correction = (T + 1 - 2 * h + h * (h - 1) / T) / T
    if correction <= 0:
        return np.nan, np.nan
    DM_adj = DM * np.sqrt(correction)
    p_one = 1 - stats.norm.cdf(DM_adj)
    return DM_adj, p_one


def dm_test(y, p_bench, p_alt, h=1):
    """H0: equal MSE. p-value one-sided (alt has LOWER MSE)."""
    return _dm_from_diff((y - p_bench) ** 2 - (y - p_alt) ** 2, h=h)


def qlike_pointwise(y, p, eps=1e-20):
    """Patton (2011) QLIKE: r - log(r) - 1, r = y / p. Robust to noise in RV proxy."""
    y = np.maximum(y, eps)
    p = np.maximum(p, eps)
    r = y / p
    return r - np.log(r) - 1


def dm_test_qlike(y, p_bench, p_alt, h=1):
    """DM on QLIKE differential. Same convention: alt < bench => p-value small."""
    return _dm_from_diff(qlike_pointwise(y, p_bench) - qlike_pointwise(y, p_alt), h=h)


def stars(p):
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# =============================================================================
# Main pipeline
# =============================================================================
def run_pipeline():
    rv_panel, feat_panel = {}, {}
    print("Loading 1-min data and computing 5-min RV...")
    for t in TICKERS:
        df = load_minute_bars(DATA_DIR / f"{t}.txt")
        f = compute_features(df, freq=RV_FREQ_MIN)
        rv_panel[t], feat_panel[t] = f["RV"], f
        print(f"  {t}: {len(f)} days "
              f"({f.index[0].date()} → {f.index[-1].date()}), "
              f"mean RV={f['RV'].mean():.3e}")

    rows = []
    fitted = {}
    forecasts = {}

    for t in TICKERS:
        fitted[t] = {}
        forecasts[t] = {}
        for h_lab, h in HORIZONS.items():
            data = build_har_design(feat_panel[t], h=h)
            n = len(data)
            n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
            train = data.iloc[:n_tr]
            val = data.iloc[n_tr:n_tr + n_va]
            test = data.iloc[n_tr + n_va:]
            tv = pd.concat([train, val])
            y_test = test["y"].values
            print(f"  [{t} | {h_lab}] train={len(train)} val={len(val)} "
                  f"test={len(test)} ({test.index[0].date()} → "
                  f"{test.index[-1].date()})")

            fitted[t][h_lab] = {}
            forecasts[t][h_lab] = {"y": y_test, "dates": test.index, "p": {}}

            preds = {}
            for m in MODELS:
                if m == "NN(4,2)":
                    pte, pin = fit_predict(m, train, test, val=val)
                    idx_in = train.index
                elif m == "ElasticNet":
                    # Pass train and val separately for hold-out tuning;
                    # function refits on train+val and returns tv-length pin.
                    pte, pin = fit_predict(m, train, test, val=val)
                    idx_in = tv.index
                else:
                    pte, pin = fit_predict(m, tv, test)
                    idx_in = tv.index
                pte = insanity_filter(pte, tv["y"])
                pin = insanity_filter(pin, tv["y"])
                preds[m] = pte
                fitted[t][h_lab][m] = pd.Series(pin, index=idx_in)
                forecasts[t][h_lab]["p"][m] = pte

            mse_har = mean_squared_error(y_test, preds["HAR"])
            qlike_har = qlike_pointwise(y_test, preds["HAR"]).mean()
            for m in MODELS:
                mse = mean_squared_error(y_test, preds[m])
                ql = qlike_pointwise(y_test, preds[m]).mean()
                if m == "HAR":
                    DM, pv, DMq, pvq = np.nan, np.nan, np.nan, np.nan
                else:
                    DM, pv = dm_test(y_test, preds["HAR"], preds[m], h=h)
                    DMq, pvq = dm_test_qlike(y_test, preds["HAR"], preds[m], h=h)
                rows.append({
                    "ticker": t, "horizon": h_lab, "model": m,
                    "mse": mse, "rel_mse": mse / mse_har,
                    "DM": DM, "p_value": pv,
                    "qlike": ql, "rel_qlike": ql / qlike_har,
                    "DM_qlike": DMq, "p_qlike": pvq,
                })

    return pd.DataFrame(rows), fitted, forecasts, rv_panel, feat_panel


# =============================================================================
# Rolling-window robustness check at h=22 (HAR vs RF)
# =============================================================================
def run_rolling_h22(feat_panel, step=22):
    """Expanding-window refit every `step` days for HAR and RF at h=22."""
    print("\nRunning rolling-window robustness (h=22)...")
    out = {}
    h = 22
    for t in TICKERS:
        data = build_har_design(feat_panel[t], h=h)
        n = len(data)
        n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
        # Initial cutoff = end of validation (paper's convention)
        cut0 = n_tr + n_va
        test_idx = np.arange(cut0, n)
        y_te = data["y"].iloc[test_idx].values
        preds_har = np.zeros(len(test_idx))
        preds_rf = np.zeros(len(test_idx))

        for start in range(0, len(test_idx), step):
            end = min(start + step, len(test_idx))
            te_block = test_idx[start:end]
            cut = test_idx[start]  # use data 0..cut-1 as expanded training
            tv_block = data.iloc[:cut]
            test_block = data.iloc[te_block]
            # HAR
            m = LinearRegression().fit(tv_block[HAR_FEATS], tv_block["y"])
            p_h = insanity_filter(m.predict(test_block[HAR_FEATS]), tv_block["y"])
            # RF
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                       max_features="sqrt",
                                       n_jobs=-1, random_state=42)
            rf.fit(tv_block[HAR_FEATS], tv_block["y"].values)
            p_r = insanity_filter(rf.predict(test_block[HAR_FEATS]),
                                  tv_block["y"])
            preds_har[start:end] = p_h
            preds_rf[start:end] = p_r

        mse_har = mean_squared_error(y_te, preds_har)
        mse_rf = mean_squared_error(y_te, preds_rf)
        out[t] = {"rel_rf": mse_rf / mse_har, "mse_har": mse_har, "mse_rf": mse_rf}
        print(f"  {t}: rolling rel-MSE(RF/HAR) = {mse_rf/mse_har:.3f}")
    return out


# =============================================================================
# Figures
# =============================================================================
def fig_rv_overview(rv_panel, outpath):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for t in TICKERS:
        ann = np.sqrt(rv_panel[t] * 252) * 100
        ax.plot(ann.index, ann.values, lw=0.7, alpha=0.85,
                label=f"{t} ({SECTOR[t]})", color=COLORS[list(COLORS)[TICKERS.index(t)]])
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Daily realised volatility from 5-minute returns, 2016–2024")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, None)
    plt.savefig(outpath); plt.close()
    return outpath


def fig_acf_persistence(fitted, outpath):
    """Paper's Figure 8 reproduction — 3 tickers × 2 horizons."""
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.5), sharex=True, sharey=True)
    show = ["HAR", "LogHAR", "RandomForest", "NN(4,2)"]
    h_show = [("h=1", "Short horizon: h = 1 day"),
              ("h=22", "Long horizon: h = 22 days")]
    for i, t in enumerate(TICKERS):
        for j, (h_key, h_title) in enumerate(h_show):
            ax = axes[i, j]
            for m in show:
                s = fitted[t][h_key][m].dropna()
                a = acf(s, nlags=120, fft=True)
                ax.plot(a, color=COLORS[m], lw=1.3, label=SHORT[m])
            ax.axhline(0, color="k", lw=0.4)
            T = len(fitted[t][h_key]["HAR"].dropna())
            ax.axhline(2 / np.sqrt(T), color="grey", lw=0.5, ls="--")
            ax.axhline(-2 / np.sqrt(T), color="grey", lw=0.5, ls="--")
            ax.set_xlim(0, 120); ax.set_ylim(-0.1, 1.02)
            if i == 0:
                ax.set_title(h_title)
            if j == 0:
                ax.set_ylabel(f"{t}\nACF of fitted RV")
            if i == 2:
                ax.set_xlabel("Lag (trading days)")
    axes[0, 1].legend(loc="upper right", ncol=2)
    fig.suptitle("In-sample autocorrelation of fitted realised variance",
                 y=1.00, fontsize=11, weight="bold")
    plt.savefig(outpath); plt.close()
    return outpath


def fig_relative_mse(results_df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), sharey=False)
    for i, t in enumerate(TICKERS):
        ax = axes[i]
        sub = results_df[results_df["ticker"] == t]
        x = np.arange(len(MODELS))
        width = 0.27
        for k, h_lab in enumerate(HORIZONS):
            vals = [sub[(sub["model"] == m) & (sub["horizon"] == h_lab)]
                    ["rel_mse"].iloc[0] for m in MODELS]
            ax.bar(x + (k - 1) * width, vals, width,
                   label=HOR_LABEL[h_lab], alpha=0.85,
                   color=["#34495e", "#3498db", "#c0392b"][k])
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in MODELS],
                           rotation=40, ha="right", fontsize=7.5)
        ax.set_title(f"{t}")
        if i == 0:
            ax.set_ylabel("Out-of-sample MSE / HAR MSE")
            ax.legend(loc="upper left", ncol=3, fontsize=7.5)
        ax.set_ylim(0, max(2.6, sub["rel_mse"].max() * 1.05))
    fig.suptitle("Forecast accuracy relative to HAR across stocks and horizons",
                 y=1.04, fontsize=11, weight="bold")
    plt.savefig(outpath); plt.close()
    return outpath


def fig_forecast_overlay(forecasts, outpath, ticker="AAPL"):
    h_key = "h=1"
    d = forecasts[ticker][h_key]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(d["dates"], np.sqrt(d["y"] * 252) * 100, color="black", lw=0.9,
            label="Realised", alpha=0.7)
    for m in ["HAR", "LogHAR", "NN(4,2)"]:
        ax.plot(d["dates"], np.sqrt(d["p"][m] * 252) * 100,
                color=COLORS[m], lw=0.9, alpha=0.85, label=SHORT[m])
    ax.set_title(f"{ticker}: 1-day-ahead realised-volatility forecasts")
    ax.set_ylabel("Annualised volatility (%)")
    ax.legend(loc="upper right", ncol=4, fontsize=7.5)
    plt.savefig(outpath); plt.close()
    return outpath


def fig_decile_mse(forecasts, outpath):
    """Paper's Figure 5 reproduction — accuracy by RV decile, h=1."""
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    h_key = "h=1"
    show = ["LogHAR", "ElasticNet", "RandomForest", "NN(4,2)"]
    for i, t in enumerate(TICKERS):
        ax = axes[i]
        d = forecasts[t][h_key]
        y = d["y"]
        thresholds = np.quantile(y, np.linspace(0.1, 0.9, 9))
        deciles = np.digitize(y, thresholds)
        for m in show:
            rel = []
            for q in range(10):
                mask = deciles == q
                if mask.sum() < 3:
                    rel.append(np.nan); continue
                e_m = np.mean((y[mask] - d["p"][m][mask]) ** 2)
                e_h = np.mean((y[mask] - d["p"]["HAR"][mask]) ** 2)
                rel.append(e_m / e_h if e_h > 0 else np.nan)
            ax.plot(range(1, 11), rel, marker="o", ms=4.5,
                    color=COLORS[m], lw=1.2, label=SHORT[m])
        ax.axhline(1.0, color="k", lw=0.5, ls="--")
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Decile of realised RV")
        if i == 0:
            ax.set_ylabel("MSE / HAR MSE")
            ax.legend(loc="best", fontsize=7.5)
        ax.set_title(t)
        ax.set_ylim(0, 2.5)
    fig.suptitle("Forecast accuracy by volatility decile (h = 1 day)",
                 y=1.04, fontsize=11, weight="bold")
    plt.savefig(outpath); plt.close()
    return outpath


def fig_rolling_robust(results_df, rolling_out, outpath):
    """Static vs rolling-refit RF rel-MSE at h=22."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(TICKERS))
    static = [results_df[(results_df.ticker == t) &
                         (results_df.horizon == "h=22") &
                         (results_df.model == "RandomForest")]
              ["rel_mse"].iloc[0] for t in TICKERS]
    rolling = [rolling_out[t]["rel_rf"] for t in TICKERS]
    width = 0.35
    ax.bar(x - width / 2, static, width, label="Static fit",
           color="#c0392b", alpha=0.85)
    ax.bar(x + width / 2, rolling, width, label="Rolling refit (every 22d)",
           color="#34495e", alpha=0.85)
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(TICKERS)
    ax.set_ylabel("MSE(RF) / MSE(HAR)")
    ax.set_title("Random-forest rel-MSE at h=22: static vs rolling refit")
    ax.legend(loc="upper right")
    for xi, (s, r) in enumerate(zip(static, rolling)):
        ax.text(xi - width / 2, s + 0.04, f"{s:.2f}",
                ha="center", fontsize=8)
        ax.text(xi + width / 2, r + 0.04, f"{r:.2f}",
                ha="center", fontsize=8)
    plt.savefig(outpath); plt.close()
    return outpath


# =============================================================================
# Tables
# =============================================================================
def table_descriptive(rv_panel, outpath_csv, outpath_md):
    rows = []
    for t in TICKERS:
        rv = rv_panel[t]
        ann = np.sqrt(rv * 252) * 100
        rows.append({
            "Ticker": t,
            "Sector": SECTOR[t],
            "N days": f"{len(rv):,}",
            "Start": str(rv.index[0].date()),
            "End": str(rv.index[-1].date()),
            "Mean RV (x1e4)": f"{rv.mean()*1e4:.2f}",
            "Median RV (x1e4)": f"{rv.median()*1e4:.2f}",
            "Mean ann. sigma (%)": f"{ann.mean():.1f}",
            "Skewness": f"{rv.skew():.2f}",
            "Kurtosis": f"{rv.kurt():.2f}",
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(outpath_csv, index=False)
    with open(outpath_md, "w") as f:
        f.write("**Table 1.** Descriptive statistics of daily realised "
                "variance (5-min subsampled). RV is reported scaled by "
                "$10^4$ for readability; annualised volatility = "
                "$\\sqrt{252 \\cdot \\mathrm{RV}}$.\n\n")
        f.write(tab.to_markdown(index=False))
    return tab


def table_main(results_df, outpath_csv, outpath_md):
    rows = []
    for h_lab in HORIZONS:
        for m in MODELS:
            row = {"Horizon": HOR_LABEL[h_lab], "Model": SHORT[m]}
            cells = []
            for t in TICKERS:
                r = results_df[(results_df.ticker == t) &
                               (results_df.horizon == h_lab) &
                               (results_df.model == m)].iloc[0]
                cell = f"{r['rel_mse']:.3f}{stars(r['p_value'])}"
                row[t] = cell
                cells.append(r["rel_mse"])
            row["Mean"] = f"{np.mean(cells):.3f}"
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(outpath_csv, index=False)
    with open(outpath_md, "w") as f:
        f.write("**Table 2.** Out-of-sample MSE relative to HAR (lower is "
                "better; 1.000 = HAR baseline). Stars: Diebold–Mariano "
                "rejection of equal predictive accuracy in favour of the "
                "model in the row, with HLN small-sample correction "
                "(*: 10%, **: 5%, ***: 1%).\n\n")
        f.write(tab.to_markdown(index=False))
    return tab


def table_qlike(results_df, outpath_csv, outpath_md):
    """Compact QLIKE robustness table: cross-section mean rel-QLIKE per (horizon, model)."""
    rows = []
    for h_lab in HORIZONS:
        row = {"Horizon": HOR_LABEL[h_lab]}
        for m in MODELS:
            if m == "HAR":
                continue
            r = results_df[(results_df.horizon == h_lab) &
                           (results_df.model == m)]
            mean_rq = r["rel_qlike"].mean()
            # Star if DM-QLIKE rejects in favour of model on the majority of stocks at 10%
            n_sig = (r["p_qlike"] < 0.10).sum()
            mark = "*" if n_sig >= 2 else ""
            row[SHORT[m]] = f"{mean_rq:.3f}{mark}"
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(outpath_csv, index=False)
    with open(outpath_md, "w") as f:
        f.write("**Table 3.** Robustness check: out-of-sample QLIKE "
                "(Patton, 2011) relative to HAR, cross-sectional average "
                "across the 3 stocks. Lower is better; 1.000 = HAR "
                "baseline. A '*' indicates Diebold–Mariano rejection on "
                "the QLIKE differential in at least 2 of 3 stocks at the "
                "10% level.\n\n")
        f.write(tab.to_markdown(index=False))
    return tab


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 64)
    print("Replication of Christensen, Siggaard, Veliyev (2023, JFE)")
    print("=" * 64)

    results_df, fitted, forecasts, rv_panel, feat_panel = run_pipeline()
    rolling_out = run_rolling_h22(feat_panel)

    print("\nGenerating figures...")
    figs = {
        "fig1_rv_overview.png": fig_rv_overview(rv_panel, "fig1_rv_overview.png"),
        "fig2_acf_persistence.png": fig_acf_persistence(fitted, "fig2_acf_persistence.png"),
        "fig3_relative_mse.png": fig_relative_mse(results_df, "fig3_relative_mse.png"),
        "fig4_forecasts.png": fig_forecast_overlay(forecasts, "fig4_forecasts.png"),
        "fig5_decile_mse.png": fig_decile_mse(forecasts, "fig5_decile_mse.png"),
        "fig6_rolling_robust.png": fig_rolling_robust(results_df, rolling_out, "fig6_rolling_robust.png"),
    }
    for k, v in figs.items():
        print(f"  saved -> {v}")

    print("\nGenerating tables...")
    tab1 = table_descriptive(rv_panel, "tab1_descriptive.csv", "tab1_descriptive.md")
    tab2 = table_main(results_df, "tab2_main_results.csv", "tab2_main_results.md")
    tab3 = table_qlike(results_df, "tab3_qlike.csv", "tab3_qlike.md")

    print("\n" + "=" * 64)
    print("TABLE 1: Descriptive statistics")
    print("=" * 64)
    print(tab1.to_string(index=False))
    print("\n" + "=" * 64)
    print("TABLE 2: Relative MSE (vs HAR) with DM stars")
    print("=" * 64)
    print(tab2.to_string(index=False))
    print("\n" + "=" * 64)
    print("TABLE 3: Relative QLIKE (cross-section mean)")
    print("=" * 64)
    print(tab3.to_string(index=False))

    print("\n" + "=" * 64)
    print("ROLLING-WINDOW ROBUSTNESS at h=22")
    print("=" * 64)
    for t in TICKERS:
        r = rolling_out[t]
        static_rf = results_df[(results_df.ticker == t) &
                               (results_df.horizon == "h=22") &
                               (results_df.model == "RandomForest")
                               ]["rel_mse"].iloc[0]
        print(f"  {t}: static rel-MSE(RF) = {static_rf:.3f}, "
              f"rolling rel-MSE(RF) = {r['rel_rf']:.3f}")

    results_df.to_csv("results_pro.csv", index=False)
    print("\nAll outputs saved. Done.")
