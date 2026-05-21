"""
Replication of Christensen, Siggaard & Veliyev (2023, JFE):
"A Machine Learning Approach to Volatility Forecasting"

Focuses on three core results:
  1. One-day-ahead MSE comparison (Table 2 of paper, M_HAR information set)
  2. One-month-ahead MSE comparison (Table 6, M_HAR information set)
  3. ACF of fitted RV series (Figure 8) — explains *why* ML wins long-horizon

Models: HAR, LogHAR, Elastic Net, Random Forest, Neural Network (4 -> 2 neurons)
Data:   1-minute OHLCV bars, subsampled to 5-min for RV computation

Run from a directory containing AAPL.txt, AMZN.txt, JPM.txt.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.stattools import acf

warnings.filterwarnings("ignore")
np.random.seed(42)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_DIR    = Path(".")               # adjust if your data lives elsewhere
TICKERS     = ["AAPL", "AMZN", "JPM"]
RV_FREQ_MIN = 5                       # subsample 1-min -> 5-min for RV (Andersen-Bollerslev convention)
ANNUALIZE   = 252 * 1e4               # paper expresses RV as annualized variance in % -- scale at the end if you want
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.10                    # test is the remaining 20%
N_LAGS      = 22                      # we lose the first 22 days to monthly lag
HORIZONS    = {"1d": 1, "1m": 22}     # one-day-ahead and one-month-ahead average


# -----------------------------------------------------------------------------
# STEP 1: LOAD + DIAGNOSE COVERAGE
# -----------------------------------------------------------------------------
def load_minute_bars(path):
    df = pd.read_csv(
        path, header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={"date": str, "time": str},
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                    format="%m/%d/%Y %H:%M")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["trading_date"] = df["datetime"].dt.date
    return df


def diagnose(df, ticker):
    """Print time-coverage info so we know whether the file is full-session or truncated."""
    bars_per_day = df.groupby("trading_date").size()
    times = df["time"].value_counts().sort_index()
    print(f"\n--- DIAGNOSTIC: {ticker} ---")
    print(f"  Rows:           {len(df):,}")
    print(f"  Date range:     {df['trading_date'].min()} -> {df['trading_date'].max()}")
    print(f"  Trading days:   {bars_per_day.shape[0]:,}")
    print(f"  Bars/day -- min:{bars_per_day.min()}, median:{int(bars_per_day.median())}, max:{bars_per_day.max()}")
    print(f"  Earliest time:  {times.index[0]}")
    print(f"  Latest time:    {times.index[-1]}")
    # 390 bars/day = full 6.5h US session; ~210 = half session
    if bars_per_day.median() >= 380:
        print("  -> FULL SESSION detected (timestamps are in NY time or equivalent).")
    elif bars_per_day.median() >= 200:
        print("  -> HALF SESSION detected. RV will be computed on morning data only.")
        print("     Report this as a deviation from the paper in your discussion.")
    else:
        print("  -> Unusual coverage. Check before continuing.")


# -----------------------------------------------------------------------------
# STEP 2: COMPUTE DAILY REALIZED VARIANCE
# -----------------------------------------------------------------------------
def compute_daily_rv(df, freq_min=5):
    """
    Resample 1-min bars to `freq_min`-min bars (using last close in each window),
    take log-returns within each day, sum squared returns -> daily RV.
    """
    df = df.set_index("datetime")
    # Resample to freq_min using last close, grouping by day to avoid overnight returns
    bars = (df["close"]
            .resample(f"{freq_min}min", label="right", closed="right")
            .last()
            .dropna())

    bars = bars.to_frame("close")
    bars["date"] = bars.index.date
    bars["logret"] = np.log(bars["close"]).groupby(bars["date"]).diff()  # NaN at day open -> excluded

    rv = (bars.groupby("date")["logret"]
              .apply(lambda r: (r.dropna() ** 2).sum()))
    rv.index = pd.to_datetime(rv.index)
    rv.name = "RV"
    # Drop days with too few intra-day obs (e.g. half-days around holidays)
    n_obs = bars.groupby("date")["logret"].apply(lambda r: r.dropna().shape[0])
    rv = rv[n_obs >= 10]               # need at least 10 returns to call it RV
    return rv


# -----------------------------------------------------------------------------
# STEP 3: BUILD HAR LAGS + FORECAST TARGETS
# -----------------------------------------------------------------------------
def build_har_dataset(rv, h=1):
    """
    Build the M_HAR feature set: daily, weekly (5-day), monthly (22-day) lagged RV.
    Target = average RV over the next h days, t+1 ... t+h.
    All lags use .shift(1).rolling(...).mean() -> no look-ahead.
    """
    df = pd.DataFrame({"RV": rv})
    df["RVD"] = df["RV"].shift(1)
    df["RVW"] = df["RV"].shift(1).rolling(5).mean()
    df["RVM"] = df["RV"].shift(1).rolling(22).mean()
    # Target: average RV from t+1 to t+h
    if h == 1:
        df["y"] = df["RV"]             # next-day RV
    else:
        # average of RV at times t, t+1, ..., t+h-1 then shift back by 0:
        # we want y_t = mean(RV_{t+1}..RV_{t+h}); shift -1 so RV_{t+1} is at row t
        df["y"] = df["RV"].shift(-1).rolling(h).mean().shift(-(h - 1))
    return df.dropna()


# -----------------------------------------------------------------------------
# STEP 4: SPLIT TRAIN / VAL / TEST
# -----------------------------------------------------------------------------
def split_train_val_test(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    n = len(df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]
    return train, val, test


# -----------------------------------------------------------------------------
# STEP 5: MODELS
# -----------------------------------------------------------------------------
FEATURES = ["RVD", "RVW", "RVM"]


def fit_predict_har(train, test):
    """Plain OLS HAR."""
    m = LinearRegression().fit(train[FEATURES], train["y"])
    return m, m.predict(test[FEATURES]), m.predict(train[FEATURES])


def fit_predict_loghar(train, test):
    """LogHAR with Jensen bias correction."""
    Xtr = np.log(train[FEATURES])
    ytr = np.log(train["y"])
    m = LinearRegression().fit(Xtr, ytr)
    resid_var = np.var(ytr - m.predict(Xtr), ddof=1)
    # Bias correction: E[exp(yhat)] = exp(mu + 0.5*var)
    test_pred = np.exp(m.predict(np.log(test[FEATURES])) + 0.5 * resid_var)
    train_pred = np.exp(m.predict(Xtr) + 0.5 * resid_var)
    return m, test_pred, train_pred


def fit_predict_elasticnet(train_val, test):
    """Elastic Net with internal CV (mimics paper's grid search)."""
    scaler = StandardScaler().fit(train_val[FEATURES])
    Xtr = scaler.transform(train_val[FEATURES])
    Xte = scaler.transform(test[FEATURES])
    m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                     alphas=np.logspace(-5, 2, 100),
                     cv=5, max_iter=10000, random_state=42)
    m.fit(Xtr, train_val["y"])
    return m, m.predict(Xte), m.predict(Xtr)


def fit_predict_rf(train_val, test):
    """Random Forest with paper's defaults: 500 trees, min_samples_leaf=5."""
    m = RandomForestRegressor(n_estimators=500, min_samples_leaf=5,
                              max_features="sqrt", n_jobs=-1, random_state=42)
    m.fit(train_val[FEATURES], train_val["y"])
    return m, m.predict(test[FEATURES]), m.predict(train_val[FEATURES])


def fit_predict_nn(train, val, test):
    """NN2: 4 -> 2 hidden neurons, ReLU, Adam.
    Both features AND target are standardized (RV is on scale ~1e-6, which
    breaks the optimizer otherwise). We invert the target-scaling at predict.
    Mimics paper's ensemble: train 10 seeds, keep the best on the validation set.
    """
    x_scaler = StandardScaler().fit(train[FEATURES])
    y_scaler = StandardScaler().fit(train[["y"]])
    Xtr = x_scaler.transform(train[FEATURES])
    Xva = x_scaler.transform(val[FEATURES])
    Xte = x_scaler.transform(test[FEATURES])
    ytr = y_scaler.transform(train[["y"]]).ravel()

    best_mse, best_model = np.inf, None
    for seed in range(10):
        m = MLPRegressor(hidden_layer_sizes=(4, 2), activation="relu",
                         solver="adam", learning_rate_init=0.001,
                         max_iter=500, early_stopping=False,
                         random_state=seed)
        m.fit(Xtr, ytr)
        val_pred = y_scaler.inverse_transform(m.predict(Xva).reshape(-1, 1)).ravel()
        val_mse = mean_squared_error(val["y"], val_pred)
        if val_mse < best_mse:
            best_mse, best_model = val_mse, m

    test_pred  = y_scaler.inverse_transform(best_model.predict(Xte).reshape(-1, 1)).ravel()
    train_pred = y_scaler.inverse_transform(best_model.predict(Xtr).reshape(-1, 1)).ravel()
    return best_model, test_pred, train_pred


# -----------------------------------------------------------------------------
# STEP 6: APPLY INSANITY FILTER (paper's: replace negative forecasts)
# -----------------------------------------------------------------------------
def insanity_filter(pred, train_rv):
    floor = train_rv.min()
    return np.where(pred < 0, floor, pred)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def run_ticker(ticker, do_acf_plot=False):
    print(f"\n========== {ticker} ==========")
    path = DATA_DIR / f"{ticker}.txt"
    df_min = load_minute_bars(path)
    diagnose(df_min, ticker)

    print(f"\n[Step 1] Computing daily realized variance @ {RV_FREQ_MIN}-min ...")
    rv = compute_daily_rv(df_min, freq_min=RV_FREQ_MIN)
    print(f"  {len(rv)} daily RV observations. Sample mean: {rv.mean():.6e}")

    results = {h: {} for h in HORIZONS}
    acf_data = {}

    for h_name, h in HORIZONS.items():
        print(f"\n[Step 2] Building HAR dataset for h={h_name} ...")
        data = build_har_dataset(rv, h=h)
        train, val, test = split_train_val_test(data)
        print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        train_val = pd.concat([train, val])

        # ----- HAR -----
        m_har, pred_har, fit_har = fit_predict_har(train_val, test)
        pred_har = insanity_filter(pred_har, train_val["y"])
        mse_har = mean_squared_error(test["y"], pred_har)
        results[h_name]["HAR"] = mse_har

        # ----- LogHAR -----
        _, pred_lh, fit_lh = fit_predict_loghar(train_val, test)
        pred_lh = insanity_filter(pred_lh, train_val["y"])
        results[h_name]["LogHAR"] = mean_squared_error(test["y"], pred_lh)

        # ----- Elastic Net -----
        _, pred_en, fit_en = fit_predict_elasticnet(train_val, test)
        pred_en = insanity_filter(pred_en, train_val["y"])
        results[h_name]["ElasticNet"] = mean_squared_error(test["y"], pred_en)

        # ----- Random Forest -----
        _, pred_rf, fit_rf = fit_predict_rf(train_val, test)
        results[h_name]["RandomForest"] = mean_squared_error(test["y"], pred_rf)

        # ----- Neural Network -----
        _, pred_nn, fit_nn = fit_predict_nn(train, val, test)
        pred_nn = insanity_filter(pred_nn, train_val["y"])
        results[h_name]["NN(4,2)"] = mean_squared_error(test["y"], pred_nn)

        # Save fitted-on-training series for ACF
        acf_data[h_name] = {
            "HAR": pd.Series(fit_har, index=train_val.index),
            "RF":  pd.Series(fit_rf,  index=train_val.index),
            "NN":  pd.Series(fit_nn,  index=train.index),
        }

    return results, acf_data, rv


def print_relative_mse(all_results):
    """Print MSE relative to HAR per ticker, per horizon."""
    print("\n" + "=" * 70)
    print("RELATIVE MSE vs HAR (lower = better than HAR)")
    print("=" * 70)
    for h_name in HORIZONS:
        print(f"\n--- Horizon: {h_name} ---")
        header = f"{'Model':<15} " + "  ".join(f"{t:>10}" for t in TICKERS) + f"  {'Mean':>10}"
        print(header)
        print("-" * len(header))
        models = ["HAR", "LogHAR", "ElasticNet", "RandomForest", "NN(4,2)"]
        for mdl in models:
            row = [all_results[t][h_name][mdl] / all_results[t][h_name]["HAR"]
                   for t in TICKERS]
            mean = np.mean(row)
            print(f"{mdl:<15} " + "  ".join(f"{v:>10.4f}" for v in row) + f"  {mean:>10.4f}")


def plot_acf_comparison(acf_data_per_ticker, ticker_to_plot="AAPL", outpath="acf_plot.png"):
    """Reproduce paper's Figure 8: ACF of in-sample fitted RV at 1-day vs 1-month."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, h_name in zip(axes, ["1d", "1m"]):
        d = acf_data_per_ticker[ticker_to_plot][h_name]
        for model_name, series in d.items():
            a = acf(series.dropna(), nlags=250, fft=True)
            ax.plot(a, label=model_name, lw=1.5)
        ax.axhline(0, color="k", lw=0.5)
        ax.axhline( 2/np.sqrt(len(d["HAR"])), color="gray", ls="--", lw=0.7)
        ax.axhline(-2/np.sqrt(len(d["HAR"])), color="gray", ls="--", lw=0.7)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF of fitted RV")
        ax.set_title(f"{ticker_to_plot}: horizon = {h_name}")
        ax.legend()
        ax.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    print(f"\nSaved ACF plot -> {outpath}")


if __name__ == "__main__":
    all_results, all_acf, all_rv = {}, {}, {}
    for t in TICKERS:
        res, ad, rv = run_ticker(t)
        all_results[t] = res
        all_acf[t] = ad
        all_rv[t] = rv

    print_relative_mse(all_results)
    plot_acf_comparison(all_acf, ticker_to_plot="AAPL", outpath="acf_aapl.png")
    plot_acf_comparison(all_acf, ticker_to_plot="JPM",  outpath="acf_jpm.png")

    # Save results to CSV so you can paste into the report
    rows = []
    for t in TICKERS:
        for h in HORIZONS:
            for m, v in all_results[t][h].items():
                rows.append({"ticker": t, "horizon": h, "model": m, "mse": v,
                             "rel_mse": v / all_results[t][h]["HAR"]})
    pd.DataFrame(rows).to_csv("results.csv", index=False)
    print("\nSaved results -> results.csv")
