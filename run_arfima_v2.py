"""G4 v2 — ARFIMA(1, d̂, 1) benchmark using cleaner AR(∞) reconstruction.

Previous v1 used truncated frac_int which produced 1000× errors. v2:
  1. Estimate d̂ via GPH on log-RV.
  2. Apply fractional differencing to get stationary z.
  3. Fit ARMA(1,1) on z (with mean restored after).
  4. To forecast log_rv_{t+k}, use the AR(∞) representation:
       log_rv_{t+k} = z_{t+k} + Σ_j π_j log_rv_{t+k-j}
     where π_j come from -(1-L)^d weight expansion, and history is
     extended with previously-forecasted log_rv values.
  5. h=22 forecast aggregated as mean of h-step log-RV forecasts,
     mapped to RV space with Jensen correction.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

import sys
sys.path.insert(0, '/Users/shakhzod/Desktop/data4rv')
from analysis_pro import (
    load_minute_bars, compute_features, build_har_design,
    HAR_FEATS, TICKERS, RV_FREQ_MIN, TRAIN_FRAC, VAL_FRAC
)


def gph_d_hat(y, m=None):
    y = np.asarray(y) - np.mean(y)
    T = len(y)
    if m is None:
        m = int(T ** 0.5)
    fft = np.fft.fft(y)
    perio = (np.abs(fft) ** 2) / (2 * np.pi * T)
    freqs = 2 * np.pi * np.arange(1, m + 1) / T
    log_perio = np.log(perio[1:m + 1])
    x = np.log(4 * np.sin(freqs / 2) ** 2)
    reg = LinearRegression().fit(x.reshape(-1, 1), log_perio)
    return float(-reg.coef_[0])


def frac_diff_weights(d, n):
    """Coefficients of (1-L)^d: w_0=1, w_k = w_{k-1} * (k-1-d)/k."""
    w = np.zeros(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w


def frac_diff(y, d, max_lag=500):
    """Apply (1-L)^d to y: z_t = Σ_{k=0}^{min(t, max_lag-1)} w_k y_{t-k}."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = frac_diff_weights(d, max_lag)
    out = np.zeros(n)
    for t in range(n):
        k_max = min(t + 1, max_lag)
        out[t] = np.dot(w[:k_max], y[t::-1][:k_max])
    return out


def arfima_h_forecast(log_rv_history, h=22, max_lag=500, p=1, q=1):
    """Forecast h log-RV values using ARFIMA(p, d̂, q).

    AR(∞) recursion: log_rv_t = (1 - (1-L)^d) log_rv_t + ε_t-like
    where the (1-L)^d expansion gives weights {w_k}; thus
      log_rv_t = Σ_{k=1}^∞ -w_k log_rv_{t-k} + z_t
    where z_t are i.i.d.-ish (after ARMA filtering).

    Forecast: at each step t+1, the z forecast comes from the fitted ARMA;
    log_rv_{t+1} = z_forecast + Σ_{k=1}^∞ -w_k log_rv_{t+1-k},
    using past log_rv (including previously-forecasted values).
    """
    log_rv_history = np.asarray(log_rv_history, dtype=float)
    d = gph_d_hat(log_rv_history)
    d = float(np.clip(d, 0.05, 0.49))

    # Fractionally difference (gives stationary z)
    z_full = frac_diff(log_rv_history, d, max_lag=max_lag)
    # Discard burn-in (truncated weights unreliable)
    burn = min(max_lag, len(z_full) // 5)
    z_clean = z_full[burn:]

    # Fit ARMA(p, q) on z (mean-centred)
    mu = z_clean.mean()
    try:
        model = ARIMA(z_clean - mu, order=(p, 0, q), trend='n')
        fit = model.fit(method='innovations_mle')
        z_forecast_centered = fit.forecast(steps=h)
        z_forecast = z_forecast_centered + mu
    except Exception:
        z_forecast = np.full(h, mu)

    # AR(∞) reconstruction: weights are negative of (1-L)^d weights for k>=1
    w = frac_diff_weights(d, max_lag)
    pi = -w  # log_rv_t = z_t - Σ_{k=1} w_k log_rv_{t-k}
    # Actually: (1-L)^d log_rv_t = z_t  =>  log_rv_t = z_t + Σ_{k=1} (-w_k) log_rv_{t-k}
    # Equivalently log_rv_t = z_t - Σ_{k=1}^∞ w_k log_rv_{t-k}

    history = list(log_rv_history)
    log_rv_forecasts = []
    for k in range(h):
        new = z_forecast[k]
        # Subtract Σ w_k log_rv_{t-k} for k >= 1 (using w with negative sign convention)
        n_hist = min(len(history), max_lag - 1)
        for j in range(1, n_hist + 1):
            new -= w[j] * history[-j]
        history.append(new)
        log_rv_forecasts.append(new)

    return d, np.array(log_rv_forecasts)


def har_h22_forecast(feat, h=22):
    df = build_har_design(feat, h=h)
    n = len(df)
    n_tr, n_va = int(n * TRAIN_FRAC), int(n * VAL_FRAC)
    cut = n_tr + n_va
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    m = LinearRegression().fit(train[HAR_FEATS], train["y"])
    pred = m.predict(test[HAR_FEATS])
    return pred, test["y"].values, test.index


def main():
    print("Loading and building RV series...")
    feat = {}
    rv_panel = {}
    for t in TICKERS:
        df = load_minute_bars(Path(t + ".txt"))
        f = compute_features(df, freq=RV_FREQ_MIN)
        feat[t] = f
        rv_panel[t] = f["RV"]

    h = 22
    rows = []
    for t in TICKERS:
        rv_t = rv_panel[t]
        log_rv = np.log(np.maximum(rv_t.values, 1e-12))

        # HAR baseline (gives test set with proper alignment)
        har_pred, har_y, har_index = har_h22_forecast(feat[t], h=h)
        n_test = len(har_pred)
        print(f"\n=== {t} (test n={n_test}) ===")

        # ARFIMA forecast aligned with HAR's test set
        # For each test day i in HAR's test set, ARFIMA uses log_rv up to that day
        # Refit ARFIMA every 22 days for speed
        arfima_preds = np.zeros(n_test)
        d_hats = []
        for i in range(n_test):
            # Position in raw RV time series for HAR's test row i
            har_test_date = har_index[i]
            # Find position in rv_t
            try:
                t_pos = rv_t.index.get_loc(har_test_date)
            except KeyError:
                # Use position approximate
                t_pos = len(rv_t) - n_test + i
            if i % 22 == 0:
                history = log_rv[:t_pos]
                d, fc = arfima_h_forecast(history, h=h)
                d_hats.append(d)
                # h-day mean log-RV forecast; convert to mean RV with Jensen correction
                mean_log_fc = fc.mean()
                # Variance of innovations from fit (use sample variance of recent z)
                # Use a simple Jensen correction with σ² estimated from recent fractional residuals
                burn = min(500, len(history) // 5)
                z_full = frac_diff(history, d, max_lag=500)
                z_resid = z_full[burn:]
                sigma2 = float(np.var(z_resid))
                rv_fc = float(np.exp(mean_log_fc + 0.5 * sigma2))
            arfima_preds[i] = rv_fc

        d_hat_mean = np.mean(d_hats)
        # HAR aligned
        mse_har = mean_squared_error(har_y, har_pred)
        mse_arfima = mean_squared_error(har_y, arfima_preds)
        rel_mse = mse_arfima / mse_har
        print(f"  d̂ = {d_hat_mean:.3f}")
        print(f"  HAR MSE       = {mse_har:.3e}, mean pred {har_pred.mean():.3e}, mean y {har_y.mean():.3e}")
        print(f"  ARFIMA MSE    = {mse_arfima:.3e}, mean pred {arfima_preds.mean():.3e}")
        print(f"  rel-MSE       = {rel_mse:.3f}")
        rows.append({"ticker": t, "d_hat": d_hat_mean,
                     "mse_arfima": mse_arfima, "mse_har": mse_har,
                     "rel_mse": rel_mse,
                     "mean_arfima": arfima_preds.mean(),
                     "mean_y": har_y.mean()})

    out = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("ARFIMA RESULTS (h = 22, M_HAR-equivalent):")
    print("=" * 60)
    print(out.to_string(index=False))
    print(f"\nCross-section mean rel-MSE: {out['rel_mse'].mean():.3f}")
    out.to_csv("arfima_results.csv", index=False)


if __name__ == "__main__":
    main()
