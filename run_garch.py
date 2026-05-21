"""GARCH(1,1) benchmark at h = 22 on AAPL, AMZN, JPM.

Hansen & Lunde (2005) "Does anything beat GARCH(1,1)?" — the canonical
pre-ML volatility-forecasting baseline. Audit C6.3 flagged this as
NAMED-not-EXECUTED.

Approach:
  1. Daily returns r_t for each stock (from existing 5-min features:
     daily return = sum of intraday log-returns).
  2. Roll GARCH(1,1) at each test day t: fit on r[:t], forecast h-step
     variance σ²_{t+1}, ..., σ²_{t+h}, take mean as GARCH's h-day mean-RV
     equivalent. Refit every 22 days for compute (matching ARFIMA).
  3. Compute rel-MSE vs HAR at h=22 on the same test window.

Returns alignment matched to HAR's test set via build_har_design output.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from arch import arch_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, '/Users/shakhzod/Desktop/data4rv')
from analysis_pro import (
    load_minute_bars, compute_features, build_har_design,
    HAR_FEATS, TICKERS, RV_FREQ_MIN, TRAIN_FRAC, VAL_FRAC
)


def garch_h22_forecast(returns_scaled):
    """Fit GARCH(1,1) on returns × 100 (avoids arch's numerical warnings),
    forecast h=22 days, return the mean h-step conditional variance
    (de-scaled to original return scale, i.e. divide by 100^2).
    """
    am = arch_model(returns_scaled, mean='Zero', vol='GARCH', p=1, q=1,
                     rescale=False)
    res = am.fit(disp='off', show_warning=False)
    fc = res.forecast(horizon=22, reindex=False)
    h_step_vars = fc.variance.iloc[-1].values  # shape (22,)
    mean_h22 = h_step_vars.mean()
    return mean_h22 / (100.0 ** 2)  # de-scale back


def rolling_garch_h22(returns, cut, h=22, refit_step=22):
    """Returns 1-D array of length (n - cut - h + 1)."""
    n = len(returns)
    test_idx = np.arange(cut, n - h + 1)
    preds = np.zeros(len(test_idx))
    last_pred = None

    for k, t in enumerate(test_idx):
        if k % refit_step == 0:
            history = returns[:t]
            try:
                last_pred = garch_h22_forecast(history * 100.0)
            except Exception as e:
                # Fallback: long-run variance
                last_pred = float(np.var(history[-252:]))
        preds[k] = last_pred
    return preds, test_idx


def har_h22_for_comparison(feat, h=22):
    """HAR forecast on the same test window."""
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
    print("Loading minute data and building daily returns...")
    feat = {}
    daily_returns = {}
    rv_panel = {}
    for t in TICKERS:
        df = load_minute_bars(Path(t + ".txt"))
        f = compute_features(df, freq=RV_FREQ_MIN)
        feat[t] = f
        rv_panel[t] = f["RV"]
        # Use daily return = sum of intraday 5-min log-returns
        daily_returns[t] = f["r_daily"]
        print(f"  {t}: {len(f)} days, daily-ret std = "
              f"{f['r_daily'].std():.4f}")

    print("\nRunning rolling GARCH(1,1) at h=22 with refit every 22 days...")
    rows = []
    h = 22
    for t in TICKERS:
        # HAR baseline + test set alignment
        har_pred, har_y, har_index = har_h22_for_comparison(feat[t], h=h)
        n_test = len(har_pred)

        # Align returns with HAR's HAR-design test rows
        # HAR design drops first ~22 rows for RVM lags; daily returns are full length
        # Find position of HAR test rows in daily_returns
        returns = daily_returns[t].values
        # The first test row in HAR-design corresponds to date har_index[0]
        # We need the position of har_index[0] in daily_returns[t].index
        first_test_pos = daily_returns[t].index.get_loc(har_index[0])
        cut = first_test_pos

        garch_preds, _ = rolling_garch_h22(returns, cut=cut, h=h)

        # Align lengths
        min_len = min(len(garch_preds), len(har_pred))
        garch_preds = garch_preds[-min_len:]
        har_pred_aligned = har_pred[-min_len:]
        har_y_aligned = har_y[-min_len:]

        mse_har = mean_squared_error(har_y_aligned, har_pred_aligned)
        mse_garch = mean_squared_error(har_y_aligned, garch_preds)
        rel_mse = mse_garch / mse_har
        print(f"  {t}: HAR MSE = {mse_har:.3e}  GARCH MSE = {mse_garch:.3e}  "
              f"rel-MSE = {rel_mse:.3f}")
        print(f"       mean HAR pred = {har_pred_aligned.mean():.3e}, "
              f"mean GARCH pred = {garch_preds.mean():.3e}, "
              f"mean y = {har_y_aligned.mean():.3e}")
        rows.append({"ticker": t, "rel_mse": rel_mse,
                     "mse_garch": mse_garch, "mse_har": mse_har,
                     "n_test": min_len})

    out = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("GARCH(1,1) RESULTS SUMMARY (h = 22, RV target):")
    print("=" * 60)
    print(out.to_string(index=False))
    print(f"\nCross-section mean rel-MSE: {out['rel_mse'].mean():.3f}")
    out.to_csv("garch_results.csv", index=False)
    print("\nSaved: garch_results.csv")


if __name__ == "__main__":
    main()
