# A Machine Learning Approach to Volatility Forecasting

Replication of Christensen, Siggaard and Veliyev (2023), *"A Machine Learning Approach to Volatility Forecasting"*, *Journal of Financial Economics*.

Coursework report for IFTE0004 (Financial Analytics and Machine Learning), UCL Institute of Finance and Technology. The accompanying write-up is in [main.tex](main.tex); compiled output is the three-page body plus appendix described in the report.

---

## Scope

The original paper runs a 22-model horse race across 29 DJIA constituents over 2001–2017 ($T = 4{,}257$). This replication tests the five headline findings on a smaller cross-section (AAPL, AMZN, JPM) over 2016–2024 (2,264 NYSE sessions), and asks whether CSV's long-horizon machine-learning advantage survives a different regime composition.

The code in this repository covers the eight headline models used in the body of the report:

- HAR (Corsi, 2009) — OLS benchmark
- LogHAR — log-RV with Jensen bias correction
- Elastic Net (Zou and Hastie, 2005)
- Random Forest (Breiman, 2001)
- Feed-forward neural network, 4 → 2 ReLU, 10-seed ensemble (subset of CSV's NN$_2$)

The full 22-model contest (LevHAR, SHAR, HARQ, Ridge/Lasso/Post-Lasso, Bagging, Gradient Boosting, NN$_1$–NN$_4$), ALE attribution, MCS retention and the VaR application are reported in the paper appendix but not in this minimal replication script.

Two information sets are used in the paper: M_HAR (three HAR lags) and M_ALL (HAR lags plus eight macro / market covariates). The script here runs M_HAR only; the macro feature build sits in `fetch_macro.py` and is documented in the report.

## Data

Three text files of 1-minute OHLCV bars are expected in the project root:

```
AAPL.txt
AMZN.txt
JPM.txt
```

Each is a headerless CSV with seven columns: `date, time, open, high, low, close, volume`, where `date` is `MM/DD/YYYY` and `time` is `HH:MM` (US/Eastern, full session, 390 bars/day). The exact parsing assumed by the loader is in [replicate.py:50](replicate.py#L50).

These files are not distributed with the repository. The data used in the report were obtained via institutional access (Refinitiv Tick History); commercially equivalent sources include Kibot, AlgoSeek, FirstRate Data and Polygon.io. Any 1-minute dataset that produces ≥ 380 bars on regular trading days will work; the loader runs a coverage diagnostic (`diagnose()`) on startup that flags truncated sessions or non-standard time grids.

Realised variance is computed as the sum of squared 5-minute log-returns within each session, following Andersen and Bollerslev (1998). Overnight returns are excluded. Days with fewer than 10 valid intra-day returns (typically half-day sessions around US holidays) are dropped from the RV series.

## Reproducing the headline results

```bash
pip install -r requirements.txt
python replicate.py
```

Expected runtime is 3–5 minutes on a recent laptop; the Random Forest and the 10-seed NN ensemble dominate the wall-clock. The script produces:

- Console diagnostics for each ticker (session coverage, RV summary, train/val/test sizes)
- A relative-MSE table versus HAR at $h = 1$ and $h = 22$
- `results.csv` — full MSE matrix (ticker × horizon × model)
- `acf_aapl.png`, `acf_jpm.png` — ACF of fitted RV (the mechanism plot from CSV Figure 8)

The seed is fixed at 42 for the NumPy global state and the sklearn estimators, so re-runs are deterministic up to BLAS thread non-determinism in Random Forest.

## Train / validation / test protocol

A chronological 70 / 10 / 20 split is used throughout. For the sample window above the test set begins in March 2023. K-fold cross-validation is deliberately avoided for hyperparameter selection — RV is strongly autocorrelated, and random folds leak future information into validation; tuning is on the held-out 10% block instead.

The first 22 observations of each series are lost to the monthly HAR lag. The 1-month target ($h = 22$) is the average RV from $t+1$ to $t+22$, which removes a further 22 observations from the end of the training data per the no-look-ahead convention.

Negative point forecasts are floored at the in-sample RV minimum (CSV's "insanity filter"). Reported MSE is computed on the original RV scale, not on log-RV; LogHAR predictions are back-transformed with the standard $\exp(\hat{y} + \tfrac{1}{2}\hat\sigma^2)$ correction before scoring.

## Files

| File | Purpose |
|---|---|
| [replicate.py](replicate.py) | End-to-end pipeline: data load, RV construction, model fits, evaluation, plots |
| [main.tex](main.tex) | LaTeX source for the coursework report |
| [requirements.txt](requirements.txt) | Python dependencies |
| [LICENSE](LICENSE) | MIT |

## Known deviations from the paper

These are documented in full in §2 of the report; the key ones a reader should be aware of when comparing numbers to CSV Table 2:

1. **Sample window** — 2016–2024 here vs 2001–2017 in CSV. The test window in this replication therefore spans a different volatility regime (post-COVID, pre- and post-2024 Aug carry-trade unwind), and the long-horizon ML advantage reported in CSV is sensitive to that.
2. **Cross-section** — 3 stocks vs 29. Significance tests at the cross-sectional level are not meaningful here and are not reported.
3. **NN ensemble size** — 10 seeds (this script) vs CSV's top-10-of-100. The paper's protocol is reported in the appendix contest.
4. **M_ALL covariates** — implied volatility (OptionMetrics, licensed) is proxied by VIX, reducing M_ALL from 12 to 11 features. This is a deliberate replicability choice: the headline M_ALL gains in CSV are not reproducible without a paid OptionMetrics subscription, which the paper does not flag.

## Citation

If you build on this code, please cite the original paper:

> Christensen, K., Siggaard, M., and Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Economics*.

## License

MIT — see [LICENSE](LICENSE).
