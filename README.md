# A Machine Learning Approach to Volatility Forecasting

Replication of Christensen, Siggaard and Veliyev (2023), *"A Machine Learning Approach to Volatility Forecasting"*, *Journal of Financial Econometrics* 21(5), 1680–1727.

Coursework report for IFTE0004 (Financial Analytics and Machine Learning), UCL Institute of Finance and Technology. The accompanying write-up is in [main.tex](main.tex); the three-page body plus appendix corresponds to the compiled PDF described in the report.

---

## Scope

The original paper runs a 22-model horse race across 29 DJIA constituents over 2001–2017 ($T = 4{,}257$). This replication tests the five headline findings on a smaller cross-section (AAPL, AMZN, JPM) over 2016–2024 (2,264 NYSE sessions), and asks whether CSV's long-horizon machine-learning advantage survives a different regime composition.

The repository implements the full 22-model contest on both information sets (M_HAR and M_ALL), plus two extensions to the paper that are not in CSV: an ARFIMA(1, d̂, 1) long-memory benchmark (motivated by Diebold, 2021) and a GARCH(1,1) benchmark in the style of Hansen and Lunde (2005).

Headline models (8-model contest, body of report):

- HAR (Corsi, 2009) — OLS benchmark
- LogHAR — log-RV with Jensen bias correction
- LevHAR (Corsi and Renò, 2012)
- SHAR (Patton and Sheppard, 2015) — signed semivariance decomposition
- HARQ (Bollerslev, Patton and Quaedvlieg, 2016) — RVD × √RQ interaction
- Elastic Net (Zou and Hastie, 2005)
- Random Forest (Breiman, 2001)
- Feed-forward neural network, 4 → 2 ReLU, top-3-of-10 seed ensemble

Appendix 22-model contest adds Ridge, Lasso, Adaptive Lasso, Post-Lasso, Bagging, Gradient Boosting, NN$_1$–NN$_4$ in both single-fit and 10-of-100 ensemble form, on both M_HAR and M_ALL.

## Data

### Intraday OHLCV (required, not distributed)

Three text files of 1-minute OHLCV bars are expected in the project root:

```
AAPL.txt
AMZN.txt
JPM.txt
```

Each is a headerless CSV with seven columns: `date, time, open, high, low, close, volume`, where `date` is `MM/DD/YYYY` and `time` is `HH:MM` (US/Eastern, full session, 390 bars/day). The exact parsing assumed by the loader is in [replicate.py:50](replicate.py#L50).

These files are not distributed with the repository because of size (~140 MB total) and licensing. Commercially equivalent sources include Kibot, AlgoSeek, FirstRate Data and Polygon.io. Any 1-minute dataset that produces ≥ 380 bars on regular trading days will work; the loader runs a coverage diagnostic that flags truncated sessions or non-standard time grids.

Realised variance is computed as the sum of squared 5-minute log-returns within each session, following Andersen and Bollerslev (1998). Overnight returns are excluded. Days with fewer than 10 valid intra-day returns are dropped from the RV series.

### Macro covariates (M_ALL information set)

Seven of CSV's nine M_ALL covariates are pulled from public sources by [fetch_macro.py](fetch_macro.py):

| Covariate | Source |
|---|---|
| VIX | Yahoo Finance (`^VIX`) — proxies CSV's per-stock OptionMetrics IV |
| HSI | Yahoo Finance (`^HSI`) — daily squared log-return |
| US3M | FRED (`DGS3MO`) |
| ADS | Philadelphia Fed (Aruoba–Diebold–Scotti) |
| EPU | Baker, Bloom and Davis daily EPU index |
| EA | yfinance earnings-announcement dummy, per ticker |
| M1W, $VOL | Constructed from OHLCV inside [analysis_full.py](analysis_full.py) |

OptionMetrics IV (licensed) is not available and is proxied by VIX; this reduces M_ALL from 12 to 11 features. See §2 of the report for the implication for headline comparability.

## Reproducing the analysis

```bash
pip install -r requirements.txt
```

Each entry below maps a report claim to the script that produced it. Run order: macro fetch → headline contest → appendix contest → benchmarks → report build.

| Report claim | Script | Output |
|---|---|---|
| Headline relative-MSE (Table 2), QLIKE (Table 3), per-stock QLIKE (Table 3b), decile plot (Figure 6) | [analysis_pro.py](analysis_pro.py) | `results_pro.csv`, `fig5_decile_mse.png` |
| 22-model contest on M_HAR + M_ALL (Tables 4, 5, 6, 7) | [analysis_full.py](analysis_full.py) | `results_full.csv`, `mcs_results.csv`, `var_results.csv`, `ale_importance.csv`, `tab_full_*.csv` |
| ARFIMA(1, d̂, 1) benchmark cited in Insight (rel-MSE 0.85 at h=22) | [run_arfima_v2.py](run_arfima_v2.py) | `arfima_results.csv` |
| GARCH(1,1) benchmark cited in Insight (rel-MSE 0.66 at h=22) | [run_garch.py](run_garch.py) | `garch_results.csv` |
| M_ALL covariate build (VIX, HSI, US3M, ADS, EPU, EA) | [fetch_macro.py](fetch_macro.py) | `data/macro.csv` |
| Summary aggregation across scripts (cross-section means, formatted tables) | [summarise_full.py](summarise_full.py) | regenerates `tab_full_*.md` and the headline LaTeX rows |
| Aggregation into final LaTeX tables and figures | [build_report.py](build_report.py) | `tab*.md`, `tab*.csv`, `fig*.png` |
| Minimal 5-model headline (HAR / LogHAR / EN / RF / NN on M_HAR at h=1, h=22), self-contained | [replicate.py](replicate.py) | `results.csv`, `acf_aapl.png`, `acf_jpm.png` |

The fastest path to the headline numbers from a clean checkout:

```bash
python fetch_macro.py          # ~30s, requires internet
python analysis_pro.py         # ~5 min, produces 8-model headline
python analysis_full.py        # ~25 min, produces 22-model appendix
python run_arfima_v2.py        # ~5 min, ARFIMA benchmark
python run_garch.py            # ~1 min, GARCH(1,1) benchmark
python build_report.py         # regenerates LaTeX tables and figures
```

[replicate.py](replicate.py) is a standalone simplified script intended as a self-contained sanity check — it only fits 5 models on M_HAR and does not require any of the other modules. It is not what produced the full appendix tables in the report.

## Train / validation / test protocol

A chronological 70 / 10 / 20 split is used throughout. For the 2016–2024 sample window the test set begins in March 2023. K-fold cross-validation is deliberately avoided for hyperparameter selection — RV is strongly autocorrelated, and random folds leak future information into validation; tuning is on the held-out 10% block instead.

The first 22 observations of each series are lost to the monthly HAR lag. The $h = 22$ target is the average RV from $t+1$ to $t+22$, which removes a further 22 observations from the end of the training data per the no-look-ahead convention.

Negative point forecasts are floored at the in-sample RV minimum (CSV's "insanity filter"). The 8-model headline contest adds a universal $2\times$ in-sample-max ceiling — see §2 of the report for the rationale. Reported MSE is computed on the original RV scale, not on log-RV; LogHAR predictions are back-transformed with the standard $\exp(\hat{y} + \tfrac{1}{2}\hat\sigma^2)$ correction before scoring.

## Reproducibility

Random seed 42 is set for NumPy's global state and for every sklearn estimator that accepts one (`MLPRegressor`, `RandomForestRegressor`, `ElasticNetCV`, `BaggingRegressor`, `GradientBoostingRegressor`). The 10-seed NN ensemble iterates over seeds 0–9, ranks by held-out validation MSE, and averages the top 3 in [analysis_pro.py](analysis_pro.py) (top 10 of 100 in [analysis_full.py](analysis_full.py), matching CSV).

Results are deterministic up to BLAS thread non-determinism in Random Forest; set `OMP_NUM_THREADS=1` for bit-exact replication.

Python 3.10+ is required. Library versions are pinned with lower bounds in [requirements.txt](requirements.txt); the report was produced with NumPy 1.26, Pandas 2.1, scikit-learn 1.3, statsmodels 0.14, arch 6.2.

## Files

| File | Purpose |
|---|---|
| [analysis_pro.py](analysis_pro.py) | 8-model headline contest (Tables 2, 3, 3b; Figure 6) |
| [analysis_full.py](analysis_full.py) | 22-model contest on M_HAR + M_ALL (Tables 4–7) |
| [run_arfima_v2.py](run_arfima_v2.py) | ARFIMA(1, d̂, 1) long-memory benchmark |
| [run_garch.py](run_garch.py) | GARCH(1,1) variance-equation benchmark |
| [fetch_macro.py](fetch_macro.py) | Downloads and aligns the M_ALL macro covariates |
| [summarise_full.py](summarise_full.py) | Aggregates per-stock results into cross-section tables |
| [build_report.py](build_report.py) | Generates the LaTeX tables and figures from result CSVs |
| [replicate.py](replicate.py) | Standalone 5-model headline (self-contained sanity check) |
| [main.tex](main.tex) | LaTeX source for the coursework report |
| [requirements.txt](requirements.txt) | Python dependencies |
| [LICENSE](LICENSE) | MIT |

## Known deviations from the paper

These are documented in full in §2 of the report; the key ones a reader should be aware of when comparing numbers to CSV Table 2:

1. **Sample window** — 2016–2024 here vs 2001–2017 in CSV. The test window in this replication spans a different volatility regime (post-COVID, the August 2024 carry-trade unwind), and CSV's long-horizon ML advantage is sensitive to that.
2. **Cross-section** — 3 stocks vs 29. Cross-sectional standard errors scale by $\sqrt{29/3} \approx 3.1\times$ under equal-per-stock noise; significance at the panel level is not meaningful and is not reported.
3. **NN ensembling** — top 3 of 10 seeds in the 8-model headline; paper-exact top 10 of 100 in the 22-model appendix contest. Drop-out, learning-rate shrinkage and early stopping are not implemented.
4. **M_ALL covariates** — implied volatility (OptionMetrics, licensed) is proxied by VIX, reducing M_ALL from 12 to 11 features. CSV's M_ALL headline gains are not reproducible without an OptionMetrics subscription, a deployment constraint CSV do not discuss.
5. **Activation** — standard ReLU instead of Leaky-ReLU (sklearn limitation; CSV Footnote 7 confirms negligible effect).

## Citation

If you build on this code, please cite the original paper:

> Christensen, K., Siggaard, M., and Veliyev, B. (2023). A machine learning approach to volatility forecasting. *Journal of Financial Econometrics*, 21(5), 1680–1727.

## License

MIT — see [LICENSE](LICENSE).
