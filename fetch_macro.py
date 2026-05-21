"""
Download the seven external M_ALL covariates that are publicly available:
  - VIX  (CBOE volatility index, Yahoo ^VIX)
  - HSI  (Hang Seng index, Yahoo ^HSI; we use its daily squared log return)
  - US3M (3-month US T-bill rate, FRED DGS3MO)
  - ADS  (Aruoba-Diebold-Scotti business conditions, Philly Fed)
  - EPU  (Economic Policy Uncertainty daily, Baker-Bloom-Davis)
  - EA   (per-ticker earnings-announcement dummy, yfinance)

The remaining two M_ALL covariates we construct from the existing OHLCV:
  - M1W  (1-week momentum)         computed in analysis_full.py
  - DVOL (Δ log dollar volume)     computed in analysis_full.py

IV is paper-only (OptionMetrics, licensed). We follow the standard public-data
workaround and proxy IV by VIX. This is a known limitation: VIX is index-level,
whereas the paper's IV is per-stock; the substitution should be flagged in
the report.

Output: data/macro.csv with daily-frequency columns aligned by date.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUT_DIR = Path("data")
OUT_DIR.mkdir(exist_ok=True)

START = "2015-01-01"   # gives us 2015 for M1W lag warm-up, then 2016+
END   = "2025-01-15"

TICKERS = ["AAPL", "AMZN", "JPM"]


def fetch_yahoo(symbol: str, label: str) -> pd.Series:
    df = yf.download(symbol, start=START, end=END,
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol}")
    # yfinance can return a MultiIndex on columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].copy()
    s.name = label
    s.index = pd.to_datetime(s.index).normalize()
    return s


def fetch_fred(series_id: str, label: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, na_values=".")
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.normalize()
    s = df.set_index("observation_date")[series_id].astype(float)
    s.name = label
    return s


def fetch_ads() -> pd.Series:
    url = ("https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
           "ads/ads_index_most_current_vintage.xlsx")
    df = pd.read_excel(url, engine="openpyxl")
    # File uses YYYY:MM:DD date format
    df["Date"] = pd.to_datetime(df["Date"].str.replace(":", "-"))
    s = df.set_index("Date")["ADS_Index"].astype(float)
    s.name = "ADS"
    s.index = s.index.normalize()
    return s


def fetch_epu_daily() -> pd.Series:
    url = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"])
    )
    s = df.set_index("date")["daily_policy_index"].astype(float)
    s.name = "EPU"
    s.index = s.index.normalize()
    return s


def fetch_earnings(ticker: str) -> pd.Series:
    """Per-ticker earnings-announcement dates from yfinance.
    Returns a Series with index = announcement dates, value = 1."""
    t = yf.Ticker(ticker)
    ed = t.get_earnings_dates(limit=80)
    if ed is None or len(ed) == 0:
        return pd.Series(dtype=int, name="EA")
    # The index is a TimezoneAwareDatetimeIndex of NY-local timestamps; normalise
    dates = pd.to_datetime(ed.index.date)
    return pd.Series(1, index=pd.DatetimeIndex(dates).normalize().unique(),
                     name="EA").sort_index()


def hsi_to_squared_return(hsi_close: pd.Series) -> pd.Series:
    log_ret = np.log(hsi_close).diff()
    out = log_ret ** 2
    out.name = "HSI"
    return out


def main():
    print("Fetching macro covariates for M_ALL ...")

    print("  VIX  (Yahoo ^VIX)")
    vix = fetch_yahoo("^VIX", "VIX")

    print("  HSI  (Yahoo ^HSI, daily squared log return)")
    hsi_close = fetch_yahoo("^HSI", "HSI_close")
    hsi = hsi_to_squared_return(hsi_close)

    print("  US3M (FRED DGS3MO, first-differenced later)")
    us3m = fetch_fred("DGS3MO", "US3M")

    print("  ADS  (Philly Fed)")
    ads = fetch_ads()

    print("  EPU  (Baker-Bloom-Davis daily)")
    epu = fetch_epu_daily()

    macro = pd.concat([vix, hsi, us3m, ads, epu], axis=1)
    macro.index = pd.to_datetime(macro.index).normalize()
    macro = macro.sort_index()

    # Forward-fill macroeconomic indicators across non-trading days (paper
    # convention for daily-aligned macro variables).
    macro = macro.ffill(limit=5)

    # Earnings dummies per ticker
    print("  Earnings dates (yfinance, per ticker)")
    for t in TICKERS:
        ea = fetch_earnings(t)
        col = f"EA_{t}"
        if len(ea) == 0:
            macro[col] = 0
        else:
            ea_full = pd.Series(0, index=macro.index, name=col)
            for d in ea.index:
                if d in ea_full.index:
                    ea_full.loc[d] = 1
            macro[col] = ea_full
        print(f"    {t}: {macro[col].sum()} announcement days")

    out_path = OUT_DIR / "macro.csv"
    macro.to_csv(out_path)
    print(f"\nSaved: {out_path}")
    print(f"  shape: {macro.shape}")
    print(f"  date range: {macro.index.min().date()} → {macro.index.max().date()}")
    print(f"  columns: {macro.columns.tolist()}")
    print(f"  NaN counts (per col):\n{macro.isna().sum()}")


if __name__ == "__main__":
    main()
