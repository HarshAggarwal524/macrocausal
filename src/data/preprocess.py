"""
src/data/preprocess.py

Takes the five raw CSV files (FEDFUNDS, CPI, GS10, SP500, VIX) and produces
one clean, aligned, monthly panel ready for modeling.

Pipeline:
  1. Load each file correctly (FRED files vs Yahoo files have different
     header structures)
  2. Resample everything to a consistent monthly grid
  3. Transform each series into the statistically correct form
     (returns for prices, percent change for index levels, raw level
     for rates and VIX)
  4. Merge into a single panel, aligned by date, dropping incomplete rows
  5. Standardize (zero mean, unit variance) for neural network training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os


def load_fred_csv(path: str) -> pd.DataFrame:
    """
    Loads a FRED-sourced CSV (no extra header rows).
    FRED files have a clean structure: first row is the header,
    every row after that is a date and a value.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def load_yahoo_csv(path: str) -> pd.DataFrame:
    """
    Loads a Yahoo Finance CSV, skipping the two extra header rows
    that yfinance writes ('Price,...' and 'Ticker,...') before the
    real 'Date,...' header. Without skiprows=2, pandas tries to parse
    the word "Ticker" as a date and crashes.
    """
    df = pd.read_csv(path, index_col=0, skiprows=2)
    df.index = pd.to_datetime(df.index)
    return df


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resamples a DataFrame to monthly frequency, anchored to month-end.
    Takes the last available value within each calendar month.

    This guards against small date misalignments between sources
    (e.g. FRED stamping the 1st of the month, Yahoo stamping a
    slightly different trading day) by forcing everything onto the
    exact same monthly grid before we try to merge them.
    """
    return df.resample("ME").last()
    # "ME" = Month End frequency.
    # .last() takes the most recent value within each month —
    # appropriate for both rates (FEDFUNDS) and prices (SP500).


def compute_returns(price_series: pd.Series) -> pd.Series:
    """
    Converts a price level series into monthly log returns.

    Log returns are preferred over simple percentage returns because
    they are additive over time (this month's log return + next
    month's log return = two-month log return) and are closer to
    normally distributed, which matters for the statistical methods
    used later (VAR, Granger causality).
    """
    return np.log(price_series / price_series.shift(1))
    # shift(1) gives you last month's value, aligned against this
    # month's value. log(this_month / last_month) is the log return.
    # The very first row will be NaN, since there is no "previous
    # month" to compare the first observation against.


def compute_pct_change(level_series: pd.Series) -> pd.Series:
    """
    Converts an index level series (like CPI) into the month-over-month
    percentage change — i.e., the inflation rate. This is the standard
    convention used by economists and central banks when discussing
    inflation, so we follow it here rather than using log returns.
    """
    return level_series.pct_change()
    # pandas built-in: (this_month - last_month) / last_month
    # Like compute_returns, the first row will be NaN.


def build_macro_panel(
    fedfunds_path: str,
    cpi_path: str,
    gs10_path: str,
    sp500_path: str,
    vix_path: str,
) -> pd.DataFrame:
    """
    Loads all five raw series, transforms them appropriately,
    aligns them to a common monthly index, and returns one clean panel.
    """

    # Step 1: Load each file with the correct loader
    fedfunds = load_fred_csv(fedfunds_path)
    cpi      = load_fred_csv(cpi_path)
    gs10     = load_fred_csv(gs10_path)
    sp500    = load_yahoo_csv(sp500_path)
    vix      = load_yahoo_csv(vix_path)

    # Step 2: Resample everything to monthly, just to be safe
    fedfunds = resample_monthly(fedfunds)
    cpi      = resample_monthly(cpi)
    gs10     = resample_monthly(gs10)
    sp500    = resample_monthly(sp500)
    vix      = resample_monthly(vix)

    # Step 3: Apply the correct transformation to each series
    sp500_returns = compute_returns(sp500.iloc[:, 0]).rename("SP500_RETURN")
    cpi_inflation = compute_pct_change(cpi.iloc[:, 0]).rename("CPI_INFLATION")
    fedfunds_rate = fedfunds.iloc[:, 0].rename("FEDFUNDS")
    gs10_rate     = gs10.iloc[:, 0].rename("GS10")
    vix_level     = vix.iloc[:, 0].rename("VIX")
    # VIX is kept as a raw level (not a return) deliberately —
    # regime detection tomorrow needs its actual level
    # (VIX < 20 = calm, 20-30 = stressed, > 30 = crisis).
    # Converting it to a return would destroy the information
    # needed for that step.
    # FEDFUNDS and GS10 are already rates (percentages), not
    # cumulative levels, so they are left as-is — only renamed
    # for clarity in the final panel.

    # Step 4: Merge all five into one DataFrame, aligned by date
    panel = pd.concat(
        [fedfunds_rate, cpi_inflation, gs10_rate, sp500_returns, vix_level],
        axis=1
    )
    # axis=1 merges side-by-side as columns, automatically aligning
    # on the shared DateTimeIndex. Any date missing in one series
    # will produce a NaN in that column for that row.

    # Step 5: Drop rows with any missing values
    before = panel.shape[0]
    panel = panel.dropna()
    after = panel.shape[0]
    print(f"Dropped {before - after} rows with missing values "
          f"({before} -> {after} rows remaining)")
    # We expect to lose exactly 1 row here: the very first month,
    # because both compute_returns and compute_pct_change produce
    # NaN for the first observation (there's no "previous month" to
    # compare against). If you lose more than 1-2 rows, investigate —
    # it likely means the raw series didn't actually start on the
    # same date.

    return panel
    # NOTE: this return statement must be indented to the same level
    # as the code above it, INSIDE the function. If it is accidentally
    # dedented to sit outside the function body, Python will silently
    # return None instead of raising an error, and you will only find
    # out later when something downstream tries to call .to_csv() on
    # None and crashes. This was the exact bug from the previous step.


def standardize_panel(panel: pd.DataFrame):
    """
    Scales every column to zero mean and unit variance.
    Required for the neural network (Day 7) to train stably —
    without this, variables on very different scales (e.g. SP500
    returns ~0.01 vs FEDFUNDS ~5.0) cause unstable gradients, because
    the variable with the largest raw scale would dominate the loss
    function purely due to its scale, not its actual importance.

    Returns:
        scaled_panel: DataFrame with the same shape and column names,
                      but every value transformed to (x - mean) / std
        scaler: the fitted StandardScaler object, returned so it can
                be reused later to convert model outputs back into
                real-world units if needed
    """
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(panel)
    scaled_panel = pd.DataFrame(
        scaled_values, index=panel.index, columns=panel.columns
    )
    return scaled_panel, scaler


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    panel = build_macro_panel(
        fedfunds_path="data/raw/fred_fedfunds.csv",
        cpi_path="data/raw/fred_cpi.csv",
        gs10_path="data/raw/fred_gs10.csv",
        sp500_path="data/raw/yf_sp500.csv",
        vix_path="data/raw/yf_vix.csv",
    )

    # Save the unscaled version too — useful for human-readable inspection.
    # Scaled numbers like -0.34 are meaningless to a human eye; this file
    # lets you sanity-check real values, e.g. "was inflation really 1.2%
    # that month?"
    panel.to_csv("data/processed/macro_panel_raw_units.csv")

    scaled_panel, scaler = standardize_panel(panel)

    # This is the file your models will actually train on.
    scaled_panel.to_csv("data/processed/macro_panel.csv")

    print("\nFinal panel shape:", scaled_panel.shape)

    print("\nFirst 3 rows (scaled):")
    print(scaled_panel.head(3))

    print("\nColumn means (should be ~0):")
    print(scaled_panel.mean().round(4))

    print("\nColumn std devs (should be ~1):")
    print(scaled_panel.std().round(4))