# MacroCausal: Learning What Drives What in Financial Markets

MacroCausal is a research-oriented machine learning project that discovers dynamic causal relationships between macroeconomic indicators and financial market variables using time-series analysis.

The project addresses the limitation of traditional financial machine learning models that rely on correlation by learning directional cause-and-effect relationships that evolve across different market regimes.

The core methodology combines Neural Granger Causality, automatic market regime detection, and rolling walk-forward evaluation to build interpretable causal graphs from multivariate financial time series.

The system uses monthly macroeconomic data from the Federal Reserve Economic Data (FRED) database and market data from Yahoo Finance, including the Federal Funds Rate, Consumer Price Index (CPI), 10-Year Treasury Yield, S&P 500 Index, and VIX.

The final output consists of regime-aware causal graphs, market regime visualizations, walk-forward stability analysis, and quantitative comparisons with established macroeconomic theory.
