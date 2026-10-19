---
license: mit
tags:
- finance
size_categories:
- 1K<n<10K
---

# Unofficial Antam Gold Price History (IDR)

This repository provides an unofficial historical record of Antam gold selling prices in Indonesian Rupiah (IDR), compiled from publicly accessible information on the official Antam Logam Mulia website.

It is intended for research, analysis, and educational use cases such as time series analysis, forecasting experiments, and financial data tutorials.

The series covers roughly 2010-01-04 through 2026-08-03 (about 5,100 records).

## Dataset Overview

Main file:

- `Antam_historical_gold_prices.csv`
  - `Time (ms)`: Unix timestamp in milliseconds, as returned by the source chart API.
  - `Gold Price`: Antam gold selling price in IDR per gram.
  - `Date`: UTC datetime derived from `Time (ms)`, formatted `YYYY-MM-DD HH:MM:SS`.

Raw files (under `raw/`):

- `raw/Antam_historical_gold_prices_raw.csv`
  - `Time (ms)`, `Gold Price` only — the source pairs without the derived `Date` column.
- `raw/sell.htm`
  - A saved capture of the source endpoint response, kept for provenance.

## Example Usage

Using `pandas` directly with the CSV:

```python
import pandas as pd

df = pd.read_csv("Antam_historical_gold_prices.csv", parse_dates=["Date"])
print(df.head())
```

Using the `datasets` library:

```python
from datasets import load_dataset

ds = load_dataset("theonegareth/antam_historical_gold_prices", data_files="Antam_historical_gold_prices.csv")
df = ds["train"].to_pandas()
print(df.head())
```

## Potential Use Cases

- Time series analysis:
  - Trend and regime changes in local gold prices.
  - Volatility and drawdown analysis.
- Forecasting experiments:
  - ARIMA / SARIMA
  - Prophet
  - Gradient boosting / tree-based models
  - Neural networks (LSTM/Transformer) for time series.
- Educational materials:
  - Feature engineering with financial time series.
  - Time-based train-test splits.
  - Model evaluation and overfitting discussions.
- Benchmarking:
  - Compare Antam local gold prices against:
    - International gold prices,
    - IDR exchange rates,
    - Inflation or macroeconomic indicators.

## Data Preparation Notes

How the files are built:

- The source endpoint returns the full history as `[Time (ms), Gold Price]` pairs.
- `raw/Antam_historical_gold_prices_raw.csv` stores those pairs as-is.
- `Antam_historical_gold_prices.csv` adds a `Date` column computed as the UTC time of `Time (ms)`.
- Records are sorted chronologically and deduplicated by `Time (ms)`.

## Attribution and Source

- Underlying price information originates from publicly accessible pages on:
  - `https://www.logammulia.com`
  - Specifically: the "Grafik Emas Harian" (daily gold chart) and related public price views.
- This repository contains a cleaned, tabular representation created for analytical convenience.

Please always verify current prices directly from the official website.

## Important Notes and Disclaimer

- This dataset is:
  - Unofficial.
  - For research, analysis, and educational purposes only.
- This project is:
  - Not affiliated with,
  - Not sponsored by, and
  - Not endorsed by
  - PT Aneka Tambang Tbk / Antam Logam Mulia or any related entity.
- All trademarks, logos, and brand names are the property of their respective owners.
- Users are responsible for ensuring their use of this dataset complies with:
  - The terms and conditions of the original data source,
  - Applicable laws and regulations,
  - Their own organizational policies.
- For authoritative, complete, and up-to-date information on Antam gold prices, always refer directly to:
  - `https://www.logammulia.com`

If you are a rights holder and have concerns or requests (e.g. corrections, restrictions, takedown), please open an issue on the repository or contact the maintainer, and we will respond promptly.
