# Time Series Module (`series`)

The `series` module in tinyshift provides a set of tools for quantitative time series analysis, focusing on key characteristics for MLOps, forecasting, and pattern detection. It covers metrics and transformations for volatility, intermittency, seasonality strength, trend, and entropy.

## Features

### 1. Outlier Detection & Volatility

- **`hampel_filter`**  
  Robustly detects outliers using the median and median absolute deviation (MAD) in a rolling window.  
  **When to use:** To identify outliers in financial series, sensor data, or any data sensitive to extreme noise.

- **`bollinger_bands`**  
  Computes Bollinger Bands, which indicate periods of high or low volatility.  
  **When to use:** To detect volatility regime changes, breakouts, and overbought/oversold zones.

### 2. Forecastability, Entropy & Intermittency

- **`foreca`**  
  Measures the forecastability (omega) of a series, based on spectral entropy.  
  **When to use:** To assess the predictability potential of a series. Values near 1 indicate strong structure (low entropy), near 0 indicate noise (high entropy).

- **`adi_cv`**  
  Computes Average Days of Inventory (ADI) and Coefficient of Variation (CV), useful for classifying series by intermittency and variability.  
  **When to use:**  
    - **High ADI:** intermittent series (e.g., sporadic sales)
    - **High CV:** erratic/lumpy series (high variability)
    - **Low ADI & Low CV:** smooth, predictable series

### 3. Trend, Seasonality & Indicators

- **`hurst_exponent`**  
  Measures long-term trend (persistence or anti-persistence) in a series.  
  **When to use:**  
    - H < 0.5: mean-reverting (anti-persistent)
    - H ≈ 0.5: random walk
    - H > 0.5: persistent trend

- **`relative_strength_index` (RSI)**  
  Momentum oscillator, useful for identifying overbought/oversold zones and trend strength.

## Summary: When to use each function?

| Metric/Function              | Range         | Interpretation                                              | Question You Want to Answer                                      | Recommended Usage                                              |
|------------------------------|--------------|-------------------------------------------------------------|------------------------------------------------------------------|---------------------------------------------------------------|
| **Hampel Filter**            | 0 or 1       | Outlier presence (per point)                                | “Is this point an outlier compared to its local window?”         | Detecting local outliers/volatility in time series             |
| **Bollinger Bands**          | 0 or 1       | Outlier/volatility regime (per point)                       | “Is this value outside the expected volatility range?”           | Identifying volatility shifts, overbought/oversold conditions  |
| **Forecastability (ForeCA)** | 0 → 1        | Forecastability (1 = highly predictable, 0 = noise)         | “How predictable is this time series?”                           | Assessing predictability, strength of seasonality/trend        |
| **ADI / CV**                 | ADI: 1 → ∞   | ADI: Intermittency; CV: Variability                         | “Is this series intermittent or erratic?”                        | Classifying demand: smooth, intermittent, erratic, lumpy       |
| **Hurst Exponent**           | 0 → 1        | Trend persistence (H>0.5: persistent, H<0.5: mean-reverting)| “Does this series have a persistent trend or mean reversion?”    | Detecting long-term memory, trend, or random walk              |
| **Relative Strength Index (RSI)**  | 0 → 100      | Momentum/overbought-oversold indicator                      | “Is the series overbought or oversold?”                          | Identifying momentum, trend strength, reversal points          |
