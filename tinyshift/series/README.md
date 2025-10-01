# Time Series Module (`series`)

The `series` module of tinyshift provides quantitative tools for time series analysis, focusing on key features for MLOps, forecasting, and pattern detection. It covers metrics and transformations for volatility, intermittency, seasonality strength, trend, entropy, and complexity.

## Features

### 1. Outlier Detection & Volatility

- **`hampel_filter`**  
  Robustly detects outliers using the median and median absolute deviation (MAD) in a moving window.  
  **When to use:** To identify outliers in w series, sensors, or any data sensitive to extreme noise.

- **`bollinger_bands`**  
  Calculates Bollinger Bands, indicating periods of high or low volatility.  
  **When to use:** To detect volatility regime changes, breakouts, and overbought/oversold zones.

### 2. Forecastability, Entropy, Intermittency & Complexity

- **`foreca`**  
  Measures the forecastability (omega index) of a series, based on spectral entropy.  
  **When to use:** To assess the predictability potential of a series. Values close to 1 indicate strong structure (low entropy), values near 0 indicate noise (high entropy).

- **`adi_cv`**  
  Calculates Average Days of Inventory (ADI) and Coefficient of Variation (CV), useful for classifying series by intermittency and variability.  
  **When to use:**  
    - **High ADI:** intermittent series (e.g., sporadic sales)
    - **High CV:** erratic/lumpy series (high variability)
    - **Low ADI & CV:** smooth and predictable series

- **`sample_entropy` (SampEn)**  
  Computes Sample Entropy, a robust measure of complexity and irregularity in time series. Low values indicate more regularity (repetitive patterns), high values indicate greater complexity (less repetition, more "randomness").  
  **When to use:** To quantify the complexity/irregularity of a series, especially in physiological, financial, or industrial contexts. Useful for comparing variability patterns between series or periods.

- **`entropy_volatility`**  
  Measures the volatility of a time series based on the entropy of returns or differences. Higher values indicate greater uncertainty/volatility in the series dynamics.  
  **When to use:** To quantify the local instability or unpredictability of a series, especially useful in financial series, sensors, and industrial processes.

- **`maximum_achievable_accuracy`**  
  Calculates the theoretical maximum predictability of a time series, based on Shannon entropy. Indicates the upper limit of how predictable a series can be, given its distribution pattern.  
  **When to use:** To assess the maximum potential accuracy of predictive models, or compare the predictability limit between different series.

### 3. Trend & Memory

- **`hurst_exponent`**  
  Estimates the Hurst exponent and p-value for the random walk hypothesis. The Hurst exponent measures trend persistence or long-term memory in a time series.  
  **When to use:**  
    - H < 0.5: anti-persistent series (tends to mean-revert)
    - H ≈ 0.5: random walk
    - H > 0.5: persistent series (long-term trend)  
  The p-value indicates whether the random walk hypothesis can be rejected.

- **`relative_strength_index` (RSI)**  
  Momentum oscillator, useful for identifying overbought/oversold zones and trend strength.

## Summary: When to use each function?

| Metric/Function              | Range         | Interpretation                                             | Question You Want to Answer                                         | Recommended use                                              |
|------------------------------|--------------|------------------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------|
| **Hampel Filter**            | 0 or 1       | Outlier presence (per point)                               | “Is this point an outlier relative to its local window?”    | Detect local outliers/volatility in time series              |
| **Forecastability (ForeCA)** | 0 → 1        | Forecastability (1 = highly predictable, 0 = noise)        | “How predictable is this time series?”                      | Assess predictability, seasonality/trend strength            |
| **ADI / CV**                 | ADI: 1 → ∞   | ADI: Intermittency; CV: Variability                        | “Is this series intermittent or erratic?”                   | Classify demand: smooth, intermittent, erratic, lumpy        |
| **Sample Entropy (SampEn)**  | 0 → ∞        | Complexity/regularity (low = more regular, high = more complex) | “How complex or irregular is this time series?”             | Quantify complexity, compare variability patterns            |
| **Hurst Exponent**           | 0 → 1        | Trend persistence/long-term memory                         | “Does the series have persistent trend or mean-revert?”     | Detect long memory, trend, or random walk                    |
| **Entropy Volatility**       | 0 → log(N)   | Volatility based on entropy of returns                     | “How uncertain/volatile is the local series dynamics?”       | Quantify instability, compare volatility between series      |
| **Maximum Achievable Accuracy** | 0 → 1     | Theoretical predictability limit (1 = fully predictable)   | “What is the maximum achievable forecast accuracy for this series?” | Assess model limits, compare series                        |
| **Bollinger Bands**          | 0 or 1       | Signals breakouts and volatility regimes                   | “Is the value outside the expected volatility range?”        | Identify volatility changes, overbought/oversold zones       |
| **Relative Strength Index (RSI)** | 0 → 100 | Relative strength/momentum index                           | “Is the series overbought or oversold?”                      | Detect trend strength, reversal points                       |
