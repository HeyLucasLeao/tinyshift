# Visualization Module (`plot`)

The `plot` module provides comprehensive visualization tools for exploratory data analysis, correlation analysis, and time series diagnostics. Built on Plotly for interactive, publication-ready visualizations that support both statistical analysis and MLOps monitoring workflows.

## Features

### 1. Correlation Analysis (`correlation.py`)

#### **`corr_heatmap`**
Generates an interactive correlation heatmap with diverging color scale and automatic feature handling.

```python
from tinyshift.plot import corr_heatmap
import numpy as np

# Basic usage
X = np.random.randn(100, 5)
corr_heatmap(X, width=800, height=600)
```

**Parameters:**
- `X`: numpy array or pandas DataFrame with numeric features
- `width`, `height`: Figure dimensions in pixels (default: 1600x1600)  
- `fig_type`: Display type ('notebook' for Jupyter, None for default)

**When to use:** 
- Identify multicollinearity in features
- Detect feature relationships before modeling
- Visual correlation matrix exploration

---

### 2. Time Series Diagnostics (`diagnostic.py`)

#### **`seasonal_decompose`**
Performs MSTL (Multiple Seasonal-Trend decomposition using Loess) with trend significance testing and residual analysis.

```python
from tinyshift.plot import seasonal_decompose
import pandas as pd

# Multiple seasonality decomposition
seasonal_decompose(
    X=time_series,
    periods=[7, 365],  # Weekly and yearly patterns
    nlags=10,
    width=1300,
    height=1200
)
```

**Parameters:**
- `X`: Time series data (numpy array, list, or pandas Series)
- `periods`: Single period (int) or multiple periods (list) for seasonal components
- `nlags`: Number of lags for Ljung-Box residual test (default: 10)
- `width`, `height`: Figure dimensions (default: 1300x1200)

**Output Components:**
- **Trend**: Long-term directional movement
- **Seasonal**: Regular periodic patterns 
- **Residuals**: Remaining unexplained variation
- **Statistics Panel**: Trend significance (R²/p-value) and Ljung-Box test

**When to use:**
- Decompose complex time series with multiple seasonalities
- Validate seasonal patterns in demand forecasting
- Assess model residuals for autocorrelation

---

#### **`stationarity_analysis`**
Comprehensive stationarity testing using Augmented Dickey-Fuller test with visual rolling statistics.

```python
from tinyshift.plot import stationarity_analysis

stationarity_analysis(
    X=time_series,
    window=30,  # Rolling window size
    width=1200,
    height=800
)
```

**Parameters:**
- `X`: Time series data
- `window`: Rolling window size for statistics (default: 30)
- `width`, `height`: Figure dimensions (default: 1200x800)

**Output:**
- Original time series plot
- Rolling mean and standard deviation
- ADF test results (statistic, p-value, critical values)

**When to use:**
- Test stationarity assumptions before ARIMA modeling
- Identify trend and variance changes over time
- Validate differencing transformations

---

#### **`residual_analysis`**
Comprehensive residual diagnostics for model validation and assumption testing.

```python
from tinyshift.plot import residual_analysis

residual_analysis(
    residuals=model_residuals,
    nlags=20,
    width=1400,
    height=1000
)
```

**Parameters:**
- `residuals`: Model residual values
- `nlags`: Number of lags for autocorrelation analysis (default: 20)
- `width`, `height`: Figure dimensions (default: 1400x1000)

**Output Panels:**
1. **Residuals vs Time**: Temporal patterns and heteroscedasticity
2. **Q-Q Plot**: Normality assessment
3. **ACF/PACF**: Autocorrelation structure
4. **Histogram**: Distribution shape
5. **Statistics**: Ljung-Box test, ARCH test, normality tests

**When to use:**
- Validate regression model assumptions
- Diagnose time series model adequacy
- Identify remaining patterns in residuals

---

#### **`pami` (Permutation Auto Mutual Information)**
Visualizes nonlinear autocorrelation using permutation-based mutual information across multiple lags.

```python
from tinyshift.plot import pami

pami(
    X=time_series,
    max_lag=24,
    width=1000,
    height=600
)
```

**Parameters:**
- `X`: Time series data
- `max_lag`: Maximum lag to compute mutual information (default: 24)
- `width`, `height`: Figure dimensions (default: 1000x600)

**When to use:**
- Detect nonlinear autocorrelation patterns
- Identify optimal lag structure for nonlinear models
- Complement traditional ACF/PACF analysis

---

## Function Comparison Matrix

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`corr_heatmap`** | Correlation visualization | Tabular data | Interactive heatmap | Feature selection, multicollinearity detection |
| **`seasonal_decompose`** | Time series decomposition | Time series | Trend/seasonal/residual components | Seasonal pattern analysis, forecasting prep |
| **`stationarity_analysis`** | Stationarity testing | Time series | ADF test + rolling stats | ARIMA modeling prep, trend detection |
| **`residual_analysis`** | Model diagnostics | Residuals | Multiple diagnostic plots | Model validation, assumption testing |
| **`pami`** | Nonlinear correlation | Time series | Mutual information by lag | Nonlinear dependency detection |

---

## Integration with TinyShift Workflow

### **Data Quality Assessment**
```python
# 1. Correlation analysis for feature engineering
corr_heatmap(X_features)

# 2. Stationarity check before drift detection
stationarity_analysis(target_series)
```

### **Model Validation**
```python
# 3. Residual diagnostics after model fitting
residual_analysis(model.residuals_)

# 4. Seasonal validation for time series models
seasonal_decompose(y_true - y_pred, periods=[7, 30])
```

### **Advanced Pattern Detection**
```python
# 5. Nonlinear autocorrelation analysis
pami(feature_series, max_lag=48)
```