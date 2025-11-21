# Visualization Module (`plot`)

The `plot` module provides comprehensive visualization tools for exploratory data analysis, correlation analysis, time series diagnostics, and classification model evaluation. Built on Plotly for interactive, publication-ready visualizations that support both statistical analysis and MLOps monitoring workflows.

## Features

### 1. Classification Model Evaluation (`calibration.py`)

#### **Model Calibration & Reliability**

#### **`reliability_curve`**
Generates a reliability curve (calibration curve) for binary classifiers, plotting true probability vs predicted probability.

```python
from tinyshift.plot import reliability_curve

reliability_curve(
    clf=classifier,
    X=X_test,
    y=y_test,
    model_name="RandomForest",
    n_bins=15
)
```

**Parameters:**
- `clf`: Trained classifier with predict_proba method
- `X`: Input feature data for evaluation
- `y`: True binary labels (0 or 1)
- `model_name`: Name to display in legend (default: "Model")
- `n_bins`: Number of bins for the curve (default: 15)
- `fig_type`: Display renderer (default: None)

**When to use:** 
- Assess model calibration quality
- Identify over/under-confident predictions
- Compare calibration across different models

---

#### **`score_distribution`**
Displays histogram of predicted probability scores to understand model confidence patterns.

```python
from tinyshift.plot import score_distribution

score_distribution(
    clf=classifier,
    X=X_test,
    nbins=20
)
```

**Parameters:**
- `clf`: Trained classifier with predict_proba method
- `X`: Input feature data
- `nbins`: Number of histogram bins (default: 15)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Analyze distribution of model confidence
- Identify calibration issues (e.g., overconfidence)
- Understand prediction patterns

---

#### **Classification Performance**

#### **`confusion_matrix`**
Interactive confusion matrix heatmap with percentage annotations for binary classification.

```python
from tinyshift.plot import confusion_matrix

confusion_matrix(
    clf=classifier,
    X=X_test,
    y=y_test,
    percentage_by_class=True
)
```

**Parameters:**
- `clf`: Trained classifier with predict method
- `X`: Input feature data
- `y`: True binary labels
- `fig_type`: Display renderer (default: None)
- `percentage_by_class`: Show percentages by class vs overall (default: True)

**When to use:**
- Evaluate classification performance
- Identify class-specific errors
- Compare FP/FN trade-offs

---

#### **Conformal Prediction**

#### **`efficiency_curve`**
Visualizes efficiency and validity trade-off for conformal prediction classifiers across different error rates.

```python
from tinyshift.plot import efficiency_curve

efficiency_curve(
    clf=conformal_classifier,
    X=X_test,
    width=800,
    height=400
)
```

**Parameters:**
- `clf`: Conformal classifier with predict_set method
- `X`: Input feature data
- `fig_type`: Display renderer (default: None)
- `width`: Figure width in pixels (default: 800)
- `height`: Figure height in pixels (default: 400)

**When to use:**
- Assess conformal predictor calibration
- Optimize efficiency vs validity trade-off
- Validate coverage guarantees

---

#### **Statistical Distributions**

#### **`beta_confidence_analysis`**
Analyzes model confidence for production deployment using Beta distribution visualization to assess model reliability.

```python
from tinyshift.plot import beta_confidence_analysis

# High confidence model (many successes, few failures)
beta_confidence_analysis(
    alpha=95,  # successes/correct predictions
    beta_param=5,  # failures/incorrect predictions
    fig_type=None
)

# Low confidence model (few successes, many failures)
beta_confidence_analysis(alpha=15, beta_param=85)
```

**Parameters:**
- `alpha`: Model successes/correct predictions (must be positive)
- `beta_param`: Model failures/incorrect predictions (must be positive)
- `fig_type`: Display renderer (default: None)

**When to use:**
- Assess model readiness for production deployment
- Evaluate deployment confidence based on success/failure ratio
- Visualize risk assessment for MLOps decision making
- Compare model reliability across different validation periods

---

### 2. Correlation Analysis (`correlation.py`)

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

### 3. Time Series Diagnostics (`diagnostic.py`)

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

### Binary Classification Model Evaluation

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`reliability_curve`** | Model calibration assessment | Classifier + test data | Calibration curve | Evaluate prediction confidence accuracy |
| **`score_distribution`** | Confidence pattern analysis | Classifier + features | Score histogram | Identify over/underconfident predictions |
| **`confusion_matrix`** | Classification performance | Classifier + test data | Interactive heatmap | Analyze class-specific errors |
| **`efficiency_curve`** | Conformal prediction trade-offs | Conformal classifier | Efficiency vs validity | Optimize prediction set performance |
| **`beta_confidence_analysis`** | Production confidence assessment | Alpha/beta parameters | PDF plot | Evaluate model deployment readiness |

### Time Series & Correlation Analysis

| Function | Purpose | Input Type | Key Output | Best Use Case |
|----------|---------|------------|------------|---------------|
| **`corr_heatmap`** | Correlation visualization | Tabular data | Interactive heatmap | Feature selection, multicollinearity detection |
| **`seasonal_decompose`** | Time series decomposition | Time series | Trend/seasonal/residual components | Seasonal pattern analysis, forecasting prep |
| **`stationarity_analysis`** | Stationarity testing | Time series | ADF test + rolling stats | ARIMA modeling prep, trend detection |
| **`residual_analysis`** | Model diagnostics | Residuals | Multiple diagnostic plots | Model validation, assumption testing |
| **`pami`** | Nonlinear correlation | Time series | Mutual information by lag | Nonlinear dependency detection |

---

## Integration with TinyShift Workflow

### **Classification Model Validation**
```python
# 1. Model calibration assessment
reliability_curve(clf, X_test, y_test, model_name="XGBoost")

# 2. Prediction confidence analysis
score_distribution(clf, X_test)

# 3. Performance evaluation
confusion_matrix(clf, X_test, y_test)

# 4. Production deployment confidence
beta_confidence_analysis(alpha=successes, beta_param=failures)
```

### **Conformal Prediction Optimization**
```python
# 4. Efficiency-validity trade-off analysis
efficiency_curve(conformal_clf, X_test)
```

### **Data Quality Assessment**
```python
# 5. Correlation analysis for feature engineering
corr_heatmap(X_features)

# 6. Stationarity check before drift detection
stationarity_analysis(target_series)
```

### **Model Validation**
```python
# 7. Residual diagnostics after model fitting
residual_analysis(model.residuals_)

# 8. Seasonal validation for time series models
seasonal_decompose(y_true - y_pred, periods=[7, 30])
```

### **Advanced Pattern Detection**
```python
# 9. Nonlinear autocorrelation analysis
pami(feature_series, max_lag=48)
```

---

## Summary: Classification Function Quick Reference

### Model Calibration & Performance
| Metric/Function | Input Required | Output | Question You Want to Answer |
|----------------|----------------|--------|----------------------------|
| **`reliability_curve`** | Classifier + X + y | Calibration curve | "Are my model's confidence scores accurate?" |
| **`score_distribution`** | Classifier + X | Score histogram | "How confident is my model in its predictions?" |
| **`confusion_matrix`** | Classifier + X + y | Performance heatmap | "What types of errors is my model making?" |
| **`efficiency_curve`** | Conformal classifier + X | Efficiency vs validity | "How efficient are my prediction sets?" |
| **`beta_confidence_analysis`** | Alpha + beta parameters | PDF visualization | "How confident can I be putting this model in production?" |