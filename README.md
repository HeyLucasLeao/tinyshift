# TinyShift

**TinyShift** is a lightweight, sklearn-compatible Python library designed for **data drift detection**, **outlier identification**, and **MLOps monitoring** in production machine learning systems. The library provides modular, easy-to-use tools for detecting when data distributions or model performance change over time, with comprehensive visualization capabilities.

For enterprise-grade solutions, consider [Nannyml](https://github.com/NannyML/nannyml).

## Features

- **Data Drift Detection**: Categorical and continuous data drift monitoring with multiple distance metrics
- **Outlier Detection**: **HBOS**, **PCA-based** and **SPAD** outlier detection algorithms  
- **Time Series Analysis**: Seasonality decomposition, trend analysis, and forecasting diagnostics

## Technologies Used

- **Python 3.10+** 
- **Scikit-learn 1.3.0+**
- **Pandas 2.3.0+** 
- **NumPy**
- **SciPy**
- **Statsmodels 0.14.5+**
- **Plotly 5.22.0+** (optional, for plotting)

## 📦 Installation

Install TinyShift using pip:

```bash
pip install tinyshift
```

### Development Installation

Clone and install from source:

```bash
git clone https://github.com/HeyLucasLeao/tinyshift.git
cd tinyshift
pip install -e .
```

## 📖 Quick Start

### 1. Categorical Data Drift Detection

TinyShift provides sklearn-compatible drift detectors that follow the familiar `fit()` and `score()` pattern:

```python
import pandas as pd
from tinyshift.drift import CatDrift

# Load your data
df = pd.read_csv("data.csv")
reference_data = df[df["date"] < '2024-07-01']
analysis_data = df[df["date"] >= '2024-07-01'] 

# Initialize and fit the drift detector
detector = CatDrift(
    freq="D",                    # Daily frequency
    func="chebyshev",           # Distance metric
    drift_limit="auto",         # Automatic threshold detection
    method="expanding"          # Comparison method
)

# Fit on reference data
detector.fit(reference_data)

# Score new data for drift
drift_scores = detector.predict(analysis_data)
print(drift_scores)
```

Available distance metrics for **categorical** data:
- `"chebyshev"`: Maximum absolute difference between distributions
- `"jensenshannon"`: Jensen-Shannon divergence  
- `"psi"`: Population Stability Index

### 2. Continuous Data Drift Detection

For numerical features, use the continuous drift detector:

```python
from tinyshift.drift import ConDrift

# Initialize continuous drift detector
detector = ConDrift(
    freq="W",                   # Weekly frequency  
    func="ws",                  # Wasserstein distance
    drift_limit="auto",
    method="expanding"
)

# Fit and score
detector.fit(reference_data)
drift_scores = detector.score(analysis_data)
```

### 3. Outlier Detection

TinyShift includes sklearn-compatible outlier detection algorithms:

```python
from tinyshift.outlier import SPAD, HBOS, PCAReconstructionError

# SPAD (Simple Probabilistic Anomaly Detector)
spad = SPAD(plus=True)
spad.fit(X_train)

outlier_scores = spad.decision_function(X_test)
outlier_labels = spad.predict(X_test)

# HBOS (Histogram-Based Outlier Score)
hbos = HBOS(dynamic_bins=True)
hbos.fit(X_train, nbins="fd")
scores = hbos.decision_function(X_test)

# PCA-based outlier detection
pca_detector = PCAReconstructionError()
pca_detector.fit(X_train)
pca_scores = pca_detector.decision_function(X_test)
```
### 4. Time Series Analysis and Diagnostics

TinyShift provides time series analysis capabilities:

```python
from tinyshift.plot import seasonal_decompose
from tinyshift.series import trend_significance, permutation_auto_mutual_information

# Seasonal decomposition with multiple periods
seasonal_decompose(
    time_series, 
    periods=[7, 365],  # Weekly and yearly patterns
    width=1200, 
    height=800
)

# Test for significant trends
trend_result = trend_significance(time_series, alpha=0.05)
print(f"Significant trend: {trend_result}")

# Stationary Analysis
fig = stationarity_analysis(time_series)
```

### 5. Advanced Modeling Tools

```python
from tinyshift.modelling import filter_features_by_vif
from tinyshift.stats import bootstrap_bca_interval

# Detect multicollinearity
mask = filter_features_by_vif(X, trehshold=5, verbose=True)
X.columns[mask]

# Bootstrap confidence intervals
confidence_interval = bootstrap_bca_interval(
    data, 
    statistic=np.mean, 
    alpha=0.05, 
    n_bootstrap=1000
)
```

## 📁 Project Structure

```
tinyshift/
├── association_mining/          # Market basket analysis tools
│   ├── analyzer.py             # Transaction pattern analysis
│   └── encoder.py              # Data encoder
├── drift/                      # Data drift detection 
│   ├── base.py                 # Base drift detection classes  
│   ├── categorical.py          # CatDrift for categorical features
│   └── continuous.py           # ConDrift for numerical features
├── examples/                   # Jupyter notebook examples
│   ├── drift.ipynb            # Drift detection examples
│   ├── outlier.ipynb          # Outlier detection demos
│   ├── series.ipynb           # Time series analysis
│   └── transaction_analyzer.ipynb
├── modelling/                  # ML modeling utilities
│   ├── multicollinearity.py   # VIF-based multicollinearity detection
│   ├── residualizer.py        # Residualizer Feature
│   └── scaler.py              # Custom scaling transformations
├── outlier/                    # Outlier detection algorithms
│   ├── base.py                 # Base outlier detection classes
│   ├── hbos.py                 # Histogram-Based Outlier Score
│   ├── pca.py                  # PCA-based outlier detection  
│   └── spad.py                 # Simple Probabilistic Anomaly Detector
├── plot/                       # Visualization capabilities  
│   ├── correlation.py          # Correlation analysis plots
│   └── diagnostic.py           # Time series diagnostics plots
├── series/                     # Time series analysis tools
│   ├── forecastability.py     # Forecast quality metrics
│   ├── outlier.py             # Time series outlier detection
│   └── stats.py               # Statistical analysis functions
└── stats/                      # Statistical utilities
    ├── bootstrap_bca.py        # Bootstrap confidence intervals
    ├── statistical_interval.py # Statistical interval estimation
    └── utils.py               # General statistical utilities
```

```
tinyshift
├── LICENSE
├── README.md
├── poetry.lock
├── pyproject.toml
├── tinyshift
│   ├── association_mining
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── encoder.py
│   ├── examples
│   │   ├── outlier.ipynb
│   │   ├── tracker.ipynb
│   │   └── transaction_analyzer.ipynb
│   ├── modelling
│   │   ├── __init__.py
│   │   ├── multicollinearity.py
│   │   ├── residualizer.py
│   │   └── scaler.py
│   ├── outlier
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hbos.py
│   │   ├── pca.py
│   │   └── spad.py
│   ├── plot
│   │   ├── __init__.py
│   │   ├── correlation.py
│   │   └── plot.py
│   ├── series
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── forecastability.py
│   │   ├── outlier.py
│   │   └── stats.py
│   ├── stats
│   │   ├── __init__.py
│   │   ├── bootstrap_bca.py
│   │   ├── series.py
│   │   ├── statistical_interval.py
│   │   └── utils.py
│   ├── tests
│   │   ├── test.pca.py
│   │   ├── test_hbos.py
│   │   └── test_spad.py
│   └── tracker
│       ├── __init__.py
│       ├── anomaly.py
│       ├── base.py
│       ├── categorical.py
│       ├── continuous.py
│       └── performance.py
```


### Development Setup

```bash
git clone https://github.com/HeyLucasLeao/tinyshift.git
cd tinyshift
pip install -e ".[all]"
```

## 📋 Requirements

- **Python**: 3.10+
- **Core Dependencies**: 
  - pandas (>2.3.0)
  - scikit-learn (>1.3.0) 
  - statsmodels (>=0.14.5)
- **Optional Dependencies**:
  - plotly (>5.22.0) - for visualization
  - kaleido (<=0.2.1) - for static plot export
  - nbformat (>=5.10.4) - for notebook support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [Nannyml](https://github.com/NannyML/nannyml)

