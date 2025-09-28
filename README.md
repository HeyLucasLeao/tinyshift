# TinyShift

**TinyShift** is a small experimental Python library designed to detect **data drifts** and **performance drops** in machine learning models over time. The main goal of the project is to provide quick and tiny monitoring tools to help identify when data or model performance unexpectedly change.
For more robust solutions, I highly recommend [Nannyml.](https://github.com/NannyML/nannyml)

## Technologies Used

- **Python 3.x**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **Plotly**
- **Scipy**

## Installation

To install **TinyShift** in your development environment, use **pip**:


```bash
pip install tinyshift
```
If you prefer to clone the repository and install manually:
```bash
git clone https://github.com/HeyLucasLeao/tinyshift.git
cd tinyshift    
pip install .
```

> **Note:** If you want to enable plotting capabilities, you need to install the extras using UV:

```bash
uv install --all-extras
```

## Usage
Below are basic examples of how to use TinyShift's features.
### 1. Data Drift Detection
To detect data drift, simply score in a new dataset to compare with the reference data. The DataDriftDetector will calculate metrics to identify significant differences.

```python
from tinyshift.detector import CategoricalDriftDetector

df = pd.DataFrame("examples.csv")
df_reference = df[(df["datetime"] < '2024-07-01')].copy()
df_analysis = df[(df["datetime"] >= '2024-07-01')].copy()

detector = CategoricalDriftTracker(df_reference, 'discrete_1', "datetime", "W", drift_limit='mad')

analysis_score = detector.score(df_analysis, "discrete_1", "datetime")

print(analysis_score)
```

### 2. Performance Tracker
To track model performance over time, use the PerformanceMonitor, which will compare model accuracy on both old and new data.
```python
from tinyshift.tracker import PerformanceTracker

df_reference = pd.read_csv('refence.csv')
df_analysis = pd.read_csv('analysis.csv')
model = load_model('model.pkl') 
df_analysis['prediction'] = model.predict(df_analysis["feature_0"])

tracker = PerformanceTracker(df_reference, 'target', 'prediction', 'datetime', "W")

analysis_score = tracker.score(df_analysis, 'target', 'prediction', 'datetime')

print(analysis_score)
```

### 3. Visualization
TinyShift also provides graphs to visualize the magnitude of drift and performance changes over time.
```python
tracker.plot.scatter(analysis_score, fig_type="png")

tracker.plot.bar(analysis_score, fig_type="png")
```

### 4. Outlier Detection
To detect outliers in your dataset, you can use the models provided by TinyShift. Currently, it offers the Histogram-Based Outlier Score (HBOS), Simple Probabilistic Anomaly Detector (SPAD), and SPAD+.

```python
from tinyshift.outlier import SPAD

df = pd.read_csv('data.csv')

spad_plus = SPAD(plus=True)
spad_plus.fit(df)

anomaly_scores = spad_plus.decision_function(df)
anomaly_pred = spad_plus.predict(df)

print(anomaly_scores)
print(anomaly_pred)
```
### 5. Anomaly Tracker
The Anomaly Tracker in TinyShift allows you to identify potential outliers based on the drift limit and anomaly scores generated during training. By setting a drift limit, the tracker can flag data points that exceed this threshold as possible outliers.

```python
from tinyshift.tracker import AnomalyTracker

model = load_model('model.pkl') 

tracker = AnomalyTracker(model, drift_limit='mad')

df_analysis = pd.read_csv('analysis.csv')

outliers = tracker.score(df_analysis)

print(outliers)
```
In this example, the `AnomalyTracker` is initialized with a reference model and a specified drift limit. The `score` method evaluates the analysis dataset, calculating anomaly scores and flagging data points that exceed the drift limit as potential outliers.

## Project Structure
The basic structure of the project is as follows:
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

### License
This project is licensed under the MIT License - see the LICENSE file for more details.
