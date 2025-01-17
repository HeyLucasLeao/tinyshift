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

> **Note:** If you want to enable plotting capabilities, you need to install the extras using Poetry:

```bash
poetry install --all-extras
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

detector = CategoricalDriftDetector(df_reference, 'discrete_1', "datetime", "W", drift_limit='mad')

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

print(anomaly_scores)
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
└── tinyshift
    ├── examples
    │   ├── outlier.ipynb
    │   └── tracker.ipynb
    ├── outlier
    │   ├── __init__.py
    │   ├── base.py
    │   ├── hbos.py
    │   └── spad.py
    ├── plot
    │   ├── __init__.py
    │   └── plot.py
    ├── tests
    │   ├── test_hbos.py
    │   └── test_spad.py
    └── tracker
        ├── anomaly.py
        ├── base.py
        ├── categorical.py
        ├── continuous.py
        └── performance.py      
```

### License
This project is licensed under the MIT License - see the LICENSE file for more details.
