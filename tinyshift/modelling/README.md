# ML Modeling Utilities Module (`modelling`)

The `modelling` module provides sklearn-compatible preprocessing and feature engineering tools designed for robust machine learning workflows. Includes multicollinearity detection, feature residualization, and advanced scaling techniques optimized for real-world data challenges.

## Features

### 1. Multicollinearity Detection (`multicollinearity.py`)

#### **`filter_features_by_vif`** - Variance Inflation Factor Feature Selection
Iteratively removes features with high VIF values to reduce multicollinearity using parallel computation for efficiency.

```python
from tinyshift.modelling import filter_features_by_vif
import numpy as np

# Generate correlated features
X = np.random.randn(1000, 10)
X[:, 5] = X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1  # Highly correlated

# Filter features by VIF
feature_mask = filter_features_by_vif(
    X, 
    threshold=5.0,     # VIF threshold
    verbose=True,      # Show progress
    n_jobs=-1         # Use all CPU cores
)

# Apply filtering
X_filtered = X[:, feature_mask]
print(f"Kept {feature_mask.sum()} out of {len(feature_mask)} features")
```

**Parameters:**
- `X`: Feature matrix (numpy array or pandas DataFrame)
- `threshold`: VIF threshold for feature removal (default: 5.0)
  - 1: No correlation
  - 1-5: Moderate correlation  
  - 5-10: High correlation
  - >10: Very high correlation (recommend removal)
- `verbose`: Print progress information
- `n_jobs`: CPU cores for parallel computation (-1 = all cores)

**Returns:**
- Boolean mask indicating which features to keep

**When to use:**
- Before linear regression, logistic regression
- When dealing with highly correlated feature sets
- Feature selection for interpretable models
- Preprocessing step to improve model stability

---

### 2. Feature Residualization (`residualizer.py`)

#### **`FeatureResidualizer`** - Linear Dependency Removal via Residualization
Sklearn-compatible transformer that reduces multicollinearity by replacing correlated features with their residuals from linear models.

```python
from tinyshift.modelling import FeatureResidualizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Initialize residualizer
residualizer = FeatureResidualizer()

# Fit and transform
X_residualized = residualizer.fit_transform(
    X, 
    corrcoef=0.8,      # Correlation threshold
    corr_type="abs"    # Consider absolute correlations
)

# Pipeline integration
pipeline = Pipeline([
    ('residualizer', FeatureResidualizer()),
    ('regressor', LinearRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**Parameters:**
- `corrcoef`: Absolute correlation threshold (default: 0.8)
- `corr_type`: 
  - `"abs"`: Consider absolute correlations (default)
  - `"pos"`: Only positive correlations

**Key Features:**
- **Preserves Information**: Removes linear relationships while keeping unique variance
- **Order Independence**: Processes features by total correlation strength
- **Prevents Cycles**: Avoids circular dependencies in residualization
- **Pipeline Compatible**: Full sklearn transformer interface

**Algorithm:**
1. Compute correlation matrix between all features
2. Rank features by total correlation with others
3. For each highly correlated feature:
   - Fit linear model using correlated predictors
   - Replace feature with residuals (observed - predicted)
   - Mark as residualized to prevent future use as predictor

**When to use:**
- Alternative to VIF filtering when you want to keep all features
- Preprocessing for linear models with multicollinearity
- When domain knowledge suggests all features are important
- Interpretable feature engineering for correlated predictors

---

### 3. Robust Scaling (`scaler.py`)

#### **`RobustGaussianScaler`** - Three-Step Robust Normalization
Advanced scaler combining winsorization, power transformation, and standardization for robust handling of outliers and non-Gaussian distributions.

```python
from tinyshift.modelling import RobustGaussianScaler
import numpy as np

# Data with outliers and skewness
X = np.exp(np.random.randn(1000, 5))  # Log-normal distribution
X[0, :] = 1000  # Add outliers

# Robust scaling
scaler = RobustGaussianScaler()
X_scaled = scaler.fit_transform(
    X,
    winsorize_method="mad",     # Median Absolute Deviation bounds
    power_method="yeo-johnson"  # Power transformation
)

# Check transformation results
print("Original skewness:", np.mean([scipy.stats.skew(X[:, i]) for i in range(X.shape[1])]))
print("Scaled skewness:", np.mean([scipy.stats.skew(X_scaled[:, i]) for i in range(X_scaled.shape[1])]))

# Inspect winsorization bounds
print("Winsorization bounds:", scaler.winsorization_bounds_)
```

**Three-Step Process:**

1. **Winsorization (Outlier Clipping)**
   - Clips extreme values based on statistical intervals
   - Methods: `"stddev"`, `"mad"`, `"iqr"`, etc.
   - Reduces impact of outliers while preserving data structure

2. **Power Transformation (Distribution Normalization)**
   - `"yeo-johnson"`: Works with positive and negative values
   - `"box-cox"`: Requires strictly positive data
   - Transforms data toward Gaussian distribution

3. **Standard Scaling (Final Normalization)**
   - Zero mean, unit variance
   - Applied after power transformation for optimal results

**Parameters:**
- `winsorize_method`: Statistical method for outlier bounds
  - `"stddev"`: Mean ± k×standard deviations
  - `"mad"`: Median ± k×MAD (robust to outliers)
  - `"iqr"`: Interquartile range based
- `power_method`: Transformation type
  - `"yeo-johnson"`: Handles positive/negative values (default)
  - `"box-cox"`: Positive values only, often more effective

**Attributes:**
- `winsorization_bounds_`: List of (lower, upper) clipping bounds per feature
- `power_transformer_`: Fitted PowerTransformer object
- `scaler_`: Fitted StandardScaler object

**When to use:**
- Data with outliers and non-Gaussian distributions
- Before algorithms sensitive to scale (SVM, neural networks, PCA)
- When simple StandardScaler or RobustScaler insufficient

---

## Tool Comparison Matrix

| Tool | Purpose | Input Requirement | Output | Computational Cost | Best Use Case |
|------|---------|------------------|--------|-------------------|---------------|
| **`filter_features_by_vif`** | Multicollinearity removal | Numeric features | Feature mask | O(p³×k) iterations | Linear models, interpretability |
| **`FeatureResidualizer`** | Linear dependency removal | Numeric features | Residualized features | O(p²×n) | Keep all features, remove correlations |
| **`RobustGaussianScaler`** | Robust normalization | Numeric features | Scaled features | O(n×p) | Outlier-robust scaling |

**Legend:** p = features, n = samples, k = iterations
---

## Statistical Foundations

### **VIF Calculation**
For feature *i*, VIF is calculated as:
```
VIF_i = 1 / (1 - R²_i)
```
where R²_i is from regressing feature *i* on all other features.

### **Residualization Process**
For correlated feature *i* with predictors *j₁, j₂, ..., jₖ*:
```
ŷᵢ = β₀ + β₁xⱼ₁ + β₂xⱼ₂ + ... + βₖxⱼₖ
residual_i = yᵢ - ŷᵢ
```

### **Power Transformation**
- **Yeo-Johnson**: Handles positive and negative values
- **Box-Cox**: λ parameter estimated via MLE for optimal normality

---

## Performance Considerations

| Operation | Time Complexity | Memory Usage | Parallelization |
|-----------|----------------|--------------|-----------------|
| **VIF Filtering** | O(p³×k) | O(p²) | ✅ Feature-level |
| **Residualization** | O(p²×n) | O(p²) | ❌ Sequential |
| **Robust Scaling** | O(n×p) | O(p) | ❌ Feature-wise |

**Recommendations:**
- Use `n_jobs=-1` for VIF filtering on high-dimensional data
- Consider feature pre-filtering before residualization for p > 1000
- Robust scaling is efficient even for large datasets

---
