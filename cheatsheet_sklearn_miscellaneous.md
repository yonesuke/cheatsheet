# Scikit-learn Cheatsheet: Miscellaneous Tools

This category contains various utility models and comparisons that don't fit into the main categories.

## Key Techniques

### 1. Isotonic Regression (`IsotonicRegression`)
- Fits a non-decreasing function to data.
- Non-parametric: No assumption about the functional form (e.g., linear vs quadratic).
- Useful for probability calibration and modeling monotonic relationships.

### 2. Anomaly & Outlier Detection Comparison
- Scikit-learn provides several algorithms, each with different strengths:
  - **`IsolationForest`**: Fast, based on tree ensembles. Best for high-dimensional data.
  - **`LocalOutlierFactor` (LOF)**: Based on local density. Good for finding outliers relative to their clusters.
  - **`OneClassSVM`**: Best for novelty detection when the training data contains only "normal" samples.

### 3. Dimensionality Reduction Bounds
- **`johnson_lindenstrauss_bound`**: A theoretical tool to estimate the number of dimensions required to preserve distances between points when using random projections.

### 4. Kernel Ridge Regression (`KernelRidge`)
- Combines Ridge regression (L2) with the kernel trick.
- Similar to SVR but uses squared error loss and has a closed-form solution.

## Code Snippet: Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# 1. Isolation Forest (Global anomaly)
iso = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso.fit_predict(X) # -1 for anomaly, 1 for normal

# 2. Local Outlier Factor (Local anomaly)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = lof.fit_predict(X)
```

## Code Snippet: Isotonic Regression

```python
from sklearn.isotonic import IsotonicRegression
ir = IsotonicRegression(out_of_bounds='clip')
y_fit = ir.fit_transform(x, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
