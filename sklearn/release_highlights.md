# Scikit-learn Cheatsheet: Release Highlights (Modern Features)

Scikit-learn evolves rapidly. This cheatsheet highlights key features introduced in recent major versions (0.22 to 1.7+).

## Key Recent Milestones

### 1. Histogram-based GBDT (v0.21+)
- **`HistGradientBoostingClassifier/Regressor`**: Extremely fast gradient boosting, inspired by LightGBM. Supports native categorical features and missing values.

### 2. Interaction & Constraint Support
- **Monotonic Constraints**: Force a model (GBDT) to be always increasing or decreasing with respect to a feature.
- **Interaction Constraints**: Control which features are allowed to interact in tree splits.

### 3. Advanced Encoders & Transformers
- **`TargetEncoder` (v1.3+)**: Efficiently handles high-cardinality categorical features by using the target variable's mean. It includes smoothing to prevent overfitting.
- **`SplineTransformer` (v1.0+)**: For non-linear feature engineering, providing a better alternative to polynomial features in many cases.

### 4. Plotting & Displays API
- Unified API for plotting common charts: `RocCurveDisplay`, `ConfusionMatrixDisplay`, `PrecisionRecallDisplay`, `CalibrationDisplay`, `DecisionBoundaryDisplay`.

### 5. Native Categorical Support
- `HistGradientBoosting` can now handle categorical data directly without one-hot encoding if `categorical_features` parameter is set.

## Code Snippet: Modern Features

```python
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import TargetEncoder

# 1. Native Categorical GBDT
hgb = HistGradientBoostingClassifier(categorical_features=[0, 3, 5])
hgb.fit(X, y)

# 2. Easy Confusion Matrix Plotting
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(hgb, X_test, y_test)

# 3. Target Encoding
te = TargetEncoder()
X_encoded = te.fit_transform(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
