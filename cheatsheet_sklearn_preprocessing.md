# Scikit-learn Cheatsheet: Preprocessing

Preprocessing is the transformation of raw data into a format that is more suitable for machine learning algorithms.

## What can be done?
- **Scaling**: Adjust the range or distribution of numeric features.
- **Encoding**: Convert categorical strings into numeric values.
- **Transformation**: Handle skewed data or non-linear relationships.
- **Discretization**: Bin continuous variables into intervals.

## Key Tools
1. **Scalers**:
   - `StandardScaler`: $z = (x - \mu) / \sigma$. Standardizes to mean 0, variance 1.
   - `MinMaxScaler`: Scales to range [0, 1]. Sensitive to outliers.
   - `RobustScaler`: Uses median and interquartile range (IQR). Best for outlier-heavy data.
2. **Encoders**:
   - `OneHotEncoder`: Binary column per category.
   - `OrdinalEncoder`: Maps categories to integers [0, 1, 2...].
   - `TargetEncoder`: Encode categories based on the target mean (useful for high-cardinality).
3. **Power Transforms**:
   - `PowerTransformer`: Box-Cox or Yeo-Johnson transforms to make data more Gaussian-like.
   - `QuantileTransformer`: Maps data to a uniform or normal distribution.
4. **Generating Features**:
   - `PolynomialFeatures`: Creates $x^2, xy, y^2$. Captures interactions between features.

## Tips
- **Fit on Train, Transform on Test**: Never fit a scaler on the whole dataset to avoid data leakage.
- **Standardization**: Required for distance-based models (KNN, SVM, K-Means) and Gradient Descent.

## Code Snippet: Common Preprocessing

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer

# 1. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 2. Interaction Terms
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# 3. Target Encoding (for categorical features)
from sklearn.preprocessing import TargetEncoder
encoder = TargetEncoder(smooth='auto')
X_encoded = encoder.fit_transform(X_cat, y)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
