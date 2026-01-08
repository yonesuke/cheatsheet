# Scikit-learn Cheatsheet: Developing Estimators

Scikit-learn allows you to create custom estimators that work seamlessly with `Pipeline`, `GridSearchCV`, and other utilities.

## What can be done?
- **Custom Transformers**: Implement specialized feature engineering.
- **Custom Models**: Wrap external libraries or unique algorithms into scikit-learn interface.
- **Enforcement**: Ensure your code follows the "Scikit-Learn API contract".

## Key Components
1. **Base Classes**:
   - `BaseEstimator`: Provides `get_params` and `set_params`.
   - `TransformerMixin`: Provides `fit_transform`.
   - `ClassifierMixin`: Provides `score` (accuracy) and sets `_estimator_type`.
   - `RegressorMixin`: Provides `score` (R2).
2. **Validation Utilities**:
   - `check_X_y`: Ensures data format and target are consistent.
   - `check_array`: Standardizes input array (handling NaN, types, etc.).
   - `check_is_fitted`: Raises an error if the model hasn't been fit yet.

## The Contract
- **`__init__`**: Must NOT have side effects. Should only assign arguments to attributes. No logic!
- **`fit(X, y)`**: Must return `self`.
- **`transform(X)`** or **`predict(X)`**: Perform the actual work after fitting.

## Code Snippet: Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

class MyLogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_constant=1.0):
        # Store parameters (no logic here!)
        self.add_constant = add_constant

    def fit(self, X, y=None):
        # Validate data
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        # Return self
        return self

    def transform(self, X):
        # Ensure it was fit
        check_is_fitted(self)
        X = check_array(X)
        # Apply transformation
        return np.log(X + self.add_constant)

# Validation tests
from sklearn.utils.estimator_checks import check_estimator
# check_estimator(MyLogTransformer()) # Runs many automated tests
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
