# Scikit-learn Cheatsheet: Impute (Handling Missing Data)

The `impute` module provides strategies to handle missing values (NaN) in datasets.

## What can be done?
- **Univariate Imputation**: Fill missing values using a single feature's statistics.
- **Multivariate Imputation**: Predict missing values using all other available features (more accurate).
- **Nearest Neighbor Imputation**: Fill based on similar samples.
- **Marking Missingness**: Add a binary indicator feature for where data was missing.

## Key Algorithms
1. **`SimpleImputer`**:
   - `strategy='mean'`, `'median'`, `'most_frequent'`, or `'constant'`.
   - Fast and simple baseline.
2. **`IterativeImputer`**:
   - Models each feature with missing values as a function of others in a round-robin fashion.
   - Inspired by R's MICE (Multivariate Imputation by Chained Equations).
3. **`KNNImputer`**:
   - Finds $K$ nearest neighbors for each sample with a missing value and averages their values for that feature.
4. **`MissingIndicator`**:
   - Useful when the *fact* that a value is missing is informative.

## Tips
- Always `fit` on training data and `transform` on test data.
- If using `IterativeImputer`, you must enable it first as it is experimental: `from sklearn.experimental import enable_iterative_imputer`.

## Code Snippet: Advanced Imputation Pipeline

```python
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer

X = [[1, 2], [np.nan, 3], [7, 6], [4, np.nan]]

# 1. Simple Mean Imputation
imp_mean = SimpleImputer(strategy='mean')
X_simple = imp_mean.fit_transform(X)

# 2. KNN Imputation (Weights by distance)
imp_knn = KNNImputer(n_neighbors=2, weights="distance")
X_knn = imp_knn.fit_transform(X)

# 3. Iterative Imputation
imp_iter = IterativeImputer(max_iter=10, random_state=0)
X_iter = imp_iter.fit_transform(X)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
