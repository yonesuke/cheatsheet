# Scikit-learn Cheatsheet: Multioutput

Multioutput (also known as multitarget or multiresponse) regression and classification involves predicting multiple target variables for each sample.

## What can be done?
- **Predict Multiple Values**: Simultaneously predict several continuous variables (regression) or categorical variables (classification).
- **Model Target Dependencies**: Use chains to let predictions for one target inform the prediction for the next.
- **Multilabel Classification**: Assign multiple labels to a single sample.

## Key Strategies
1. **`MultiOutputRegressor` / `MultiOutputClassifier`**:
   - Independently fits one regressor/classifier per target.
   - Simple baseline that assumes targets are independent.
2. **`RegressorChain` / `ClassifierChain`**:
   - Arranges estimators in a chain.
   - Target $i$ is predicted using the input features plus the predictions of targets $0, \dots, i-1$.
   - Captures correlations between targets.

## Data Format
- Target $y$ should be a 2D array of shape `(n_samples, n_targets)`.

## Code Snippet: Multioutput Regression & Chains

```python
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Multi-target data (X: 100x10, Y: 100x3)
X = np.random.randn(100, 10)
Y = np.random.randn(100, 3)

# 1. Independent Multioutput
mor = MultiOutputRegressor(RandomForestRegressor())
mor.fit(X, Y)

# 2. Regressor Chain (captures target dependencies)
chain = RegressorChain(RandomForestRegressor())
chain.fit(X, Y)
Y_pred = chain.predict(X_new)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
