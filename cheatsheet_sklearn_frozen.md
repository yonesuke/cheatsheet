# Scikit-learn Cheatsheet: Frozen Estimators

A `FrozenEstimator` is a wrapper that prevents a fitted estimator from being updated during a subsequent `fit` call.

## What can be done?
- **Transfer Learning**: Use a pre-trained model as a fixed feature extractor in a larger pipeline.
- **Multi-stage Training**: Fit one part of a model, freeze it, and then fit a second part that depends on the first.
- **Resource Optimization**: Avoid re-fitting expensive models when only a small part of the pipeline changes.

## Key Concept
- The `FrozenEstimator` delegates `predict`, `transform`, and other methods to the underlying estimator but its `fit` method does nothing (effectively "no-op").

## Code Snippet: Freezing an Estimator

```python
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Fit an estimator normally
clf = LogisticRegression().fit(X_train, y_train)

# 2. Freeze it
frozen_clf = FrozenEstimator(clf)

# 3. Use in a new pipeline
# When pipe.fit() is called, frozen_clf.fit() does nothing.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', frozen_clf)
])
pipe.fit(X_new, y_new) 
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
