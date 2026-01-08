# Scikit-learn Cheatsheet: Feature Selection

Feature selection reduces the number of input variables to reduce overfitting, improve accuracy, and decrease computational cost.

## What can be done?
- **Filter Methods**: Use statistical scores to pick features.
- **Wrapper Methods**: Search for the best subset of features using a model.
- **Embedded Methods**: Let models (like Lasso or Random Forest) select features during training.

## Key Algorithms
1. **`SelectKBest` (Filter)**:
   - Selects high-scoring features based on `f_classif` (ANOVA), `chi2`, or `mutual_info_classif`.
2. **`RFE` (Recursive Feature Elimination - Wrapper)**:
   - Iteratively fits a model and removes the least important features.
   - **`RFECV`**: Automatically finds the optimal number of features using cross-validation.
3. **`SelectFromModel` (Embedded)**:
   - Uses `feature_importances_` or `coef_` attributes of a fitted estimator.
   - Works well with `Lasso` (L1 penalty) or `RandomForest`.
4. **`SequentialFeatureSelector`**:
   - Greedy forward or backward selection. More robust but slower than RFE.

## Theoretical Background
- **Mutual Information**: Captures any kind of statistical dependency (nonlinear), whereas F-test/Chi2 only capture linear relationships.
- **L1 Regularization (Lasso)**: Forces coefficients to be exactly zero, performing automatic selection.

## Computational Complexity
- **Filter**: Very fast.
- **RFE**: Slow, as it fits the model multiple times ($O(p)$ times).

## Code Snippet: Feature Selection Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# 1. Filter: Select top 10 features via ANOVA
selector = SelectKBest(score_func=f_classif, k=10)

# 2. Wrapper: Recursive Feature Elimination
# Requires an estimator that provides feature importance (e.g. RF, Linear)
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=5)

# 3. Embedded: Select from Model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
sfm = SelectFromModel(Lasso(alpha=0.1))

# Integrating into Pipeline
pipe = Pipeline([
    ('feature_selection', selector),
    ('classification', RandomForestClassifier())
])
pipe.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
