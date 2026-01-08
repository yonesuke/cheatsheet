# Scikit-learn Cheatsheet: Inspection (Interpretability)

The `inspection` module provides tools to understand how a model makes decisions and which features are most important.

## What can be done?
- **Feature Importance**: Quantify how much each feature contributes to the prediction.
- **Partial Dependence**: Visualize the relationship between a target and a feature, averaging out others.
- **Decision Boundaries**: Plot how a classifier separates classes in a 2D plane.

## Key Tools
1. **`permutation_importance`**:
   - Measures importance by randomly shuffling a single feature and seeing how much the score drops.
   - **Advantage**: Model-agnostic and doesn't overemphasize high-cardinality features (unlike Random Forest's impurity-based importance).
2. **`PartialDependenceDisplay`**:
   - Shows the marginal effect one or two features have on the predicted outcome.
   - Helps identify if the relationship is linear, monotonic, or complex.
3. **`DecisionBoundaryDisplay`**:
   - Convenient tool to plot the class decision boundaries in a 2D feature space.

## Comparison: Global vs Local Importance
- **Impurity-based (RF)**: Fast, but biased toward features with many unique values.
- **Permutation-based**: Slower, but more reliable and works on any model.

## Code Snippet: Permutation Importance & PDP

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier().fit(X_train, y_train)

# 1. Permutation Importance
# n_repeats: number of times to shuffle each feature
result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)
# result.importances_mean contains the importance scores

# 2. Partial Dependence Plot (PDP)
# features: index or name of features to plot
display = PartialDependenceDisplay.from_estimator(clf, X, features=[0, 1, (0, 1)])
# (0, 1) plots a 2D interaction contour
```

## Code Snippet: Decision Boundary Display

```python
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# Only works with 2 features (X has shape [n, 2])
disp = DecisionBoundaryDisplay.from_estimator(
    clf, X, response_method="predict", cmap=plt.cm.RdYlBu, alpha=0.8
)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
