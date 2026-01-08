# Scikit-learn Cheatsheet: Model Selection & Evaluation

The `model_selection` module provides tools to split data, tune hyperparameters, and evaluate model performance.

## What can be done?
- **Robust Evaluation**: Use Cross-Validation to ensure the model generalizes.
- **Hyperparameter Tuning**: Find the best settings for your model (e.g., $C$ in SVM, $k$ in KNN).
- **Diagnostics**: Analyze underfitting vs. overfitting using learning curves.
- **Threshold Tuning**: Select the optimal decision threshold for classification.

## Key Tools
1. **Cross-Validation Splitters**:
   - `KFold`: Standard split.
   - `StratifiedKFold`: Preserves class proportions (essential for classification).
   - `TimeSeriesSplit`: Respects temporal order.
2. **Hyperparameter Search**:
   - `GridSearchCV`: Exhaustive search over specified parameter values.
   - `RandomizedSearchCV`: Samples from distributions (faster, often as good as grid search).
   - `HalvingGridSearch`: Efficient search using "successive halving" (early stopping for poor params).
3. **Visualization Displays**:
   - `LearningCurveDisplay`: Plots score vs. training set size.
   - `ValidationCurveDisplay`: Plots score vs. single hyperparameter.
   - `RocCurveDisplay`, `PrecisionRecallDisplay`.

## Best Practices
- **Nested Cross-Validation**: Use to get an unbiased estimate of the performance when performing hyperparameter tuning.
- **Scoring**: Specify `scoring='roc_auc'` or `scoring='f1'` to optimize for metrics other than accuracy.

## Code Snippet: Grid Search & Cross-Validation

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

# 1. Setup Parameter Grid
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 2. Setup Search
cv = StratifiedKFold(n_splits=5)
# refit=True: Fits the best model on the whole training set
grid = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy', refit=True)

# 3. Execution
grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
best_model = grid.best_estimator_
```

## Code Snippet: Learning Curve

```python
from sklearn.model_selection import LearningCurveDisplay
import matplotlib.pyplot as plt

display = LearningCurveDisplay.from_estimator(
    estimator, X, y, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
)
display.plot()
plt.show()
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
