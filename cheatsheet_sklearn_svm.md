# Scikit-learn Cheatsheet: Support Vector Machines (SVM)

Support Vector Machines are powerful supervised learning models used for classification, regression, and outlier detection.

## What can be done?
- **High-Dimensional Classification**: Extremely effective in spaces where the number of features is greater than the number of samples.
- **Non-linear Separation**: Use the "kernel trick" to handle complex decision boundaries.
- **Robust Regression**: Predict continuous values while only being sensitive to errors within a "tube" (SVR).
- **Novelty Detection**: Identify if new data comes from the same distribution as training data (OneClassSVM).

## Key Algorithms
1. **`SVC` / `SVR`**:
   - The main models for classification and regression.
   - Use `kernel='rbf'` (default), `'poly'`, or `'linear'`.
2. **`LinearSVC` / `LinearSVR`**:
   - Faster than `SVC(kernel='linear')` for large datasets.
   - Does not support the kernel trick.
3. **`OneClassSVM`**:
   - Unsupervised outlier detection.

## Key Hyperparameters
- **`C`**: Regularization parameter. 
  - Large `C`: Smaller margin, fits training data better (risk of overfitting).
  - Small `C`: Larger margin, simplifies the decision boundary (risk of underfitting).
- **`gamma`**: Defines how far the influence of a single training example reaches.
  - Large `gamma`: Local influence (complex boundaries).
  - Small `gamma`: Global influence (simpler boundaries).

## Computational Complexity
- $O(n_f \cdot n_s^2)$ to $O(n_f \cdot n_s^3)$ for most solvers. Not suitable for datasets with $> 100,000$ samples.

## Code Snippet: SVC with RBF Kernel

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# 1. Tuning C and Gamma
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

# 2. Linear SVC for large-scale data
from sklearn.svm import LinearSVC
l_svc = LinearSVC(C=1.0, max_iter=10000)
l_svc.fit(X_train, y_train)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
