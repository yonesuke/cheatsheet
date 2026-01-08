# Scikit-learn Cheatsheet: Linear Models

Linear models assume that the target value is expected to be a linear combination of the input variables.

## What can be done?
- **Regression**: Predict continuous values (Price, Temperature).
- **Classification**: Predict categories (Spam/Not Spam).
- **Regularization**: Avoid overfitting by penalizing large coefficients.
- **Robust Regression**: Handle outliers in the data.

## Key Algorithms
1. **Regression**:
   - `LinearRegression`: Ordinary Least Squares (no penalty).
   - `Ridge`: L2 penalty (prevents large coefficients, handles multicollinearity).
   - `Lasso`: L1 penalty (forces coefficients to zero, performs feature selection).
   - `ElasticNet`: Combination of L1 and L2.
2. **Classification**:
   - `LogisticRegression`: Despite name, it's for classification.
   - `SGDClassifier`: Linear model optimized via Stochastic Gradient Descent (excellent for large data).
3. **Robust Regression**:
   - `RANSACRegressor`: Fits model by ignoring outliers.
   - `HuberRegressor`: Less sensitive to outliers than OLS.
4. **Bayesian**:
   - `BayesianRidge`: Ridge with automated hyperparameter estimation.

## Theoretical Background
- **Loss Functions**: OLS uses Mean Squared Error. Logistic Regression uses Log-Loss.
- **Regularization**: 
  - $L_1$: $\alpha \cdot ||w||_1$ (Sparsity)
  - $L_2$: $0.5 \cdot \alpha \cdot ||w||_2^2$ (Small weights)

## Computational Complexity
- **OLS**: $O(p^2 n + p^3)$ for $n$ samples, $p$ features.
- **SGD**: $O(k \cdot n \cdot p)$ ( $k$: iterations). Scale linearly with samples.

## Code Snippet: Lasso and Ridge

```python
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

# 1. Regression with Regularization
# alpha: Strength of penalty (higher = more regularization)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

# 2. Classification with L1 (Sparse features)
# solver='liblinear' or 'saga' required for L1
log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

# 3. Large Scale learning
from sklearn.linear_model import SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
