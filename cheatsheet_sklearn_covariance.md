# Scikit-learn Cheatsheet: Covariance Estimation

Covariance estimation is used to understand the relationship between variables and is a core component of many algorithms like LDA and Mahalanobis distance.

## What can be done?
- **Relationship Discovery**: Calculate how variables change together.
- **Outlier Detection**: Use Mahalanobis distance to find samples that deviate from the distribution.
- **Structure Learning**: Find sparse relationships where most variables are conditionally independent (graphical models).

## Key Algorithms
1. **`EmpiricalCovariance`**:
   - Standard Maximum Likelihood Estimator.
   - Unbiased but can be inaccurate (high variance) when the number of features is large relative to samples.
2. **Shrinkage Methods (`LedoitWolf`, `OAS`)**:
   - Mix empirical covariance with a simple target (like identity matrix).
   - `LedoitWolf`: Optimally calculates the shrinkage coefficient.
   - `OAS`: Similar to Ledoit-Wolf but designed for small sample sizes.
3. **`GraphicalLasso`**:
   - Learns a sparse inverse covariance matrix (precision matrix) using L1 penalty.
   - Useful for discovering network structures (which variables depend on which).
4. **`MinCovDet` (Minimum Covariance Determinant)**:
   - Robust estimator that ignores outliers.
   - Highly recommended for outlier detection data cleaning.

## Computational Complexity
- **Empirical**: $O(p^2 \cdot n)$ ( $n$: samples, $p$: features).
- **GraphicalLasso**: $O(p^3)$ or higher depending on convergence.

## Code Snippet: Robust vs Empirical Covariance

```python
import numpy as np
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# Generate data with outliers
X = np.random.randn(100, 2)
X[0] = [10, 10] # Outlier

# 1. Standard Estimation (skewed by outlier)
emp_cov = EmpiricalCovariance().fit(X)
print("Empirical Location:", emp_cov.location_)

# 2. Robust Estimation (ignores outlier)
robust_cov = MinCovDet().fit(X)
print("Robust Location:", robust_cov.location_)

# 3. Get Mahalanobis Distances to detect outliers
distances = robust_cov.mahalanobis(X)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
