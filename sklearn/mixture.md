# Scikit-learn Cheatsheet: Mixture Models

Mixture models represent data as being generated from a mixture of several component distributions (usually Gaussian).

## What can be done?
- **Soft Clustering**: Get probabilities of a sample belonging to each cluster.
- **Density Estimation**: Model complex probability distributions as sums of Gaussians.
- **Outlier Detection**: Samples in very low-density regions of the mixture are outliers.

## Key Algorithms
1. **`GaussianMixture` (GMM)**:
   - Uses the Expectation-Maximization (EM) algorithm to fit a set of Gaussians.
   - You must specify the number of components $K$.
2. **`BayesianGaussianMixture`**:
   - A variant that integrates over parameters using Variational Inference.
   - Can automatically "zero out" unnecessary components, helping discover the true number of clusters.

## Covariance Types
GMM allows different constraints on the covariance matrices:
- `'full'`: Each component has its own general covariance (most flexible).
- `'tied'`: All components share the same general covariance.
- `'diag'`: Each component has its own diagonal covariance.
- `'spherical'`: Each component has its own single variance (simplest).

## Theoretical Background
- **EM Algorithm**: 
  - **E-step**: Estimate the probability (responsibility) of each sample for each Gaussian.
  - **M-step**: Update Gaussian parameters to maximize likelihood based on responsibilities.
- **BIC/AIC**: Criteria used to select the optimal number of components.

## Computational Complexity
- $O(k \cdot n \cdot p^2)$ where $n$: samples, $p$: features, $k$: components.

## Code Snippet: GMM Clustering

```python
from sklearn.mixture import GaussianMixture
import numpy as np

# 1. Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# 2. Predict labels (Hard clustering)
labels = gmm.predict(X)

# 3. Predict probabilities (Soft clustering)
probs = gmm.predict_proba(X)

# 4. Find optimal components via BIC
bic_scores = []
n_components_range = range(1, 10)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n).fit(X)
    bic_scores.append(gmm.bic(X))
# Best n is the one where BIC is lowest
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
