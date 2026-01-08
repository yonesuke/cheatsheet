# Scikit-learn Cheatsheet: Nearest Neighbors

Nearest Neighbors algorithms find a predefined number of training samples closest in distance to a new point.

## What can be done?
- **Supervised Learning**: Classification and Regression based on consensus of neighbors.
- **Unsupervised Learning**: Find neighbors for manifold learning or data analysis.
- **Density Estimation**: Estimate the probability distribution of data (KDE).
- **Anomaly Detection**: Find points that have low local density.

## Key Algorithms
1. **`KNeighborsClassifier/Regressor`**:
   - Standard KNN. Majority vote for classification, average for regression.
2. **`RadiusNeighborsClassifier`**:
   - Based on points within a fixed distance $r$. Good for data with varied density.
3. **`NearestNeighbors`**:
   - Unsupervised learner. Used for retrieving neighbors (`kneighbors()`) or graph building (`kneighbors_graph()`).
4. **`KernelDensity` (KDE)**:
   - Represents the data as a sum of kernels. Provides a smooth continuous density estimate.
5. **`LocalOutlierFactor` (LOF)**:
   - Compares local density of a point to its neighbors. High LOF score = Outlier.
6. **`NeighborhoodComponentsAnalysis` (NCA)**:
   - Learn a linear transformation to improve KNN accuracy.

## Computational Complexity
- **Brute Force**: $O[D \cdot N^2]$ (Too slow for large data).
- **KD-Tree / Ball-Tree**: $O[D \cdot N \log N]$ (Faster for low dimensions).
- **Inference**: Can be expensive for large $N$ since data must be kept.

## Code Snippet: KNN and KDE

```python
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
import numpy as np

# 1. KNN Classification
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski', p=2)
knn.fit(X_train, y_train)

# 2. Unsupervised Neighbor Search
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

# 3. Kernel Density Estimation
# bandwidth: The "width" of the kernels. Crucial parameter to tune!
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
log_density = kde.score_samples(X_plot)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
