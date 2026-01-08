# Scikit-learn Cheatsheet: Clustering

Clustering is an unsupervised learning task that groups a set of objects such that objects in the same group (cluster) are more similar to each other than to those in other groups.

## What can be done?
- **Data Partitioning**: Group data into $K$ clusters (e.g., K-Means).
- **Density Discovery**: Find arbitrarily shaped clusters based on density (e.g., DBSCAN).
- **Hierarchy Building**: Create a tree of clusters (Agglomerative).
- **Anomalies Identification**: Points that don't fit into any cluster (noise in DBSCAN/OPTICS).

## Key Algorithms
1. **`KMeans`**:
   - Simplest and most common. Minimizes within-cluster sum-of-squares.
   - **MiniBatchKMeans**: Faster version for large datasets.
2. **`DBSCAN` / `HDBSCAN`**:
   - Density-based. Can find non-spherical clusters and marked outliers.
3. **`AgglomerativeClustering`**:
   - Hierarchical. Can incorporate "connectivity constraints" to group only adjacent points.
4. **`MeanShift`**:
   - Centroid-based. Finds peaks in a distribution; chooses number of clusters automatically.
5. **`AffinityPropagation`**:
   - Based on message passing between data points.

## Evaluation Metrics (Internal)
- **Silhouette Coefficient**: Measures how similar a point is to its own cluster compared to others. Higher (near 1) is better.
- **Calinski-Harabasz Index**: Ratio of between-cluster variance to within-cluster variance.
- **Davies-Bouldin Index**: Average "similarity" between clusters. Lower is better.

## Computational Complexity
- **K-Means**: $O(T \cdot K \cdot n \cdot p)$ (T: iterations, K: clusters, n: samples, p: features).
- **DBSCAN**: $O(n \cdot \log n)$ with spatial index or $O(n^2)$ without.
- **Agglomerative**: $O(n^2 \cdot \log n)$ to $O(n^3)$.

## Code Snippet: Clustering & Evaluation

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Scaling is CRITICAL for distance-based clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. K-Means
kmeans = KMeans(n_clusters=3, n_init='auto', random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)
print("K-Means Silhouette:", silhouette_score(X_scaled, labels_kmeans))

# 2. DBSCAN
# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other.
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
