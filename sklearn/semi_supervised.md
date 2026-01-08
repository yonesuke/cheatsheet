# Scikit-learn Cheatsheet: Semi-Supervised Learning

Semi-supervised learning is used when you have a small amount of labeled data and a large amount of unlabeled data.

## What can be done?
- **Label Propagation**: Predict labels for the unlabeled portion of your dataset.
- **Self-Training**: Use a standard supervised model to iteratively label the unlabeled data.
- **Benefit**: Can significantly improve performance over purely supervised learning when labeling is expensive.

## Key Algorithms
1. **`SelfTrainingClassifier`**:
   - A wrapper that can use any classifier that provides `predict_proba`.
   - In each iteration, it adds the most confident predictions on unlabeled data to the training set.
2. **`LabelPropagation`**:
   - Uses a similarity graph between all data points to "spread" labels.
   - Based on the "Label Propagation" algorithm by Zhu and Ghahramani.
3. **`LabelSpreading`**:
   - Similar to LabelPropagation but uses a regularized graph Laplacian. 
   - Generally more robust to noise.

## Input Format
- Labels for unlabeled samples must be marked as `-1`.

## Theoretical Background
- **Manifold Assumption**: Points on the same low-dimensional manifold should have the same label.
- **Cluster Assumption**: Points in the same cluster are likely to have the same class.

## Code Snippet: Self-Training & Label Spreading

```python
import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier, LabelSpreading
from sklearn.svm import SVC

# 1. Mark unlabeled data as -1
y_train_mixed = np.copy(y_train)
random_unlabeled_indices = np.random.rand(len(y_train)) < 0.7
y_train_mixed[random_unlabeled_indices] = -1

# 2. Self-Training Wrapper
# threshold: minimum probability to include a sample in the next iteration
svc = SVC(probability=True, gamma="auto")
self_training_model = SelfTrainingClassifier(svc, threshold=0.75)
self_training_model.fit(X_train, y_train_mixed)

# 3. Label Spreading (Graph-based)
# kernel: 'knn' or 'rbf'
label_spread = LabelSpreading(kernel='knn', n_neighbors=7)
label_spread.fit(X_train, y_train_mixed)
predicted_labels = label_spread.transduction_
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
