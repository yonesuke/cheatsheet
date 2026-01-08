# Scikit-learn Cheatsheet: Decision Trees

Decision Trees are non-parametric supervised learning methods used for classification and regression.

## What can be done?
- **Automatic Interaction Discovery**: Naturally capture non-linear relationships between features.
- **Interpretability**: Easily visualize exactly how a decision is made.
- **Handling Mixed Data**: Can handle both numerical and categorical data (though scikit-learn's current implementation requires encoding).

## Key Algorithms
1. **`DecisionTreeClassifier` / `DecisionTreeRegressor`**:
   - Predictive models that split data into branches by optimizing info gain (Gini/Entropy).
2. **`ExtraTrees`**:
   - Randomly picks splits to further reduce variance (often used in ensembles).

## Important Concepts
- **Overfitting**: Trees can grow very deep and learn the noise in data.
- **Pruning (`ccp_alpha`)**: Cost Complexity Pruning is used to remove branches that provide little predictive power.
- **Feature Importance**: Trees provide a ranking of features based on how much they reduce the impurity.

## Visualization
- **`plot_tree`**: Render the tree structure as a diagram.
- **`export_text`**: Get a text-based summary of rules.

## Code Snippet: Tree Building & Visualization

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt

# 1. Fit Tree
clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
clf.fit(X, y)

# 2. Visualize
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# 3. Export Rules
tree_rules = export_text(clf, feature_names=list(feature_names))
print(tree_rules)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
