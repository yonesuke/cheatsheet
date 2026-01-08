# Scikit-learn Overview & Index

Scikit-learn is a premier Python library for machine learning, built on top of NumPy, SciPy, and matplotlib. It provides simple and efficient tools for predictive data analysis.

## Core Capabilities

Scikit-learn is organized into several key areas:

### 1. Supervised Learning
- **[Classification](sklearn/classification.md)**: Identifying which category an object belongs to (e.g., SVM, Random Forest, Logistic Regression).
- **[Regression](sklearn/linear_model.md)**: Predicting a continuous-valued attribute (e.g., Ridge, Lasso). See also **[Tree](sklearn/tree.md)** and **[SVM](sklearn/svm.md)** for regressor variants.
- **[Ensemble Methods](sklearn/ensemble.md)**: Combining the predictions of several base estimators (e.g., Boosting, Bagging).

### 2. Unsupervised Learning
- **[Clustering](sklearn/cluster.md)**: Automatic grouping of similar objects into sets (e.g., K-Means, Spectral Clustering).
- **[Decomposition](sklearn/decomposition.md)**: Reducing the number of random variables to consider (e.g., PCA, ICA).
- **[Covariance Estimation](sklearn/covariance.md)**: Estimating the magnitude of relationship between features.

### 3. Model Building & Selection
- **[Preprocessing](sklearn/preprocessing.md)**: Feature extraction and normalization.
- **[Model Selection](sklearn/model_selection.md)**: Comparing, validating and choosing parameters and models (e.g., Grid Search, Cross Validation).
- **[Pipeline/Compose](sklearn/compose.md)**: Chaining estimators and transformers.

## Typical Workflow (The `fit`/`predict` Pattern)

Almost all objects in Scikit-learn share a uniform interface:
1.  **Estimators**: `model.fit(X_train, y_train)`
2.  **Predictors**: `model.predict(X_test)`
3.  **Transformers**: `transformer.transform(X)` or `transformer.fit_transform(X)`

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Initialize
model = RandomForestClassifier()

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)

# 4. Evaluate
print(accuracy_score(y_test, predictions))
```

## Detailed Cheatsheets

For deeper dives into specific modules, see the following:

- **Data Handling**: [Datasets](sklearn/datasets.md), [Preprocessing](sklearn/preprocessing.md), [Impute](sklearn/impute.md)
- **Supervised**: [Linear Models](sklearn/linear_model.md), [SVM](sklearn/svm.md), [Tree](sklearn/tree.md), [Neighbors](sklearn/neighbors.md), [Neural Networks](sklearn/neural_networks.md)
- **Unsupervised**: [Manifold](sklearn/manifold.md), [Mixture Models](sklearn/mixture.md), [Biclustering](sklearn/bicluster.md)
- **Advanced**: [Feature Selection](sklearn/feature_selection.md), [Inspection](sklearn/inspection.md), [Kernel Approximation](sklearn/kernel_approximation.md)
- **Meta-Estimators**: [Multiclass](sklearn/multiclass.md), [Multioutput](sklearn/multioutput.md), [Calibration](sklearn/calibration.md)
- **Developer Guide**: [Developing Estimators](sklearn/developing_estimators.md)

---
*Maintained in the `sklearn/` directory.*
