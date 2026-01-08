# Scikit-learn Cheatsheet: Classification

Classification is a supervised learning task where the goal is to predict the categorical class of new observations.

## What can be done?
- **Binary & Multiclass Classification**: Predict between two or more discrete categories.
- **Decision Boundary Analysis**: Understand how models partition the feature space.
- **Dimensionality Reduction for Class Separation**: Use techniques like LDA to maximize class separability.

## Key Algorithms
1. **`LinearDiscriminantAnalysis` (LDA)**:
   - Finds a linear combination of features that separates classes.
   - Assumes classes follow a Gaussian distribution with shared covariance.
2. **`QuadraticDiscriminantAnalysis` (QDA)**:
   - Like LDA but allows each class to have its own covariance matrix.
   - Results in quadratic decision boundaries.
3. **Common Others** (Detailed in their own sections):
   - `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GaussianNB`.

## Theoretical Background
- **Bayes' Rule**: Many classifiers (LDA, QDA, Naive Bayes) are based on modeling the class conditional density $P(X|y)$.
- **Discriminative vs Generative**: 
  - Generative (LDA, QDA, NB): Models $P(X|y)$ and $P(y)$.
  - Discriminative (Logistic Regression, SVM): Models $P(y|X)$ directly.

## Computational Complexity
- **LDA**: $O(n \cdot p^2 + p^3)$ where $n=$ samples, $p=$ features.
- **Prediction**: Usually very fast ($O(p)$ for linear models).

## Evaluation Metrics
- **Accuracy**: $(TP + TN) / Total$.
- **Precision/Recall/F1**: For imbalanced classes.
- **ROC-AUC**: Ability to distinguish between classes across thresholds.
- **Confusion Matrix**: Detailed look at where types of errors occur.

## Code Snippet: Classifier Comparison

```python
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report

X, y = load_your_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)

# QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

print("LDA Report:\n", classification_report(y_test, y_pred_lda))
print("QDA Report:\n", classification_report(y_test, y_pred_qda))
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
