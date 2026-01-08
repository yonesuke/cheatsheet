# Scikit-learn Cheatsheet: Ensemble Methods

Ensemble methods combine the predictions of several base estimators to improve generalizability and robustness over a single estimator.

## What can be done?
- **Reduce Overfitting**: Averaging techniques (Bagging).
- **Increase Accuracy**: Iterative techniques (Boosting).
- **Combine Heterogeneous Models**: Voting and Stacking.
- **Quantile Regression**: Use Gradient Boosting to predict intervals.

## Key Algorithms
1. **Bagging (Averaging)**:
   - `RandomForestClassifier/Regressor`: Builds many deep trees on subsets of data/features and averages results.
2. **Boosting (Sequential)**:
   - `GradientBoostingClassifier/Regressor`: Fits new models to the residuals of previous models.
   - `HistGradientBoosting`: Modern, extremely fast version similar to LightGBM.
   - `AdaBoost`: Focuses more on samples that previous models misclassified.
3. **Voting**:
   - `VotingClassifier`: Combines different models via majority vote (hard) or average probability (soft).
4. **Stacking**:
   - `StackingClassifier`: Trains a "final estimator" (meta-learner) to combine predictions of base learners.

## Theoretical Background
- **Bias-Variance Trade-off**: Bagging reduces variance (good for complex models like deep trees). Boosting reduces bias (good for weak models like shallow trees).
- **Out-of-Bag (OOB) Score**: Validation method for Random Forest using samples not seen by specific trees.

## Computational Complexity
- **Random Forest**: Parallelizable. $O(M \cdot n \cdot \log n)$ where $M$ is number of trees.
- **Gradient Boosting**: Sequential (harder to parallelize, except Hist).

## Code Snippet: Random Forest & HistGradientBoosting

```python
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# 1. Random Forest (Great baseline)
rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1)
# n_jobs=-1 uses all CPU cores

# 2. HistGradientBoosting (Fast for large datasets)
# Categorical support is built-in!
hgb = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1)

# 3. Voting Ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC(probability=True)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], 
    voting='soft'
)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
