# Scikit-learn Cheatsheet: Probability Calibration

Probability calibration is the process of adjusting the predicted probabilities of a classifier so they better reflect the actual likelihood of an event.

## What can be done?
- **Confidence Rating**: Transform raw model scores into reliable probability estimates (e.g., if a model says 80% probability, it should be correct 80% of the time).
- **Model Comparison**: Use Calibration Curves (Reliability Diagrams) to see which models are overconfident or underconfident.
- **Improved Decision Making**: Essential for risk-sensitive applications like medical diagnosis or financial forecasting.

## Calibration Methods
1. **Platt Scaling (`method='sigmoid'`)**:
   - Fits a logistic regression model to the classifier's outputs.
   - Best for Support Vector Machines (SVM) and when training data is small.
2. **Isotonic Regression (`method='isotonic'`)**:
   - Fits a non-parametric, non-decreasing function.
   - More powerful than Platt scaling but requires more data (~1000+ samples) and is prone to overfitting.

## Theoretical Background
- **Perfect Calibration**: A model's predicted probability equals the observed frequency of the positive class.
- **Brier Score**: A common metric to evaluate calibration. It measures the mean squared difference between predicted probability and actual outcome.
- **Cross-Validation**: Calibration should be performed on data not used for training the base estimator to avoid optimistic biases. `CalibratedClassifierCV` handles this automatically.

## Computational Complexity
- **Training**: Adds the cost of fitting a 1D regressor (Sigmoid or Isotonic).
- **Memory**: Minimal, as it only stores parameters for the mapping function.
- **Latency**: Negligible impact on prediction speed.

## Pros & Cons
### Pros
- **Interpretability**: Makes class probabilities meaningful.
- **Trust**: Allows users to know when the model is "unsure".
### Cons
- **Doesn't improve Accuracy**: Calibration adjusts probabilities but generally doesn't change the classification boundary or ROC-AUC.
- **Data Hungry**: Isotonic regression needs a decent amount of validation data.
- **Complexity**: Adds another step to the training pipeline.

## Code Snippet

```python
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

X, y = load_your_data()
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 1. Base Estimator (e.g., SVM which is notoriously uncalibrated)
base_clf = SVC(kernel='linear', C=1.0)

# 2. Wrap with Calibration
# method='sigmoid' (Platt) or 'isotonic'
calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
calibrated_clf.fit(X_train, y_train)

# 3. Predict Probabilities
probs = calibrated_clf.predict_proba(X_test)[:, 1]

# 4. Evaluation: Calibration Curve
prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
```

---
**Credits**: This cheatsheet is based on the [scikit-learn](https://scikit-learn.org/) documentation and examples, which are licensed under the [BSD 3-Clause License](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING).
Copyright (c) 2007 - 2026 The scikit-learn developers. All rights reserved.
