# Model Evaluation and Hyperparameter Tuning

## Overview

This task evaluates classification and regression models using a loan-default dataset. It covers classification metrics, ROC curves, AUC, threshold selection, hyperparameter tuning, regression evaluation, and final test-set evaluation.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

## Objectives

1. Evaluate classifiers using accuracy, precision, recall, F1-score, and confusion matrices.
2. Plot ROC curves and calculate AUC for three classifiers.
3. Select a classification threshold that achieves at least 75% recall.
4. Tune Random Forest using GridSearchCV.
5. Tune XGBoost using RandomizedSearchCV with 15 iterations.
6. Compare regression MAE, RMSE, and R² against a mean baseline.
7. Evaluate the selected classification model on the held-out test dataset.

## Validation Results

### Classification Metrics

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.800 | 0.6923 | 0.3600 | 0.4737 |
| Decision Tree | 0.840 | 0.6957 | 0.6400 | 0.6667 |
| Random Forest | 0.855 | 0.8000 | 0.5600 | 0.6588 |
| XGBoost | 0.860 | 0.7391 | 0.6800 | 0.7083 |

### ROC-AUC Results

| Model | AUC |
|---|---:|
| Logistic Regression | 0.786 |
| Random Forest | 0.921 |
| XGBoost | 0.916 |

Random Forest achieved the highest AUC of 0.921.

### Threshold Selection

The selected XGBoost threshold was 0.38.

- Precision: 75%
- Recall: 84%

This threshold met the requirement of at least 75% recall on the validation set.

### Hyperparameter Tuning

**Random Forest — GridSearchCV**
- Best cross-validation F1-score: 0.6992
- Validation F1-score: 0.6588
- Best parameters: `max_depth=None`, `min_samples_split=5`, `n_estimators=200`

**XGBoost — RandomizedSearchCV**
- Number of parameter combinations sampled: 15
- Best cross-validation F1-score: 0.7806
- Validation F1-score: 0.8000
- Best parameters: `subsample=0.7`, `n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `colsample_bytree=1.0`

## Regression Results

| Metric | Linear Regression | Mean Baseline |
|---|---:|---:|
| MAE | 2930.6047 | 3410.7209 |
| RMSE | 3859.8346 | 4415.0272 |
| R² | 0.2334 | -0.0030 |

Linear Regression performed better than the mean baseline on all three metrics.

## Final Test-Set Evaluation

The tuned XGBoost classifier achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | 0.89 |
| Precision | 0.85 |
| Recall | 0.68 |
| F1-score | 0.7556 |

### Confusion Matrix

| Actual / Predicted | No Default | Default |
|---|---:|---:|
| No Default | 144 | 6 |
| Default | 16 | 34 |

The model correctly classified 144 non-defaulters and 34 defaulters. It missed 16 actual defaulters.

## How to Run

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Ensure `loan_default.csv` is in the same folder as `task4.py`, then run:

```bash
python task4.py
```

## Files Generated

- `task4_roc_curves.png` — ROC curves for the selected classifiers.
- `task4_classification_results.csv` — Classification comparison table.
- `task4_regression_results.csv` — Regression and mean-baseline comparison.

## Conclusion

The tuned XGBoost classifier achieved 89% accuracy and an F1-score of 0.7556 on the held-out test set. Random Forest achieved the highest validation AUC, while tuned XGBoost achieved the highest validation F1-score among the two tuned models. Linear Regression outperformed the mean baseline for loan-amount prediction.

The dataset is illustrative and should not be treated as a real-world credit-risk assessment.
