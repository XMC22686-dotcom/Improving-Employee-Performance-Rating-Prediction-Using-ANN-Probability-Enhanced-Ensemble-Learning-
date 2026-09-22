# ANN Probability-Enhanced Ensemble Learning for Employee Performance Prediction

This repository contains the Python implementation of the ANN probability-enhanced ensemble learning framework developed for employee performance rating prediction.

The method combines the soft class-probability outputs of multiple artificial neural network (ANN) base learners with the original employee features. The enhanced features are then used by different second-stage classifiers, including AdaBoost, Random Forest, ExtraTrees, HistGradientBoosting, Logistic Regression, KNN, Decision Tree, and Linear SVM.

## Dataset

The experiments use the publicly available **IBM HR Analytics Employee Attrition & Performance** dataset from Kaggle.

Dataset link:  
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

The target variable is `PerformanceRating`.

The following variables are excluded before model training:

- EmployeeNumber
- EmployeeCount
- StandardHours
- Over18
- PercentSalaryHike
- Attrition

## Main Code

`ANNProb_core.py`

The code includes:

- Data preprocessing
- Train/validation/test splitting
- Multiple ANN base learners
- Weighted ANN ensemble
- ANN probability-enhanced feature construction
- Second-stage ensemble classifiers
- Conventional machine learning baselines
- LightGBM, XGBoost, CatBoost, and TabNet baselines
- ANN ensemble-size sensitivity analysis
- Feature importance analysis
- Class imbalance comparison
- Repeated stratified cross-validation
- Friedman and Wilcoxon statistical tests
- Prediction-time evaluation

## Requirements

Main Python packages include:

```text
numpy
pandas
scikit-learn
scipy
torch
imbalanced-learn
lightgbm
xgboost
catboost
pytorch-tabnet
