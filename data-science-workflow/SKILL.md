---
name: data-science-workflow
description: Use this skill when the user wants dataset analysis, exploratory data analysis, feature engineering, predictive modeling, model comparison, or actionable ML insights from tabular data such as CSV, Excel, parquet, pandas DataFrames, or database extracts. Trigger it for classification, regression, clustering, forecasting, or general machine-learning workflows that need leakage-safe preprocessing, clear Python code, and explained evaluation results.
---

# Data Science Workflow

## Overview

Use this skill for end-to-end data science and machine learning work on datasets. Favor a structured workflow: understand the data, prevent leakage, build a trustworthy baseline, evaluate clearly, and explain what the results mean.

## Quick Intake

Before writing code, summarize:

- the dataset source and shape
- the prediction or analysis goal
- the target column, if any
- the likely problem type: classification, regression, clustering, forecasting, or unsupervised exploration
- key risks such as leakage, class imbalance, small sample size, or temporal ordering

If essential information is missing, state the assumptions you will use.

## Standard Workflow

### 1. Data Understanding and EDA

Always start by inspecting:

- row and column counts
- column names and data types
- missing-value rates
- duplicate rows and obvious data quality issues
- target distribution or label balance when supervised
- summary statistics for numeric features
- category frequencies for important categorical features

Visualize when useful:

- histograms or KDE plots for distributions
- boxplots for outliers
- count plots for class balance
- correlation heatmaps for numeric features
- pairplots or scatterplots for a few high-signal relationships

Call out findings that affect modeling choices, not just raw numbers.

### 2. Preprocessing

Be explicit about leakage prevention:

- split data before fitting imputers, encoders, scalers, or selectors
- use time-aware splits for temporal data
- use stratified splits for imbalanced classification when appropriate
- fit preprocessing only on training folds or training sets
- prefer `Pipeline` and `ColumnTransformer` for reproducible preprocessing

Common preprocessing steps:

- impute missing values
- encode categorical features
- scale numeric features when model-sensitive
- handle target transforms only when justified
- remove identifiers or columns unavailable at prediction time

### 3. Feature Engineering

Create features only from information available at inference time. Prefer interpretable, business-grounded features such as:

- ratios, differences, and interactions
- date parts and elapsed-time features
- grouped aggregates when they do not leak future or target information
- text length or count features for light NLP cases
- binning when it improves robustness or interpretability

Explain why each engineered feature should help.

### 4. Modeling

Start simple, then increase complexity deliberately.

Recommended progression:

1. Establish a naive or simple baseline.
2. Train a strong classical baseline such as linear/logistic models, decision trees, random forests, or gradient boosting.
3. Use XGBoost, LightGBM-style approaches, or neural networks only when the data size and problem structure justify them.

Choose models by task:

- classification: logistic regression, random forest, gradient boosting, XGBoost
- regression: linear regression, random forest regressor, gradient boosting regressor, XGBoost
- high-dimensional or sparse data: regularized linear models often deserve an early baseline
- neural networks: reserve for large datasets, complex feature interactions, images, text, or when simpler models plateau

### 5. Evaluation

Use metrics that match the task and business objective.

- classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC when imbalanced
- regression: RMSE, MAE, R-squared, sometimes MAPE when appropriate
- clustering: silhouette score plus qualitative inspection
- forecasting: time-aware backtesting and horizon-specific error metrics

Prefer cross-validation for model comparison and keep a clean holdout set for final validation when possible.

Explain:

- how models compare
- whether performance is stable across folds
- which errors matter most
- whether the model is likely overfitting or underfitting

### 6. If Performance Is Poor

Do not stop at "the score is low." Diagnose likely causes:

- weak signal in available features
- data quality problems
- leakage in the earlier workflow
- target imbalance
- wrong metric for the business need
- underfitting from overly simple models
- overfitting from overly flexible models

Then suggest concrete next steps such as better features, more data, re-labeling, class weighting, hyperparameter tuning, threshold tuning, or a different validation scheme.

## Output Pattern

Before code, provide a short reasoning summary that covers:

- objective
- task type
- planned preprocessing
- initial model strategy
- evaluation plan

Then write clean, modular Python using standard libraries when available:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `pytorch` or `tensorflow`

Prefer small functions and reproducible pipelines over monolithic notebooks. Add concise comments only where the logic is not obvious.

## Guardrails

- Never let the target leak into feature creation or preprocessing.
- Never fit scalers, imputers, or encoders on the full dataset before the split.
- Treat IDs, timestamps, and post-outcome columns as suspicious until proven safe.
- For temporal data, preserve ordering in validation.
- For imbalanced tasks, report more than accuracy.
- If the dataset is tiny, say so and temper confidence.
- If the user asks for interpretation, include feature importance, coefficients, SHAP-style analysis, or error slices when feasible.

## Example Triggers

This skill should activate for requests like:

- "Analyze this CSV and tell me what predicts churn."
- "Build a baseline classifier for this dataset and improve it."
- "Compare random forest and XGBoost on these features."
- "Do EDA, clean the data, and train a regression model."
- "Why is my model performing badly on this tabular dataset?"
