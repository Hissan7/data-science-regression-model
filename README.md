![Logo](logos/KCL.png)

# A group of regression models using ensemble methods 

This repository contains a framework consisting of multiple regression methods, which involves training and evaluating regression models on a simulated tabular dataset. The objective is to predict a continuous outcome variable using numerical and categorical features while maximising out-of-sample predictive performance.

The project follows a full, reproducible data science pipeline including exploratory data analysis, preprocessing, model comparison, model selection, evaluation via cross-validation, and generation of final test-set predictions.

## What this framework achieves 

- Task: Supervised regression
- Target variable: Predicting outcomes on the `CW1_test` dataset
- Dataset type: Simulated tabular data

There are many features involved:

- 27 numerical variables
- 3 categorical variables (cut, color, clarity)

The evaluation metric: Out-of-sample R^2

## Exploratory analysis

- Exploratory analysis was conducted in `notebooks/01_eda.ipynb` and includes:

- Dataset inspection (shape, info, describe)
- Data cleaning 
- Distribution analysis of the target variables
- Correlation heatmap of numerical features

Key findings:

- The outcome variable is approximately unimodal (slight skew) and symmetric
- No extreme skewness or heavy outliers observed
- Most features exhibit weak to moderate linear correlations
- Predictive signal appears distributed across many variables
- These observations motivated the use of non-linear ensemble models.

## Preprocessing 

All preprocessing is implemented in `src/preprocessing.py` using `scikit-learn` pipelines to avoid data leakage.
Steps include:

- One-hot encoding of categorical features
- Standardisation of numerical features (zero mean, unit variance)
- Unified preprocessing within cross-validation folds
- This ensures consistent and reproducible transformations across models.
 
## Models Evaluated

Multiple regression model families were trained and compared empirically using 5-fold cross-validation.

#### 1. Linear Models (Baselines)

1. Ordinary Least Squares
2. Ridge Regression
3. Lasso Regression
4. Elastic Net

Performance:
Cross-validated **R^2 ≈ 0.28**

Used mainly as interpretable baselines.

#### 2. Tree-Based Ensembles

1. Random Forest Regressor
2. Extremely Randomised Trees (ExtraTrees)

Performance:
Cross-validated **R^2 ≈ 0.44-0.46**

Captured non-linearities and feature interactions more effectively.

#### 3. Gradient Boosting Models

1. Gradient Boosting Regressor 
2. XGBoost Regressor

Performance:

Gradient Boosting: **R^2 ≈ 0.47**
XGBoost: similar but slightly lower performance

#### Model choice 

Gradient Boosting Regression was selected as the final model due to the highest and most stable cross-validated performance
with a good bias–variance trade-off. Gradient boosting had the most robust behaviour across folds and showed stale consistency with the simulated nature of the dataset

Performance convergence across all the ensemble models suggests the remaining error is largely irreducible, consistent with the dataset’s construction.

## Final model configuration and hyperparameters 

The final model was trained using the following hyperparameters:

1. `n_estimators` = 300
2. `learning_rate` = 0.05
3. `max_depth` = 3
4. `subsample` = 1.0 (standard, non-stochastic Gradient Boosting)
5. `random_state` : fixed for reproducibility