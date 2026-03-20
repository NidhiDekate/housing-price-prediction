# Cross-Regional Housing Price Prediction & Generalization Study

Predicting house prices using machine learning with a focus on deep error analysis 
and cross-regional generalization testing.

## Project Overview
Most housing price projects stop at model accuracy. This project goes deeper —
analyzing WHERE models fail, WHY they fail, and whether a model trained on 
Iowa housing data can generalize to Seattle housing data.

## Datasets
- **Primary (Train/Test):** Ames Housing Dataset — 2930 rows, 82 features
- **Generalization Test:** King County Seattle — 21,613 rows, 21 features

## Project Structure
```
Housing_Project/
├── utils.py                    # Shared utility functions
├── 01_EDA.ipynb               # Exploratory Data Analysis + Hypotheses
├── 02_Preprocessing.ipynb     # Missing values, encoding, feature engineering
├── 03_Feature_Selection.ipynb # Lasso + SHAP feature selection study
├── 04_Modeling.ipynb          # 9 models compared + hyperparameter tuning
├── 05_Learning_Curves.ipynb   # Bias-variance analysis
├── 06_Error_Analysis.ipynb    # Deep error analysis by price tier + neighborhood
├── 07_Generalization.ipynb    # Ames model tested on King County (coming soon)
└── 08_SHAP.ipynb              # SHAP interpretability analysis (coming soon)
```

## Models Compared
Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, 
Random Forest, Gradient Boosting, XGBoost, SVR

## Key Results
| Model | RMSE | R² |
|-------|------|-----|
| XGBoost (tuned) | $22,108 | 0.931 |
| Gradient Boosting (tuned) | $22,962 | — |
| Random Forest | $24,255 | 0.926 |
| Linear Regression | $31,268 | 0.878 |
| SVR | $94,323 | -0.109 |

## Key Findings
1. Tree models significantly outperform linear models — confirms strong nonlinearity in housing data
2. Luxury houses (>$300k) have 4x higher RMSE than low-tier houses — model struggles with rare examples
3. OldTown neighborhood has highest error due to high price variance within neighborhood ($115k std)
4. XGBoost needs tuning but wins on final performance — Random Forest is more stable out of the box
5. Learning curves show Linear Regression is high bias — more data won't help, needs more complexity

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn

## How to Run
1. Clone this repo
2. Upload datasets to Google Drive at `Housing_Project/data/raw/`
3. Run notebooks in order (01 → 08)
4. utils.py is loaded via exec() in each notebook
