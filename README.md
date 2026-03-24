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

## Data
Datasets are not included in this repo due to size and licensing.

Download from Kaggle:
- [Ames Housing Dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
- [King County House Sales](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)

After downloading, place files in:
- `Housing_Project/data/raw/AmesHousing.csv`
- `Housing_Project/data/raw/kc_house_data.csv`

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

### Cross-Regional Transfer (Ames → King County)
| Model | RMSE | R² | Notes |
|-------|------|-----|-------|
| XGBoost (transfer) | $492,187 | -0.797 | Best transfer performance |
| Random Forest (transfer) | $510,548 | -0.934 | Slightly worse |
| Linear Regression (transfer) | $170M | -216k | Complete failure |
| XGBoost (native KC training) | $230,890 | 0.647 | Much better with native training |

## Key Findings

### Model Performance
1. Tree models significantly outperform linear models — confirms strong nonlinearity in housing data
2. XGBoost needs tuning but wins on final performance — Random Forest is more stable out of the box
3. Learning curves show Linear Regression is high bias — more data won't help, needs more complexity

### Error Analysis
4. Luxury houses (>$300k) have 4x higher RMSE than low-tier houses — model struggles with rare examples
5. OldTown neighborhood has highest error due to high price variance within neighborhood ($115k std)
6. Model performance degrades significantly in edge cases (very old houses, extreme sizes)

### Cross-Regional Generalization
7. **Models trained on Iowa fail on Seattle** — negative R² indicates worse than predicting mean
8. **Feature mismatch is critical** — only 9/174 features available from King County; 165 filled with defaults
9. **XGBoost transfers best** — complex models handle distribution shift better than linear models
10. **Feature importance shifts across regions** — Seattle prioritizes quality (grade) 2.5x more than Iowa
11. **Native training helps but limited** — RMSE improves from $492k → $231k, but still constrained by feature quality

### Interpretability
12. **SHAP bridges interpretability gap** — XGBoost + SHAP provides feature explanations comparable to linear models
13. **Linear sacrifices 29% accuracy** — the $9,000 RMSE gap from Linear → XGBoost could impact real estate decisions
14. **Context-dependent model selection** — choose Linear for regulatory/legal contexts, XGBoost for production systems

## What Makes This Project Different
- **Cross-regional generalization test** — most student projects test on same dataset; this tests real distribution shift
- **Deep error analysis** — price tier analysis, neighborhood analysis, residual patterns
- **Comprehensive model comparison** — 9 models with learning curves and bias-variance analysis
- **SHAP interpretability** — demonstrates modern ML interpretability tools for model selection decisions
- **Real-world insights** — feature mismatch, geographic context, market dynamics all explored

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn, Category Encoders

## How to Run
1. Clone this repo
2. Upload datasets to Google Drive at `Housing_Project/data/raw/`
3. Run notebooks in order (01 → 08)
4. utils.py is loaded via exec() in each notebook

## Future Work
- Test additional cross-regional pairs (e.g., Ames → California housing)
- Implement domain adaptation techniques to improve transfer performance
- Build ensemble models combining predictions from multiple regions
- Deploy best model as web API for real-time price predictions
