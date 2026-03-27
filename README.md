# HDB Resale Price Prediction
**Predictive Analytics for WOW! Real Estate Agency**

*Emma Poh & 3 others | GA Data Analytics Immersive*

## Overview
A machine learning system that predicts Singapore HDB resale flat prices using structural, locational, and temporal features — enabling WOW! Real Estate Agency to provide data-driven pricing recommendations for buyers and sellers.

## Key Results
- **Best Model:** LightGBM (tuned via RandomizedSearchCV)
- **R² Score:** ~0.88 on held-out validation set
- **MAE:** ~SGD 40,000 (mean absolute error)
- **MAPE:** ~8% (mean absolute percentage error)
- **Features Used:** 40+ engineered features from 78 raw columns
- **Training Data:** 150,634 HDB transactions (2012–2021)

## Project Structure
```
hdb-resale-prediction/
├── HDB_Resale_Price_Prediction.ipynb  # Main analysis notebook
├── streamlit_app.py                    # Interactive price calculator
├── requirements.txt                    # Python dependencies
├── data_dictionary.md                  # Column documentation
├── .gitignore
├── data/
│   ├── train.csv                       # Training data (150K rows)
│   ├── test.csv                        # Test data (16K rows)
│   └── sample_sub_reg.csv              # Kaggle submission format
├── model/
│   └── hdb_model.pkl                   # Trained model (after notebook run)
├── exports/
│   └── fig_*.png                       # Generated visualisations
└── submission/
    └── submission.csv                  # Kaggle predictions
```

## Methodology
1. **Data Quality Audit** — Missing value analysis, amenity proximity imputation, duplicate detection
2. **Exploratory Data Analysis** — 11 visualisations covering price distributions, location premiums, structural drivers, temporal trends, and amenity effects
3. **Feature Engineering** — 8 domain-informed features including remaining lease, mature estate flag, MRT accessibility score, and amenity density composites
4. **Feature Selection** — Redundancy check (|r| > 0.90) and cumulative importance curve to identify key predictors
5. **Preprocessing** — StandardScaler + OneHotEncoder via ColumnTransformer pipeline
6. **Model Comparison** — Ridge, Random Forest, LightGBM, XGBoost benchmarked against mean baseline
7. **Hyperparameter Tuning** — RandomizedSearchCV (20 iterations, 5-fold CV) on LightGBM, with sensitivity analysis on top hyperparameter combinations
8. **SHAP Interpretation** — Feature importance, beeswarm plots, dependence analysis
9. **Validation** — Time-series expanding window cross-validation, learning curve analysis, residual diagnostics, error analysis by price quartile, and calibration plot

## Top Price Drivers (SHAP)
1. **Floor area** — Dominant structural predictor
2. **Remaining lease** — Depreciation effect (newer = premium)
3. **Storey level** — Vertical premium (~SGD 2,500/floor)
4. **Town/location** — Up to SGD 120K premium (Central Area vs Sembawang)
5. **MRT proximity** — Transit accessibility drives ~15% price variation

## Online Calculator
An interactive Streamlit app lets users estimate HDB resale prices by inputting property characteristics.

**Try it live:** [HDB Resale Price Calculator](https://blank-app-in22ljofwh.streamlit.app/)

**Or run locally:**
```bash
streamlit run streamlit_app.py
```

## Tech Stack
Python 3.10+ | pandas | NumPy | scikit-learn | LightGBM | XGBoost | SHAP | Matplotlib | Seaborn | Streamlit

## Setup
```bash
pip install -r requirements.txt
jupyter notebook HDB_Resale_Price_Prediction.ipynb
```

## Author
**Emma Poh & 3 others** — GA Data Analytics Immersive
- Email: emmalynpoh@gmail.com
- LinkedIn: [linkedin.com/in/emmalynpoh](https://www.linkedin.com/in/emmalynpoh)
- GitHub: [github.com/emmabiee](https://github.com/emmabiee)
