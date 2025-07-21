# CO₂ Factor Forecasting

This repository contains a **machine learning pipeline** for forecasting the **CO₂ emission factor** with 1-hour resolution for the next 7 days, using renewable energy production and weather-related features.

---

## Project Overview

The goal is to design, train, and evaluate machine learning models that predict the CO₂ emission factor using **historical data** and **key energy/weather features**.  
The project compares several models, selects the best one, and provides forecasting capability.

---

## Dataset

**Target variable:** `CO2_factor`

### Main Features:
- `solar_volume`
- `onshore_wind_volume`
- `offshore_wind_volume`
- `electricity_volume`
- `biomass_volume`
- `radiation`
- `sunshine`
- `wind_speed`
- `precipitation`
- `temp`

### Time-based Features:
- `hour_sin`, `hour_cos`
- `month`
- `day_of_week`

---

## Models Used

The following models were tested and compared:

- **Linear Models:** OLS, Ridge, Lasso
- **Tree-Based Models:** Random Forest, XGBoost (tuned and untuned), LightGBM
- **Other Models:** KNN Regressor, MLP Neural Network

---

## Best Performing Model

**Tuned XGBoost** achieved the best performance:

- **RMSE (train):** 0.0124  
- **MAE (train):** 0.0092  
- **R² (train):** 0.9830  
- **MAE (test):** 0.0127  
- **R² (test):** 0.9784  
- **CV RMSE (test):** 0.0214  

---

## Key Steps

### **Data Preprocessing:**
- Feature engineering (time-based features, cyclic encoding for hours)
- Correlation and multicollinearity analysis (VIF)
- Stepwise regression for feature selection

### **Model Training:**
- Train-test split (80%-20%)
- Cross-validation for robust evaluation
- Hyperparameter tuning (GridSearchCV for XGBoost)

### **Model Evaluation:**
- Metrics: RMSE, MAE, R² (train and test sets)
- Feature importance (XGBoost)
- Visualizations: Actual vs Predicted plots, residuals analysis, QQ-plots

### **Forecasting:**
The **final tuned XGBoost model** is saved as `xgboost_tuned_model.pkl` for future predictions.

---

## Repository Contents
- `python_code.ipynb` – Main notebook containing the full analysis and modeling pipeline.
- `xgboost_tuned_model.pkl` – Serialized trained model for forecasting.

