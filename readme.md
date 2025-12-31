# House Price Prediction

This repository contains trained machine learning models for predicting house prices using various regression algorithms. The models include **Random Forest**, **XGBoost**, and their tuned versions for improved performance.

---

## Project Structure

```
trained_models/
├── rf_model.pkl        # Random Forest model
├── xgb_model.pkl       # XGBoost model
├── best_rf.pkl         # Tuned Random Forest model
└── best_xgb.pkl        # Tuned XGBoost model
```

---

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn folium
```

---

## Usage

Load the trained models and predict house prices:

```python
import joblib

# Load models
rf_model = joblib.load('trained_models/rf_model.pkl')
xgb_model = joblib.load('trained_models/xgb_model.pkl')
best_rf = joblib.load('trained_models/best_rf.pkl')
best_xgb = joblib.load('trained_models/best_xgb.pkl')

# Sample input for prediction
sample_input = [[3, 2, 2000, 10000, 2, 0, 0, 3, 8, 2000, 0, 1990, 0, 98001, 47.5480, -121.9836, 2000, 10000]]

# Predict prices
print('Random Forest Prediction: $', rf_model.predict(sample_input)[0])
print('XGBoost Prediction: $', xgb_model.predict(sample_input)[0])
print('Tuned Random Forest Prediction: $', best_rf.predict(sample_input)[0])
print('Tuned XGBoost Prediction: $', best_xgb.predict(sample_input)[0])
```

---

## Model Evaluation

| Model                   | MAE       | MSE           | RMSE      | R² Score |
|-------------------------|-----------|---------------|-----------|----------|
| Tuned XGBoost Regressor | 67,629.61 | 1.827e+10     | 135,169.38| 0.879    |
| XGBoost Regressor       | 70,577.65 | 2.088e+10     | 144,512.74| 0.862    |
| Random Forest           | 72,756.75 | 2.200e+10     | 148,335.95| 0.854    |
| Tuned Random Forest     | 74,077.71 | 2.207e+10     | 148,567.43| 0.854    |
| Linear Regression       |104,244.46 | 3.042e+10     | 174,401.38| 0.799    |
| Ridge Regression        |127,487.65 | 4.507e+10     | 212,290.64| 0.702    |
 
**Note:** 
- RMSE (lower is better)		
- MAE (lower is better)
- R² (closer to 1 is better)
  
**Observation:** Tuned XGBoost performs best with the lowest errors and highest R².

---

## Key Features

- Predict house prices based on 18 features including bedrooms, bathrooms, sqft, grade, house age, and location.
- Hyperparameter tuning for Random Forest and XGBoost to optimize performance.
- Visualizations for actual vs predicted prices, feature correlations, and error metrics.

---

## Conclusion

- **Tuned XGBoost**: Most reliable with highest accuracy.
- **Random Forest & Tuned RF**: Strong performance, slightly below XGBoost.
- **Linear & Ridge Regression**: Higher error due to non-linear relationships in data.

---

## References

- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)

