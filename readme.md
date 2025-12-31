# House Price Prediction

A machine learning project that predicts house prices using multiple regression models, compares their performance, and provides an interactive **Streamlit web application** for visualization and inference.
---

## 🧠 Models Used

- Linear Regression  
- Ridge Regression  
- Random Forest Regressor  
- Tuned Random Forest Regressor  
- **Tuned XGBoost Regressor**

## 📌 Key Highlights

- Implemented and compared **Linear, Ridge, Random Forest, and XGBoost** models  
- Applied **hyperparameter tuning** for Random Forest and XGBoost  
- Evaluated models using **MAE, MSE, RMSE, and R²**  
- Built an interactive **Streamlit app (`app.py`)**  
- Large trained models handled using **Git LFS**

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

## 🖥️ Streamlit Application

A Streamlit web application has been added using **`app.py`**, allowing users to:

- View predictions  
- Compare model performance  
- Interact with trained models  

### ▶️ Run the app locally

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
House-Price-Prediction/
│
├── app.py                  # Streamlit application
├── trained_models/         # Trained ML models (Git LFS)
├── artifacts/              # Predictions & test data
├── notebooks/              # EDA & model training notebooks
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Git & Git LFS
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

