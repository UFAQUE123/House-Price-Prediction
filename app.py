#-------------------------------House Price Prediction App-------------------------------------------#
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import joblib

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="House Price Prediction App",
    layout="wide"
)
st.title("🏠 House Price Prediction – Model Evaluation Dashboard")
st.markdown("""Predict house prices using **Machine Learning model** trained on historical data, also 
            visualize comparison of multiple regression models performance and its price distribution.""")
st.divider()
# -------------------- LOAD MODELS --------------------
rf_model = joblib.load("trained_models/rf_model.pkl")
best_rf = joblib.load("trained_models/best_rf.pkl")
xgb_model = joblib.load("trained_models/xgb_model.pkl")
best_xgb = joblib.load("trained_models/best_xgb.pkl")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🔢 Enter House Features")

bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 0, 10, 2)
floors = st.sidebar.number_input("Floors", 1, 5, 2)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
view = st.sidebar.selectbox("View", [0, 1, 2, 3, 4])
condition = st.sidebar.slider("Condition", 1, 5, 3)
grade = st.sidebar.slider("Grade", 1, 13, 8)

sqft_living = st.sidebar.number_input("Sqft Living", 300, 10000, 2000)
sqft_lot = st.sidebar.number_input("Sqft Lot", 500, 20000, 10000)
sqft_above = st.sidebar.number_input("Sqft Above", 300, 10000, 2000)
sqft_basement = st.sidebar.number_input("Sqft Basement", 0, 5000, 0)

house_age = st.sidebar.slider("House Age (Years)", 0, 120, 35)
renovated = st.sidebar.selectbox("Renovated", [0, 1])

zipcode = st.sidebar.number_input("Zipcode", 98000, 99999, 98001)
lat = st.sidebar.number_input("Latitude", 47.0, 48.0, 47.5480)
long = st.sidebar.number_input("Longitude", -123.0, -121.0, -121.9836)

sqft_living15 = st.sidebar.number_input("Sqft Living 15", 300, 10000, 2000)
sqft_lot15 = st.sidebar.number_input("Sqft Lot 15", 500, 20000, 10000)

# -------------------- INPUT DATAFRAME --------------------
input_data = pd.DataFrame([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    waterfront, view, condition, grade, sqft_above,
    sqft_basement, house_age, renovated, zipcode,
    lat, long, sqft_living15, sqft_lot15
]], columns=[
    'bedrooms','bathrooms','sqft_living','sqft_lot','floors',
    'waterfront','view','condition','grade','sqft_above',
    'sqft_basement','house_age','renovated','zipcode',
    'lat','long','sqft_living15','sqft_lot15'
])
# -------------------- PREDICTION --------------------
st.subheader("📊 Prediction Result")
if st.button("Predict House Price"):
    st.success("Tuned XGBoost (Best Model): " f"💰 ${best_xgb.predict(input_data)[0]:,.2f}")   
#----------------------------------loadind model's predictions-------------
y_test = np.load("artifacts/y_test.npy")
ypipe_pred = np.load("artifacts/ypipe_pred.npy")
y_pred_ridge = np.load("artifacts/y_pred_ridge.npy")
y_pred_rf = np.load("artifacts/y_pred_rf.npy")
y_pred_xgb = np.load("artifacts/y_pred_xgb.npy")
y_pred_best_rf = np.load("artifacts/y_pred_best_rf.npy")
y_pred_best_xgb = np.load("artifacts/y_pred_best_xgb.npy")
#----------------------------------Load metrics table--------------------
results = pd.read_csv("artifacts/model_results.csv")
#----------------------------------Tabs allocation--------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Comparison Table",
    "📉 Actual vs Predicted Price Distribution",
    "📊 Error & Accuracy Comparison",
    "📌 Conclusion" 
])
#------------------------------------Model comparison table----------------------------
with tab1:
    st.subheader("📊 Model Performance Comparison")
    # Load metrics
    results = pd.read_csv("artifacts/model_results.csv")
    # Display table
    st.dataframe(
                 results.sort_values(by='R2 Score (close to 1 is better)', ascending=False),
                 use_container_width=True,
                 hide_index=True
                )
    st.info("Tuned XGBoost delivers the highest accuracy with the lowest error metrics.")
    st.divider()
#-------------------All models displot (actual price vs predicted price)--------------------------
with tab2:
    st.subheader("📉 Actual vs Predicted Price Distribution")
    fig1, ax = plt.subplots(3,2,figsize=(20,10))
    plt.subplots_adjust(hspace=0.6)
    # Liner Regression
    sns.kdeplot(y_test,ax=ax[0,0])
    sns.kdeplot(ypipe_pred,ax=ax[0,0])
    #Ridge REgression
    sns.kdeplot(y_test,ax=ax[0,1])
    sns.kdeplot(y_pred_ridge,ax=ax[0,1])
    # Random Forrest
    sns.kdeplot(y_test,ax=ax[1,0])
    sns.kdeplot(y_pred_rf,ax=ax[1,0])
    # XGBoost Regressor
    sns.kdeplot(y_test,ax=ax[1,1])
    sns.kdeplot(y_pred_xgb,ax=ax[1,1])
    # Tuned Random Forest
    sns.kdeplot(y_test,ax=ax[2,0])
    sns.kdeplot(y_pred_best_rf,ax=ax[2,0])
    # Tuned XGBoost Regressor
    sns.kdeplot(y_test,ax=ax[2,1])
    sns.kdeplot(y_pred_best_xgb,ax=ax[2,1])
    # legends
    ax[0,0].legend(['Actual Price','Predicted Price'])
    ax[0,1].legend(['Actual Price','Predicted Price'])
    ax[1,0].legend(['Actual Price','Predicted Price'])
    ax[1,1].legend(['Actual Price','Predicted Price'])
    ax[2,0].legend(['Actual Price','Predicted Price'])
    ax[2,1].legend(['Actual Price','Predicted Price'])
    #model name as title
    ax[0,0].set_title('Linear Regression')
    ax[0,1].set_title('Ridge Regression')
    ax[1,0].set_title('Random Forest Regression')
    ax[1,1].set_title('XGBoost Regression')
    ax[2,0].set_title('Tuned Random Forest Regression')
    ax[2,1].set_title('Tuned XGBoost Regression')
    st.pyplot(fig1)
    st.divider()

#-----------------------------All models R2, mae, mse, rmse comparison---------------------------
with tab3: 
     st.subheader("📊 Error & Accuracy Comparison")
     fig2, ax = plt.subplots(2,2,figsize=(20,10))
     plt.subplots_adjust(hspace=1.0)
     sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest Regression', 'XGBoost Regressor',
                    'Tuned Random Forest','Tuned XGBoost Regressor'],
                 y=[metrics.r2_score(y_test,ypipe_pred),
                    metrics.r2_score(y_test,y_pred_ridge),
                    metrics.r2_score(y_test,y_pred_rf),
                    metrics.r2_score(y_test,y_pred_xgb),
                    metrics.r2_score(y_test, y_pred_best_rf),
                    metrics.r2_score(y_test, y_pred_best_xgb)],
                    palette='Set2',ax=ax[0,0])
     sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest','XGBoost Regressor',
                    'Tuned Random Forest','Tuned XGBoost Regressor'],
                 y=[mean_absolute_error(y_test,ypipe_pred),
                    mean_absolute_error(y_test,y_pred_ridge),
                    mean_absolute_error(y_test,y_pred_rf),
                    mean_absolute_error(y_test,y_pred_xgb),
                    mean_absolute_error(y_test, y_pred_best_rf),
                    mean_absolute_error(y_test, y_pred_best_xgb)],
                    palette='Set2',ax=ax[0,1])
     sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest','XGBoost Regressor',
                    'Tuned Random Forest','Tuned XGBoost Regressor'],
                 y=[mean_squared_error(y_test,ypipe_pred),
                    mean_squared_error(y_test,y_pred_ridge),
                    mean_squared_error(y_test,y_pred_rf),
                    mean_squared_error(y_test,y_pred_xgb),
                    mean_squared_error(y_test, y_pred_best_rf),
                    mean_squared_error(y_test, y_pred_best_xgb)],
                    palette='Set2',ax=ax[1,0])
     sns.barplot(x=['Linear Regression','Ridge Regression','Random Forest','XGBoost Regressor',
                    'Tuned Random Forest','Tuned XGBoost Regressor'],
                 y=[np.sqrt(mean_squared_error(y_test,ypipe_pred)),
                    np.sqrt(mean_squared_error(y_test,y_pred_ridge)),
                    np.sqrt(mean_squared_error(y_test,y_pred_rf)),
                    np.sqrt(mean_squared_error(y_test,y_pred_xgb)),
                    np.sqrt(mean_squared_error(y_test, y_pred_best_rf)),
                    np.sqrt(mean_squared_error(y_test, y_pred_best_xgb))],
                    palette='Set2',ax=ax[1,1])
# graph labels
     ax[0,0].set_title('Camparison of Models Accuracy')
     ax[0,1].set_title('Camparison of Models MAE')
     ax[1,0].set_title('Camparison of Models MSE')
     ax[1,1].set_title('Camparison of Models RMSE')
     ax[0,0].set_xlabel('Model Names')
     ax[0,0].tick_params(axis='x', rotation=45)
     ax[0,1].set_xlabel('Model Names')
     ax[0,1].tick_params(axis='x', rotation=45)
     ax[1,0].set_xlabel('Model Names')
     ax[1,0].tick_params(axis='x', rotation=45)
     ax[1,1].set_xlabel('Model Names')
     ax[1,1].tick_params(axis='x', rotation=45)
     ax[0,0].set_ylabel('R2 Score')
     ax[0,1].set_ylabel('Mean Absolute Error')
     ax[1,0].set_ylabel('Mean Squared Error')
     ax[1,1].set_ylabel('Root Mean Squared Error')
     st.pyplot(fig2)
#-------------------------------------------conclusion------------------------------------------
with tab4:
     st.info("""
             ### 📌 Conclusion

            - **Tuned XGBoost** achieves the lowest errors (**MAE, MSE, RMSE**) and the highest **R²**,  
                                making it the most reliable model for housing price prediction.

            - **Random Forest** and **Tuned Random Forest** perform slightly worse than XGBoost,  
                                                            but still outperform linear models.

            - **Linear Regression** and **Ridge Regression** struggle to capture non-linear relationships  
                        in the dataset, as shown by significantly higher error metrics.
            """)
     