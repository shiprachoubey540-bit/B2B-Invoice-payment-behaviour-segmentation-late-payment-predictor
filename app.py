import streamlit as st
import pandas as pd
import joblib
import os
import xgboost

st.set_page_config(page_title="B2B Predictor", layout="wide")
st.title("📊 B2B Invoice Payment Predictor")

# 1. Load Model
MODEL_PATH = 'model/model.pkl'
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

model = load_model()

# 2. Inputs (The main ones that matter)
st.subheader("Invoice Details")
col1, col2, col3 = st.columns(3)

with col1:
    amount = st.number_input("Amount", value=1000.0)
    region = st.selectbox("Region (Encoded)", [0, 1, 2, 3]) # Simple proxy for Region_enc
with col2:
    pay_term = st.number_input("Payment Term (Days)", value=30)
    cust_age = st.number_input("Customer Age (Months)", value=12)
with col3:
    method = st.selectbox("Payment Method (Encoded)", [0, 1, 2])

# 3. Prediction
if st.button("Run Prediction", use_container_width=True):
    if model:
        try:
            # We must build a dataframe with ALL 30 COLUMNS in the EXACT order shown in your error
            data = pd.DataFrame({
                'Amount': [amount],
                'Payment_Term': [pay_term],
                'Age_Of_Customer_Months': [cust_age],
                'No_of_orders_by_customer': [5],  # Average value
                'Rank_of_order_by_customer': [1], # Average value
                'Quarter_clearing': [2],          # Current Quarter
                'Weekday_clearnum': [1],          # Monday
                'Weekday_due.1': [1], 
                'invoice_month': [4],             # April
                'invoice_quarter': [2],
                'invoice_dayofweek': [0],
                'invoice_dayofmonth': [27],
                'credit_period_days': [pay_term],
                'is_month_end_invoice': [0],
                'is_weekend_invoice': [0],
                'cust_hist_late_rate': [0.2],     # Assume 20% historical late rate
                'cust_hist_avg_overdue': [2.0],   # Assume 2 days average overdue
                'cust_invoice_seq': [1],
                'order_volume_ratio': [1.0],
                'log_amount': [pd.np.log1p(amount) if hasattr(pd, 'np') else 7.0],
                'amount_per_day': [amount/max(pay_term, 1)],
                'is_large_invoice': [1 if amount > 5000 else 0],
                'is_small_invoice': [1 if amount < 500 else 0],
                'pt_bucket_code': [1],
                'amount_x_payterm': [amount * pay_term],
                'amount_x_hist_late': [amount * 0.2],
                'payterm_x_hist_late': [pay_term * 0.2],
                'age_x_orders': [cust_age * 5],
                'Payment_Method_description_enc': [method],
                'Region_enc': [region]
            })

            # XGBoost often needs specific types
            for col in data.columns:
                if 'is_' in col or 'Quarter' in col or 'month' in col:
                    data[col] = data[col].astype(int)
                else:
                    data[col] = data[col].astype(float)

            prediction = model.predict(data)
            
            st.markdown("---")
            if prediction[0] == 1:
                st.error("### ⚠️ Result: Predicted LATE")
            else:
                st.success("### ✅ Result: Predicted ON-TIME")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
    else:
        st.error("Model not found in /model/folder")