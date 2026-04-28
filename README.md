https://b2b-invoice-payment-behaviour-segmentation-late-payment-predic.streamlit.app

Dataset: https://www.kaggle.com/datasets/sonalisingh1411/accounts-receivable-and-payment-delay-analysis?select=Dataset.csv   
## Business problem
#### Every company that sells to other businesses (B2B) struggles with 2 questions:
 * Who are the customers that chronically pay late? → Clustering
 * Which specific invoice will be paid late before it's even due? → Prediction
   
This project answers both.
 The final product is a Streamlit app where a finance team uploads invoices and gets live risk scores.
________________________________________
 #### 30 Features  Built and Why Each Exists
### Date-derived (7 features)
•	invoice_month, invoice_quarter, invoice_dayofweek, invoice_dayofmonth —  late payments spike in certain months/quarters

•	credit_period_days — longer credit = more risk

•	is_month_end_invoice — month-end invoices often get delayed

•	is_weekend_invoice — weekend invoices get processed later

### Customer history lag features (4 features) — most important
•	cust_hist_late_rate — this customer's past late payment % (shifted to prevent data leakage)

•	cust_hist_avg_overdue — how many days late on average historically

•	cust_invoice_seq — which invoice number this is for that customer

•	order_volume_ratio — rank of order / total orders (measures order regularity)

### Amount features (4 features)
•	log_amount — raw amount is skewed; log makes it normally distributed

•	amount_per_day — amount ÷ credit period = daily financial pressure

•	is_large_invoice — top 25% by value (large invoices paid differently)

•	is_small_invoice — bottom 25% by value

### Payment term bucket (1 feature)
•	pt_bucket_code — groups 0–30, 31–60, 61–90, 91–120, 120+ days

### Interaction features (4 features)
•	amount_x_payterm — high amount AND long term together

•	amount_x_hist_late — large invoice from a historically late customer

•	payterm_x_hist_late — long credit term given to a bad payer

•	age_x_orders — older customer with more orders (loyalty proxy)

### Encoded categoricals (2 features)
•	Payment_Method_description_enc, Region_enc
________________________________________
### deliverables for Member 2 are two files that get auto-saved when da.py finishes running:
* artifacts/invoices_clean.csv — the fully cleaned dataset with all 30 engineered features already added as columns. Member 2 reads this directly. They do not need to re-clean or re-engineer anything.
* artifacts/feature_names.pkl — a Python list of the exact 30 feature column names in the correct order. Member 2 loads this with joblib.load() so they use exactly the same features that have been built, in the same order.
________________________________________



# Model Training & Evaluation


> **Input from:** Stage 1 (Preprocessing) — `invoices_clean.csv`, `feature_names.pkl`
> **Output to:** Stage 3 (Deployment/API) — `model.pkl`, `scaler.pkl`, `shap_explainer.pkl`

---

###  Objective

Train, evaluate, and select the best machine learning model to predict whether an invoice payment will be **late (1)** or **on-time (0)** using the pre-processed invoice data .

---
---

###  Input Files 

| File | Description |
|------|-------------|
| `invoices_clean.csv` | Cleaned and preprocessed invoice dataset (45,839 rows) |
| `feature_names.pkl` | Approved list of 30 feature columns (no leakage) |


### Pipeline Steps

**1. Data Loading**
- Loaded `invoices_clean.csv` (45,839 invoices)
- Loaded `feature_names.pkl` → 30 features selected
- Target variable: `DelayFlag` (1 = Late, 0 = On-Time)

  **2. Temporal Train-Test Split**
- Sorted by `Doc_Date` (chronological order)
- Train: oldest 80% → 36,671 records (up to 2015-12-11)
- Test: most recent 20% → 9,168 records
- ✅ No random shuffle — prevents data leakage

**3. Class Imbalance Handling**
- Applied SMOTE on training data only
- Train set after SMOTE: 49,660 samples (balanced classes)
- ✅ Test set untouched

**4. Models Trained**

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline, uses scaled features |
| Random Forest | 200 trees, max depth 10 |
| XGBoost | 200 estimators, learning rate 0.05 |

**5. Evaluation Metrics**
All models evaluated on the held-out test set using: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.

---

###  Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.4638 | 0.7384 | 0.1029 | 0.1807 | 0.7744 |
| Random Forest | 0.8652 | 0.9732 | 0.7869 | 0.8702 | 0.9179 |
| **XGBoost ✅** | **0.9443** | **0.9837** | **0.9182** | **0.9498** | **0.9905** |

**🏆 Best Model: XGBoost**
- ROC-AUC of 0.9905 — near-perfect discrimination
- Recall of 91.8% — catches 9 out of 10 late payments
- Precision of 98.4% — almost no false alarms


---

###  SHAP Feature Importance (XGBoost)

Top features driving late payment predictions:

| Rank | Feature | Business Meaning |
|------|---------|-----------------|
| 1 | `Weekday_due.1` | Day of week the payment is due |
| 2 | `order_volume_ratio` | Invoice size vs customer's usual volume |
| 3 | `cust_hist_avg_overdue` | Customer's average historical overdue days |
| 4 | `Weekday_clearnum` | Day of week the invoice was cleared |
| 5 | `Payment_Method_description_enc` | Payment method used by customer |

> **Key insight:** Customer payment history (`cust_hist_avg_overdue`, `cust_hist_late_rate`) and timing features (`Weekday_due.1`, `invoice_dayofweek`) are the strongest predictors of late payment.

---

###  Output Files

| File | Description 
|------|-------------
| `models/model.pkl` | Trained XGBoost model 
| `models/scaler.pkl` | StandardScaler fitted on training data 
| `models/shap_explainer.pkl` | SHAP explainer for predictions 
| `results/metrics.json` | All model evaluation metrics
| `feature_names.pkl` | Feature list 







