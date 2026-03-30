# Supply Chain Delay Prediction Using The Olist Brazilian E-Commerce Dataset

## Project Summary
This project is a production-style end-to-end data science system for shipment delay prediction using the Olist Brazilian e-commerce dataset. It combines data engineering, feature engineering, supervised machine learning, evaluation, and a Streamlit dashboard to support a realistic supply-chain use case.

The system predicts whether a delivered order is late relative to the promised delivery date and presents the results in an operations-focused dashboard for exploratory analysis, risk monitoring, and shipment-level scenario scoring.

## Business Objective
The goal is to help operations, fulfillment, and logistics teams answer three practical questions:

1. Which deliveries are likely to be delayed?
2. Which shipment patterns and lanes are operationally risky?
3. How much improvement do stronger features and stronger models provide over a baseline system?

## Approved Dataset Scope
Only the following CSV files are used:

- `raw_dataset/olist_orders_dataset.csv`
- `raw_dataset/olist_customers_dataset.csv`
- `raw_dataset/olist_sellers_dataset.csv`
- `raw_dataset/olist_geolocation_dataset.csv`

## Problem Definition
- `delay_days = order_delivered_customer_date - order_purchase_timestamp`
- `is_delayed = 1` if `order_delivered_customer_date > order_estimated_delivery_date`

This makes the project suitable for operational delay-risk classification rather than exact ETA regression.

## End-to-End Methodology
### 1. Data Preparation
- Load and validate the four approved source files
- Restrict the working set to delivered orders
- Aggregate the raw geolocation table at ZIP-prefix level to reduce duplicate coordinate noise
- Merge orders with customers
- Resolve seller geography separately from seller ZIP prefixes
- Build customer and seller coordinate references using ZIP and state-level fallbacks

### 2. Feature Engineering
- Temporal features: `day_of_week`, `month`, `purchase_hour`, `is_weekend`
- Planning feature: `promised_lead_days`
- External feature: `is_holiday` using Brazil holidays
- Spatial feature: `distance_km`
- Operational proxy fields: `proxy_seller_state`, `distance_band`, `route_pair`, `seller_state_match`

### 3. Modeling
- Baseline model: Logistic Regression
- Advanced model: XGBoost
- Comparative setups:
  - Without distance features
  - With distance and operational proxy features
- Validation-based threshold tuning for final classification decisions

### 4. Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Comparative analysis across baseline and improved systems

### 5. Dashboard
- Streamlit-based interactive dashboard
- Dynamic filters for month, weekday, and distance
- Operational heatmap for lane-risk monitoring
- Shipment-level prediction interface using location-driven inputs

## Dataset Cleaning Summary
- Raw orders loaded: `99,441`
- Raw customers loaded: `99,441`
- Raw sellers loaded: `3,095`
- Raw geolocation rows loaded: `1,000,163`
- Delivered orders retained: `96,478`
- Delivered orders with complete target timestamps: `96,470`
- Final model-ready rows: `96,470`
- Train rows: `61,740`
- Validation rows: `15,436`
- Test rows: `19,294`

### Cleaning Logic
- Non-delivered orders were excluded because final delivery outcome is unknown
- Rows with missing purchase, delivered, or estimated delivery timestamps were removed from the target-ready set
- Geolocation rows were aggregated at ZIP-prefix level
- Missing coordinates were handled using state-level geolocation fallbacks
- Invalid rows with negative delivery duration or negative promised lead time were excluded

## Technical Limitation
The approved four-file subset does not contain an order-level `seller_id` bridge between orders and sellers. Because of that, the system uses a seller-location proxy instead of the exact seller tied to each shipment. This means the spatial feature is operationally useful, but it should be interpreted as a risk signal rather than an exact shipment route distance.

## Final Model Performance
Best model: `XGBoost | With Distance`

| Model | Feature Set | Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| XGBoost | With Distance | 0.66 | 0.8545 | 0.2681 | 0.4588 | 0.3384 |
| XGBoost | Without Distance | 0.64 | 0.8385 | 0.2467 | 0.4824 | 0.3264 |
| Logistic Regression | With Distance | 0.64 | 0.8186 | 0.2025 | 0.4204 | 0.2733 |
| Logistic Regression | Without Distance | 0.64 | 0.8183 | 0.2015 | 0.4185 | 0.2720 |

## Repository Structure
```text
DS HOTS/
├── app.py
├── data_processing.py
├── evaluation.py
├── feature_engineering.py
├── model_training.py
├── raw_dataset/
├── artifacts/
│   ├── figures/
│   ├── best_model_bundle.joblib
│   ├── metadata.json
│   ├── model_comparison.csv
│   ├── model_comparison.json
│   ├── processed_delay_dataset.csv
│   └── xgboost_feature_importance.csv
├── screenshots/
├── requirements.txt
├── run.txt
└── README.md
```

## Run Instructions
```powershell
python -m pip install -r requirements.txt
python model_training.py
python -m streamlit run app.py
```

## Dashboard Screenshots
### 1. Dashboard Landing View
![Dashboard Landing](<screenshots/Screenshot 2026-03-29 010022.png>)

### 2. Project Explanation Section
![Project Explanation](<screenshots/Screenshot 2026-03-29 010037.png>)

### 3. Methodology And Limitations View
![Methodology And Limitations](<screenshots/Screenshot 2026-03-29 010053.png>)

### 4. Dataset Cleaning Summary And KPI Context
![Cleaning Summary](<screenshots/Screenshot 2026-03-29 010109.png>)

### 5. KPI Overview With Distribution And Distance Analysis
![KPI Overview](<screenshots/Screenshot 2026-03-29 010137.png>)

### 6. Filtered Operational Overview
![Filtered Overview](<screenshots/Screenshot 2026-03-29 010156.png>)

### 7. Chart Interpretation Layer
![Chart Interpretation](<screenshots/Screenshot 2026-03-29 010233.png>)

### 8. Temporal Delay Analysis
![Temporal Analysis](<screenshots/Screenshot 2026-03-29 010246.png>)

### 9. Model Comparison And Feature Importance
![Model Comparison](<screenshots/Screenshot 2026-03-29 010303.png>)

### 10. Operational Lane Risk Heatmap
![Lane Risk Heatmap](<screenshots/Screenshot 2026-03-29 010323.png>)

### 11. Prediction Interface
![Prediction Interface](<screenshots/Screenshot 2026-03-29 010514.png>)

### 12. Prediction Scenario Output
![Prediction Output](<screenshots/Screenshot 2026-03-29 010532.png>)

## Generated Figures
### Delay Distribution
![Delay Distribution](<artifacts/figures/delay_distribution.png>)

### Delay By Day Of Week
![Delay By Day Of Week](<artifacts/figures/delay_by_day_of_week.png>)

### Monthly Delay Trend
![Monthly Delay Trend](<artifacts/figures/monthly_delay_trend.png>)

### Model Comparison
![Model Comparison Figure](<artifacts/figures/model_comparison.png>)

### XGBoost Feature Importance
![XGBoost Feature Importance](<artifacts/figures/xgboost_feature_importance.png>)

### Operational Lane Risk Heatmap
![Operational Lane Heatmap](<artifacts/figures/lane_risk_heatmap.png>)

### Best Model Confusion Matrix
![Confusion Matrix](<artifacts/figures/best_model_confusion_matrix.png>)

## Closing Note
This project is designed to demonstrate a realistic applied machine learning workflow for supply-chain operations: curated data scope, traceable assumptions, comparative modeling, interactive business-facing analytics, and reproducible artifacts for reporting.
