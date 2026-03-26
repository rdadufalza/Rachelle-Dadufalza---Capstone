# Credit Card Fraud Detection Pipeline

## 1. Project Overview & Business Problem
Financial institutions lose billions of dollars annually due to credit card fraud. Detecting these fraudulent transactions in real-time is critical to minimizing financial loss and protecting cardholders. However, falsely declining legitimate transactions (False Positives) creates significant customer friction.

This project presents an end-to-end Machine Learning pipeline that effectively distinguishes between fraudulent and legitimate transactions, focusing on maximizing fraud detection while keeping false positives to an absolute minimum.

## 2. Methodologies Used
*   **Data Engineering:** 
    *   Identified and removed duplicate records.
    *   Engineered a new cyclical `Hour` feature from the raw `Time` data.
    *   Applied `RobustScaler` to `Time` and `Amount` features to handle extreme outliers without distorting anomalous signals.
*   **Model Selection:** 
    *   Trained Baseline models (Logistic Regression, Random Forest) with balanced class weights.
    *   Trained an advanced Gradient Boosting model (**XGBoost**) using a calculated `scale_pos_weight` to address the severe class imbalance (~0.17% fraud rate).
*   **Feature Explainability:** 
    *   Utilized **SHAP (SHapley Additive exPlanations)** values to demystify the XGBoost model predictions, identifying PCA features `V17`, `V14`, and `V12` as top predictors.
*   **Bias & Fairness Audit:** 
    *   Synthesized a proxy sensitive attribute (`Income_Bracket`) and evaluated the model using metrics like Demographic Parity, Equalized Odds, and Disparate Impact to ensure fair treatment across transaction sizes.

## 3. Key Findings & Results
*   The **XGBoost** model emerged as the top performer, achieving an excellent balance with a **Precision of 93.6%** and **Recall of 76.8%**.
*   It achieved an **F1-Score of 0.844** and an **Area Under the Precision-Recall Curve (AUPRC) of 0.819**, vastly outperforming baselines on this highly imbalanced dataset.
*   The fairness audit revealed a healthy **Disparate Impact Score of 1.059**, confirming no severe systemic bias across different transaction amount brackets.

## 4. Reproducibility Instructions
1.  Ensure the dataset `creditcard.csv` is placed in the `data/` directory.
2.  Run the consolidated data pipeline function to clean and scale the features.
3.  Train the models using the provided notebook scripts, which automatically utilize stratified splitting and appropriate class weights.
4.  Pre-trained models are saved in the `models/` directory (`xgboost.joblib`, etc.) and can be loaded directly for inference.
