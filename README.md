# Credit Risk Assessment and Model Optimization Using TPOT and Feature Engineering

## Overview
This project involves a comprehensive pipeline for **credit risk assessment**, leveraging exploratory data analysis (EDA), advanced feature engineering, and automated machine learning using TPOT. The dataset used is sourced from **Home Credit Default Risk**, where the goal is to predict the probability of a client defaulting on a loan.

The project showcases:
- Extensive **EDA** with visualization for key insights.
- Advanced **feature engineering** to derive meaningful predictors.
- **Automated model selection and hyperparameter tuning** using TPOT.
- Evaluation of the model's performance with metrics such as **F1 Score**, **ROC-AUC**, and a detailed **Confusion Matrix**.

---

## Key Features
1. **Exploratory Data Analysis (EDA)**:
   - Distribution and correlation analysis for key variables like `AMT_GOODS_PRICE` and `TARGET`.
   - Insights into target imbalance and its impact on modeling.

2. **Feature Engineering**:
   - Aggregation and transformation of numerical and categorical data across multiple datasets:
     - Application, Bureau, Credit Card Balance, POS Cash, Installments Payments, and Previous Applications.
   - Derivation of new features like `EXT_SOURCE_COMB1`, `CREDIT_OVER_INCOME`, and `AGE_OVER_CHILDREN`.

3. **Automated Machine Learning**:
   - Used TPOT to select the best model pipeline, combining **Gaussian Naive Bayes**, **Random Forest**, and feature selectors like **Linear SVC**.
   - Repeated K-Fold Cross-Validation for robust performance.

4. **Model Evaluation**:
   - Evaluated the model using:
     - **Accuracy**: `85.00%`
     - **Precision**: `26.44%`
     - **Recall**: `39.73%`
     - **F1 Score**: `31.17%`
     - **ROC-AUC Score**: `74.63%`
   - Plots: Confusion Matrix and ROC Curve for insights.

---

## Project Steps
1. **Data Preprocessing**:
   - Handled missing values using Iterative and Simple Imputation.
   - Scaled numerical features using Robust Scaler.
   - Encoded categorical features with Weight of Evidence (WoE) encoding.

2. **Feature Aggregation**:
   - Grouped features from auxiliary datasets (e.g., Bureau, Previous Applications) using aggregations like mean, median, and count.

3. **Dimensionality Reduction**:
   - Selected high-impact features based on correlation with the target variable.

4. **Model Training and Evaluation**:
   - TPOT automated the search for the best model and hyperparameters.
   - Exported the final pipeline for reproducibility.

---

## Results
- **Pipeline**: Gaussian Naive Bayes combined with feature selectors and ensemble methods.
- **Performance Metrics**:
  - **Accuracy**: `85.00%`
  - **ROC-AUC**: `74.63%`
  - **Confusion Matrix**: Visualized via heatmap.
  - **Precision/Recall** trade-off explored through the ROC Curve.

---

## How to Run
### Prerequisites
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tpot`, `category_encoders`

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name.git
