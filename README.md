# Healthcare Insurance Fraud Detection

This project presents a machine learning-based system to detect fraudulent healthcare insurance claims using Medicare datasets. Developed as part of the **Data Mining Techniques (SWE2009)** course at **VIT University**, the system identifies suspicious patterns using various classification models and data preprocessing techniques.

---

## Project Objectives

- Analyze Medicare insurance claim data involving patients, providers, diagnosis codes, and payment information.
- Detect fraudulent patterns using machine learning algorithms.
- Build predictive models to classify claims as fraudulent or legitimate.
- Address class imbalance and improve model reliability using evaluation metrics and threshold tuning.

---

## Tech Stack

- **Language**: Python 3.10.4  
- **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, FancyImpute, Statsmodels  
- **Environment**: Jupyter Notebook  

---

## Models Implemented

- Logistic Regression
- Decision Tree
- Random Forest
- Gaussian Naive Bayes
- K-Nearest Neighbors

**Random Forest** performed best with:
- **Recall**: 100%
- **F1-score**: 0.78
- **Kappa Score**: 3.36

---

## Dataset & Preprocessing

- Datasets:
  - `Train.csv`, `Test.csv`
  - `Inpatient`, `Outpatient`, and `Beneficiary` datasets

- Techniques:
  - Missing value imputation using `IterativeImputer`
  - Outlier handling with IQR method
  - Label encoding for categorical variables
  - Feature scaling with `MinMaxScaler` and `RobustScaler`
  - Merging datasets using keys like `Provider`, `ClaimID`, and `BeneID`

---

## Project Modules

### 1. **Data Collection & Cleaning**
- Loaded and merged datasets
- Cleaned data using imputation and outlier treatment

### 2. **Model Development**
- Trained multiple ML models
- Evaluated using accuracy, precision, recall, F1-score, AUC, and Cohen’s Kappa
- Used GridSearchCV for hyperparameter tuning

### 3. **Model Evaluation**
- Performance comparison across models
- Random Forest achieved optimal results for fraud detection with high sensitivity

---

## Evaluation Summary

| Model               | Accuracy | Precision | Recall | F1 Score | AUC  | Kappa |
|--------------------|----------|-----------|--------|----------|------|--------|
| Logistic Regression| 0.64     | 0.64      | 1.00   | 0.78     | 0.50 | 0.00   |
| Decision Tree      | 0.64     | 0.64      | 0.99   | 0.78     | 0.52 | 0.20   |
| **Random Forest**  | 0.64     | 0.64      | **1.00**| **0.78** | 0.50 | **3.36** |
| GaussianNB         | 0.62     | 0.63      | 0.94   | 0.76     | 0.49 | 0.001  |
| KNN                | 0.67     | 0.73      | 0.77   | 0.75     | 0.63 | 0.28   |

---

#### Dataset & Preprocessing

- **Dataset Source**: [Healthcare Provider Fraud Detection Analysis – Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
- Files Used:
  - `Train.csv`, `Test.csv`
  - `Inpatient`, `Outpatient`, and `Beneficiary` datasets
