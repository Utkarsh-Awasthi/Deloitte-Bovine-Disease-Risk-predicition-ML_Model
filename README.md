# Deloitte-Bovine-Disease-Risk-predicition-ML_Model
Machine learning pipeline for predicting bovine tuberculosis test outcomes in Northern Ireland farms (2020–2022). Includes EDA, feature engineering, class imbalance handling, and ensemble models (Logistic Regression, Random Forest, LightGBM). Evaluated using AUC and temporal validation. Built with Python and scikit-learn.
# 🐄 Bovine Tuberculosis Test Outcome Prediction

> Machine learning pipeline for predicting bovine tuberculosis (bTB) test
> outcomes at the individual animal level using farm, animal, environmental,
> and biosecurity features from Northern Ireland farms (2020–2022).

---

## 📋 Project Overview

Bovine tuberculosis (bTB) remains one of the most costly and persistent
livestock diseases in Northern Ireland and Great Britain, with significant
economic and animal welfare impacts. This project builds a supervised
binary classification pipeline to predict individual animal test results
(Positive / Negative) using a synthetic but domain-realistic dataset of
**10,000 records** across **1,000 farms** and **9,771 unique animals**.

The project covers the full data science workflow: exploratory analysis,
preprocessing, feature engineering, class imbalance handling, model
comparison, and evaluation using clinically relevant metrics.

---

## 📁 Dataset Summary

| Property            | Value                          |
|---------------------|-------------------------------|
| Records             | 10,000 test records           |
| Unique Farms        | 1,000                         |
| Unique Animals      | 9,771                         |
| Features            | 27 (farm, animal, environment)|
| Date Range          | January 2020 – December 2022  |
| Target              | `Test_Result` (Positive / Negative) |
| Class Distribution  | 89.2% Negative / 10.8% Positive |

### Feature Categories

| Category         | Features (count) | Examples                                      |
|------------------|-----------------|-----------------------------------------------|
| Farm-Level       | 10              | Herd size, farm type, proximity to other farms|
| Animal-Level     | 5               | Breed, age (months), sex, origin type         |
| Environmental    | 4               | Rainfall, temperature, wildlife proximity     |
| Test & Target    | 8               | Biosecurity score, cattle movements, season   |

> ⚠️ **Class imbalance:** The dataset has a ~9:1 negative-to-positive ratio,
> requiring specific handling strategies (class weighting, threshold tuning).

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Distribution plots for all 27 features
- Farm type breakdown: Beef (41.8%), Dairy (33.0%), Mixed (25.2%)
- Breed distribution across 6 breeds (~16–17% each)
- Correlation analysis and seasonal test pattern analysis
- Missing data audit: 3 admin-only columns with 13–21% missingness;
  all 24 analytical features are 100% complete

### 2. Preprocessing
- Label encoding for categorical features
- Standardisation of numeric features using `StandardScaler`
- Train/test split with stratification (80/20)
- Class imbalance addressed via `class_weight='balanced'` and SMOTE

### 3. Models Evaluated

| Model                | AUC    | Accuracy | Notes                              |
|----------------------|--------|----------|------------------------------------|
| Logistic Regression  | ~0.81  | ~83%     | Baseline linear model              |
| Random Forest        | ~0.89  | ~91%     | Best overall generalisation        |
| LightGBM             | ~0.91  | ~91.3%   | Best AUC; handles imbalance well   |

### 4. Evaluation Metrics
- **ROC-AUC** (primary metric — appropriate for imbalanced classification)
- Precision, Recall, F1-Score
- Confusion matrix analysis
- Threshold sensitivity analysis (precision/recall trade-off)

---

## 📊 Key Results

- **Best Model:** LightGBM with `class_weight='balanced'`
- **Test AUC:** ~0.91
- **Test Accuracy:** 91.3%
- **Precision (Positive class):** 44.8%
- **Recall (Positive class):** ~72%

> 📌 The precision/recall trade-off reflects the inherent difficulty of
> detecting a rare positive class (10.8%). In a disease surveillance context,
> **recall (sensitivity) is prioritised** to minimise missed positives,
> accepting a higher false positive rate.

### Top Predictive Features (LightGBM Feature Importance)
1. **Disease History (Farm)** — prior herd breakdowns are the strongest predictor
2. **Wildlife Proximity (km)** — spatial proximity to badger setts
3. **Biosecurity Score** — farm-level biosecurity practices (1–5 scale)
4. **Cattle Movements** — number of cattle movements in the last year
5. **Farm Location** — spatial coordinates capturing regional hotspots

---

## 🛠️ Tech Stack

| Tool            | Use                                      |
|-----------------|------------------------------------------|
| Python 3.11     | Core language                            |
| pandas / NumPy  | Data manipulation                        |
| scikit-learn    | Preprocessing, LR, RF, evaluation       |
| LightGBM        | Gradient boosting classifier             |
| imbalanced-learn| SMOTE oversampling                       |
| Matplotlib / Seaborn | Visualisation                       |
| Jupyter Notebook | Development environment                 |

---

## 📂 Repository Structure
