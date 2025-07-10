![Description](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHR2b3FrOTk0Mzg0OTFpZmRsanc4bXRxcWlpcTU2Y3hmb2lmNTg4MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/12FBrUdUj7ZkuzyI8G/giphy.gif)

# Dara Analysis: Basic Examples

This project contains basic examples of EDA and predictive modeling using Python and popular libraries. It uses a diabetes dataset to demostrate key steps in the data analysis.

---

## Dataset Used
- The dataset used is `diabetes.csv`, which includes medical variables for patients along with a binary target (`Outcome`) indicating whether or not the patient has diabetes.

---

# Libraries Used

- `pandas`
- `numpy`
- `matplotlib
- `seaborn
- `scikit-learn

---
## 📌 Analysis Workflow

### ✅ 1. Data Loading and Cleaning
- Load data from CSV.
- Replace invalid `0` values in key columns with `NaN`:
- Impute missing values using the median.

### 📊 2. Exploratory Data Analysis (EDA)
- Count plot of diabetes vs non-diabetes cases.
- Correlation matrix of all features.
- Boxplots showing feature distributions by `Outcome`.
- Outlier detection using boxplots.
- Scatter plots to explore relationships between features (e.g., Glucose vs BMI).

### 🔍 3. Predictive Modeling
- Feature scaling using `StandardScaler`.
- Train-test split of the dataset.
- Train a **Logistic Regression** model.
- Model evaluation with:
  - Confusion matrix
  - Classification report (`precision`, `recall`, `f1-score`)

---




