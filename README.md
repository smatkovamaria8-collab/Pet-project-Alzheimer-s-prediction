# Alzheimer's Disease Diagnosis Prediction

This project focuses on predicting the diagnosis of Alzheimer's disease using a machine learning pipeline built with Python and `scikit-learn`. The core model uses a Random Forest Classifier optimized through hyperparameter tuning.

## Project Workflow

1. **Data Preprocessing & Cleaning**:
   * Removed non-predictive columns (ID and `DoctorInCharge`).
   * Verified data integrity (confirmed zero duplicate records and zero missing values).

2. **Exploratory Data Analysis (EDA)**:
   * Analyzed correlations with the target variable (`Diagnosis`).
   * Identified key predictors: *Functional Assessment* (r = -0.36), *Memory Complaints* (r = 0.31), *ADL* (r = -0.33), and *Behavior Problems* (r = 0.22).

3. **Model Training & Optimization**:
   * Split the dataset into training (70%) and testing (30%) sets using **stratification** to handle class imbalance.
   * Utilized `GridSearchCV` with 5-fold cross-validation to find the optimal hyperparameters for the Random Forest model.

## Best Hyperparameters
```json
{
  "bootstrap": false,
  "max_depth": 20,
  "min_samples_leaf": 1,
  "min_samples_split": 5,
  "n_estimators": 200
}
```

## Model Evaluation

The optimized Random Forest model demonstrated high overall performance.

```text
              precision    recall  f1-score   support

         0.0       0.95      0.97      0.96       417
         1.0       0.95      0.90      0.93       228

    accuracy                           0.95       645
   macro avg       0.95      0.94      0.94       645
weighted avg       0.95      0.95      0.95       645
```

* **Overall Accuracy**: 95%

## Technology
* Python 3.x
* pandas
* scikit-learn
