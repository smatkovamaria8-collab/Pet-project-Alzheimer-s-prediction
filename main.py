import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("alzheimers_disease_data.csv")

data = data.iloc[:, 1:]
data = data.drop("DoctorInCharge", axis = 1)
data.info()
print(data.duplicated().sum())

## The database does not contain any duplicates or null values

data = data.astype("float64")

print(data.corrwith(data['Diagnosis']))

## The highest correlation with the diagnosis of Alzheimer's is found in the following variables:
# Functional Assessment (-0.36 negative weak correlation), Memory Complaints (0.31 positive weak correlation), 
# Behavior Problems (0.22 positive weak correlation), ADL (Activities of Daily Living) (-0.33 negative weak correlation)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Hyperparameter tuning was performed to find the optimal parameters for the model.

param_grid = {
    'n_estimators' : [100, 200],
    'max_depth' : [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

## The Random Forest Classifier was chosen as the model.

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Estimator:", grid_search.best_estimator_)

## The best parameters - {'bootstrap': False, 'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}

best_model_rf = grid_search.best_estimator_
y_pred = best_model_rf.predict(X_test)
print(classification_report(y_test, y_pred))

##             precision    recall  f1-score   support

#         0.0       0.95      0.97      0.96       417
#         1.0       0.95      0.90      0.93       228

#    accuracy                           0.95       645
#   macro avg       0.95      0.94      0.94       645
#weighted avg       0.95      0.95      0.95       645

## The model demonstrated high precision, recall, accuracy, and F1-scores