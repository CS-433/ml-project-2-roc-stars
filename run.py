import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
print(df)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic regression without penalty
logreg = LogisticRegression(penalty = None)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy: ', acc)
print("F1 Score:", f1)

# Logistic regression with L2 penalty
logreg_l2 = LogisticRegression(penalty = 'l2', max_iter = 500, fit_intercept = True, solver = 'lbfgs')
logreg_l2.fit(X_train, y_train)
y_pred_l2 = logreg_l2.predict(X_test)
acc_l2 = accuracy_score(y_test, y_pred_l2)
f1_l2 = f1_score(y_test, y_pred_l2)
print('Accuracy: ', acc_l2)
print("F1 Score:", f1_l2)


# Random Forest Classifier
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier2 = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf_classifier2, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and model
best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_

y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the accuracy of the best model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Best Parameters: {best_params}")
print(f"Best Model Accuracy: {accuracy_rf}")
print(f"Best Model F1 score: {f1_rf}")


# Polynomial Classification
poly_svm = make_pipeline(PolynomialFeatures(), SVC())

param_grid = {
    'polynomialfeatures__degree': [2, 3],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.1, 1, 'scale', 'auto'],
}

grid_search = GridSearchCV(estimator=poly_svm, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params2 = grid_search.best_params_
best_poly_svm_model = grid_search.best_estimator_

y_pred_svm = best_poly_svm_model.predict(X_test)

accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)
print(f"Best Parameters: {best_params2}")
print(f"Best Model Accuracy: {accuracy_svm}")
print(f"Best Model F1 score: {f1_svm}")