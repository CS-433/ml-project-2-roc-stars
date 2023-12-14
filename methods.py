import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression ====================================================================>
logreg_model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_logreg))
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)
print("Accuracy on Test Set:", accuracy_logreg)
print("Logistic Regression f1 score: ", f1_logreg) # 0.816

# Gaussian Mixture Matrix ====================================================================>
gmm_model = GaussianMixture(n_components=2, random_state=42)
gmm_model.fit(X_train)
y_pred_gmm = gmm_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred_gmm))
f1score = f1_score(y_test, y_pred_gmm)
print("GMM f1 score: ", f1score) # 0.14184397163120568

# Random Forest Classifier ====================================================================>
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

# Evaluate the accuracy of the best model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
print(f"Best Parameters: {best_params}")
print(f"Best Model Accuracy: {accuracy_rf}")
print(f"Best Model F1 score: {f1_rf}")

# Gradient boost ====================================================================>
gb_classifier = GradientBoostingClassifier(random_state=42)

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)

best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)

# Evaluate the accuracy of the best model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
f1_gb = f1_score(y_test, y_pred_gb)
print("Best Hyperparameters:", grid_search_gb.best_params_)
print("Accuracy on Test Set:", accuracy_gb)
print("F1 score on Test Set:", f1_gb) 


# Polynomial Classification ====================================================================>
poly_svm = make_pipeline(PolynomialFeatures(), SVC())

param_grid_poly = {
    'polynomialfeatures__degree': [2, 3],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.1, 1, 'scale', 'auto'],
}

grid_search_poly = GridSearchCV(estimator=poly_svm, param_grid=param_grid_poly, cv=5, scoring='accuracy')
grid_search_poly.fit(X_train, y_train)

best_params_poly = grid_search_poly.best_params_
best_poly_svm_model = grid_search_poly.best_estimator_
y_pred_poly = best_poly_svm_model.predict(X_test)

accuracy_poly = accuracy_score(y_test, y_pred_poly)
f1_poly = f1_score(y_test, y_pred_poly)
print(f"Best Parameters: {best_params_poly}")
print(f"Best Model Accuracy: {accuracy_poly}")
print(f"Best Model F1 score: {f1_poly}")
