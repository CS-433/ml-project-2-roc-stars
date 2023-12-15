import pandas as pd 
from helper import *
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# LOGISTIC REGRESSION ====================================================================>
# Hyperparameters to try
params_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1','l2']
}

# Model definition
logreg_model = LogisticRegression(random_state=42, max_iter=1000)

# Tuning and fitting using Grid Search
grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Best hyperparams
best_params_lr = grid_search.best_params_
print(best_params_lr) # C = 1, penalty='l2'

# Assess accuracy and f1 score
best_lr_model = grid_search.best_estimator_
y_pred_lr = best_lr_model.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1score_lr = f1_score(y_test, y_pred_lr)
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("LR\n","Accuracy: ", accuracy_lr, "\n", "F1 score :", f1score_lr) 
# Accuracy:  0.867816091954023 
# F1 score : 0.816

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

# K NEAREST NEIGHBORS ====================================================================>
# Hyperparameters to try
params_knn = {
    'n_neighbors': [2, 3, 5, 7, 9, 15], 
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()

# Tuning and fitting using Grid Search
grid_search = GridSearchCV(knn, params_knn, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)
best_params_knn = grid_search.best_params_
print("Best Parameters:", best_params_knn) # {'metric': 'manhattan', 'n_neighbors': 9}

# Assess accuracy and f1 score
best_knn_model = grid_search.best_estimator_
y_pred_knn = best_knn_model.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1score_knn = f1_score(y_test, y_pred_knn)
print("KNN\n","Accuracy: ", accuracy_knn, "\n", "F1 score :", f1score_knn)
# Accuracy:  0.8103448275862069 
# F1 score : 0.7027027027027027

# SUPPORT VECTOR MACHINE ====================================================================>
# Hyperparameters to try
params_svm = {
    'C': [0.1, 1, 10], 
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] 
}

# Model definition
svm_classifier = SVC()

# Tuning and fitting using Grid Search
grid_search = GridSearchCV(svm_classifier, params_svm, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Best hyperparams
best_params_svm = grid_search.best_params_
print("Best Parameters:", best_params_svm) # {'C': 1, 'gamma': 'auto', 'kernel': 'poly'}

# Assess accuracy and f1 score
best_svm_model = grid_search.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1score_svm = f1_score(y_test, y_pred_svm)
print("SVM\n","Accuracy: ", accuracy_svm, "\n", "F1 score :", f1score_svm)
# Accuracy:  0.8390804597701149 
# F1 score : 0.7741935483870969

# MULTILAYER PERCEPTRON ====================================================================>
# Hyperparameters to try
params_mlp = {
    'hidden_layer_sizes': [(50, 50), (55, 55)],
    'activation': ['relu'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000],
    'learning_rate': ['constant'],
    'batch_size': [32, 128, 256, 512],
    'momentum': [0.9, 0.95,0.99],
    'beta_1': [0.9, 0,99],
    'beta_2': [0.9, 0.99],
}

# Model definition
mlp_classifier = MLPClassifier()

# Create the grid search object
scoring_metrics = ['accuracy', 'f1_weighted']
grid_search = GridSearchCV(mlp_classifier, params_mlp, cv=5, scoring=scoring_metrics, refit='f1_weighted')

# Tuning and fitting using Grid Search
start_time = time.time()
grid_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)

# Best hyperparams
best_params_mlp = grid_search.best_params_
print("Best Hyperparameters:", best_params_mlp) # {'activation': 'relu', 'batch_size': 256, 'beta_1': 0.9, 'beta_2': 0.99, 'hidden_layer_sizes': (55, 55), 'learning_rate': 'constant', 'max_iter': 1000, 'momentum': 0.99, 'solver': 'adam'}

# Assess accuracy and f1 score
best_mlp_model = grid_search.best_estimator_
y_pred_mlp = best_mlp_model.predict(X_test)
#y_pred_mlp = grid_search.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1score_mlp = f1_score(y_test, y_pred_mlp)
print("MLP\n","Accuracy: ", accuracy_mlp, "\n", "F1 score :", f1score_mlp)
# Accuracy:  0.8563218390804598 
# F1 score : 0.7967479674796748


# GAUSSIAN MIXTURE MODEL ====================================================================>
# Hyperparameters to try
params_gmm = {
    'n_components': [2, 3, 4, 5], 
    'covariance_type': ['full', 'tied', 'diag', 'spherical']
}

# Model definition
gmm = GaussianMixture()

# Tuning and fitting using Grid Search
grid_search = GridSearchCV(gmm, params_gmm, cv=5, scoring='f1_weighted')  
grid_search.fit(X_train, y_train)  # Note: y_train is not used for GMM

# Best hyperparams
best_params = grid_search.best_params_
print("Best Parameters:", best_params) # {'covariance_type': 'diag', 'n_components': 2}

# Assess accuracy and f1 score
y_pred_gmm = grid_search.predict(X_test)
accuracy_gmm = accuracy_score(y_test, y_pred_gmm)
f1score_gmm = f1_score(y_test, y_pred_gmm)
print("GMM\n","Accuracy: ", accuracy_gmm, "\n", "F1 score :", f1score_gmm)
# Accuracy:  0.7988505747126436 
# F1 score : 0.6846846846846846