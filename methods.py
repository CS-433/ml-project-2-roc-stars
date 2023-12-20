# < ----------------------------------Import libraries----------------------------------- >
import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# < ------------------------------------Load data---------------------------------------- >
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
# Insert bias term
df.insert(0, 'Bias', 1)

# Separate inputs and target
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

# < ----------------------------------Split Dataset------------------------------------ >

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < --------------------------------Logistic Regression---------------------------------- >
# Hyperparameters to try
params_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l2', None]
}

# Model definition
logreg_model = LogisticRegression(random_state=42, max_iter=3000)
grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
model_performance(grid_search, X_test, y_test)

# < --------------------------------Random Forest Classifier----------------------------- >
# Hyperparameters to try
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Model definition
rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
model_performance(grid_search_rf, X_test, y_test)

# < --------------------------------Gradient Boosting--------------------------------------- >
# Hyperparameters to try
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Model definition
gb_classifier = GradientBoostingClassifier(random_state=42)
grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5, scoring='accuracy')
grid_search_gb.fit(X_train, y_train)
model_performance(grid_search_gb, X_test, y_test)

# < --------------------------------K Nearest Neighbors---------------------------------- >
# Hyperparameters to try
params_knn = {
    'n_neighbors': [2, 3, 5, 7, 9, 15], 
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
# Model definition
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, params_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
model_performance(grid_search_knn, X_test, y_test)
# {'metric': 'manhattan', 'n_neighbors': 9}

# < --------------------------------Support Vector Machine------------------------------- >
# Hyperparameters to try
params_svm = {
    'C': [0.1, 1, 10],  # adjust as needed
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] 
}
# Model definition
svm_classifier = SVC()
grid_search_svm = GridSearchCV(svm_classifier, params_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
model_performance(grid_search_svm, X_test, y_test)
# {'C': 1, 'gamma': 'auto', 'kernel': 'poly'}

# < ------------------------------Gaussian Mixture Model--------------------------------- >
# Hyperparameters to try
params_gmm = {
    'n_components': [2, 3, 4, 5], 
    'covariance_type': ['full', 'tied', 'diag', 'spherical']
}
# Model definition
gmm = GaussianMixture()

grid_search_gmm = GridSearchCV(gmm, params_gmm, cv=5, scoring='accuracy')  
grid_search_gmm.fit(X_train, y_train)  # Note: y_train is not used for GMM

model_performance(grid_search_gmm, X_test, y_test)
# {'covariance_type': 'diag', 'n_components': 2}
