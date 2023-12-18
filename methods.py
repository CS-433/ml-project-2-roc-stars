# < ----------------------------------Import libraries----------------------------------- >
import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# < ------------------------------------Load data---------------------------------------- >
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
# Insert bias term
df.insert(0, 'Bias', 1)

# Separate inputs and target
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

# < -------------------------------Feature Agumentation---------------------------------- >
# Feature augmentation (polynomial)
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly)
X_poly_df.columns = X_poly_df.columns.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=0)

# < --------------------------------Logistic Regression---------------------------------- >
# Hyperparameters to try
params_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l2', None]
}

# Model definition
logreg_model = LogisticRegression(random_state=42, max_iter=3000)
grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)
model_performance(grid_search, X_test, y_test)

# < --------------------Logistic Regression with feature augmentation-------------------- >
logreg_poly = LogisticRegression(random_state=42, penalty='l2', C=0.01, max_iter=2000)
logreg_poly.fit(X_train_poly, y_train_poly)
model_performance(logreg_poly, X_test_poly, y_test_poly, CV=False)

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

# < --------------------------------Gradient Boost--------------------------------------- >
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
grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=5, scoring='f1_weighted')
grid_search_gb.fit(X_train, y_train)
model_performance(grid_search_gb, X_test, y_test)

# < ----------------------------Polynomial Classification-------------------------------- >
# Hyperparameters to try
param_grid_poly = {
    'polynomialfeatures__degree': [2, 3],
    'svc__C': [0.1, 1, 10],
    'svc__gamma': [0.1, 1, 'scale', 'auto'],
}
# Model definition
poly_svm = make_pipeline(PolynomialFeatures(), SVC())
grid_search_poly = GridSearchCV(estimator=poly_svm, param_grid=param_grid_poly, cv=5, scoring='f1_weighted')
grid_search_poly.fit(X_train, y_train)
model_performance(grid_search_poly, X_test, y_test)

# < --------------------------------K Nearest Neighbors---------------------------------- >
# Hyperparameters to try
params_knn = {
    'n_neighbors': [2, 3, 5, 7, 9, 15], 
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
# Model definition
knn = KNeighborsClassifier()
grid_search_knn = GridSearchCV(knn, params_knn, cv=5, scoring='f1_weighted')
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
grid_search_svm = GridSearchCV(svm_classifier, params_svm, cv=5, scoring='f1_weighted')
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

grid_search_gmm = GridSearchCV(gmm, params_gmm, cv=5, scoring='f1_weighted')  
grid_search_gmm.fit(X_train, y_train)  # Note: y_train is not used for GMM

model_performance(grid_search_gmm, X_test, y_test)
# {'covariance_type': 'diag', 'n_components': 2}

# < -----------------------------Gaussian Process Classifier----------------------------- >
# Hyperparameters to try
param_grid_gpc = {
    'kernel': [C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))],
    'optimizer': ['fmin_l_bfgs_b'],
    'n_restarts_optimizer': [0, 1, 2],
    'max_iter_predict': [100, 200, 300],
}
# Model definition
gpc = GaussianProcessClassifier(random_state=42)
grid_search_gpc = GridSearchCV(gpc, param_grid=param_grid_gpc, cv=5)
grid_search_gpc.fit(X_train, y_train)
model_performance(grid_search_gpc, X_test, y_test)
# {'kernel': 1**2 * RBF(length_scale=1), 'max_iter_predict': 100, 'n_restarts_optimizer': 0, 'optimizer': 'fmin_l_bfgs_b'}
