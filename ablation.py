# < -------------------------------Import libraries-------------------------------------- >
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from helper import *

# < -------------------------------Load, split data-------------------------------------- >
df = pd.read_csv('Datasets/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < -------------------------------Logistic Regression----------------------------------- >
# Model definition
logreg_model = LogisticRegression(random_state=42, max_iter=3000)

# Hyperparameters to try
params_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], 
    'penalty': ['l1', 'l2'], 
}

# < --------------------------------WITHOUT OUTLIERS------------------------------------- >
X_train_nout = remove_outliers(X_train)
X_test_nout = remove_outliers(X_test)
# Tuning and fitting using Grid Search CV
grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='accuracy')
grid_search.fit(X_train_nout, y_train)

model_performance(grid_search, X_test_nout, y_test)

# < --------------------------WITH FEATURE AUGMENTATION---------------------------------- >
# Feature augmentation (polynomial)
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly)
X_poly_df.columns = X_poly_df.columns.astype(str)
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=0)
# Tuning and fitting using Grid Search CV
grid_search = GridSearchCV(logreg_model, params_lr, cv=5, scoring='accuracy')
grid_search.fit(X_train_poly, y_train)

model_performance(grid_search, X_test_poly, y_test)