import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning

# Import model
from sklearn.linear_model import LogisticRegression

# Import tuning models
from sklearn.model_selection import GridSearchCV

# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
print(1)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
print(2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Hyperparameters to test
params = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}

# Create the model
logreg_model = LogisticRegression(random_state=42, max_iter=1000)

# Evaluates the model performance for each combination of hyperparameter values using cross-validation
grid_search = GridSearchCV(logreg_model, params, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

print(best_params)

# Got best_params = {'C': 1, 'penalty': 'l2'}