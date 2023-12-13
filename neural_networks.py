import pandas as pd 
import time
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MULTILAYER PERCEPTRON

# MLP
mlp_classifier = MLPClassifier()

# Define the hyperparameter grid to search

param_grid = {
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

# Create the grid search object
scoring_metrics = ['accuracy', 'f1_weighted']
grid_search = GridSearchCV(mlp_classifier, param_grid, cv=5, scoring=scoring_metrics, refit='f1_weighted')

# Fit the grid search to the data
start_time = time.time()
grid_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_) # {'activation': 'relu', 'alpha': 0.0001, 'batch_size': 256, '
                                                         # beta_1': 0.9, 'beta_2': 0.99, 'hidden_layer_sizes': (55, 55), 
                                                         # 'learning_rate': 'constant', 'max_iter': 1000, 'momentum': 0.99, 
                                                         # 'solver': 'sgd'}
                                                         # attention ce sont pas ces params qu'on utilise

# Make predictions on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the accuracy of the tuned model
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
print("MLP\n","Accuracy: ", accuracy, "\n", "F1 score :", f1score)


# Evaluate accuracy using classification_report function
print("Classification Report:\n", classification_report(y_test, y_pred))