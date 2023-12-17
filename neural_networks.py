import pandas as pd 
import time
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Multilayer Perceptron
"""
# Standardize the features (recommended for neural networks)
scaler = StandardScaler()
X_train_sd = scaler.fit_transform(X_train)
X_test_sd = scaler.transform(X_test)
"""

# MLP
mlp_classifier = MLPClassifier()

# Define the hyperparameter grid to search

params_mlp = {
    'hidden_layer_sizes': [(50, 50), (55, 55)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'max_iter': [1000],
    'batch_size': [32, 64, 128, 256, 512],
    'beta_1': [0.9, 0,99],
    'beta_2': [0.7, 0.8, 0.9, 0.99],
}

# Create the grid search object
scoring_metrics = ['accuracy', 'f1_weighted']
grid_search = GridSearchCV(mlp_classifier, params_mlp, cv=5, scoring=scoring_metrics, refit='f1_weighted')

# Fit the grid search to the data
start_time = time.time()
grid_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_) # {'activation': 'tanh', 'batch_size': 64, 'beta_1': 0, 'beta_2': 0.99, 'hidden_layer_sizes': (50, 50), 'max_iter': 1000, 'solver': 'adam'}

# Make predictions on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the accuracy of the tuned model
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
print("MLP\n","Accuracy: ", accuracy, "\n", "F1 score :", f1score)

# Evaluate accuracy using classification_report function
print("Classification Report:\n", classification_report(y_test, y_pred))