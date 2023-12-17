import pandas as pd 
import time
from helper import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier



# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Multilayer Perceptron tuning using GRIDSEARCH ==========================================>

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
    'random_state' : [42]
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
# Accuracy:  0.8448275862068966 
# F1 score : 0.7768595041322315

# Evaluate accuracy using classification_report function
print("Classification Report:\n", classification_report(y_test, y_pred))


# Multilayer Perceptron tuning using RANDOMIZEDSEARCH ==========================================>

# Create the random search object
random_search = RandomizedSearchCV(mlp_classifier, params_mlp, cv=5, scoring=scoring_metrics, refit='f1_weighted', n_iter=10)

# Fit the grid search to the data
start_time = time.time()
random_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)

# Print the best hyperparameters
print("Best Hyperparameters:", random_search.best_params_) # {'solver': 'sgd', 'max_iter': 1000, 'hidden_layer_sizes': (55, 55), 'beta_2': 0.7, 'beta_1': 0.9, 'batch_size': 32, 'activation': 'relu'}

# Make predictions on the test set using the best model
y_pred_rand = random_search.predict(X_test)

# Evaluate the accuracy of the tuned model
accuracy_rand = accuracy_score(y_test, y_pred_rand)
f1score_rand = f1_score(y_test, y_pred_rand)
print("MLP\n","Accuracy: ", accuracy_rand, "\n", "F1 score :", f1score_rand)

# Evaluate accuracy using classification_report function
print("Classification Report:\n", classification_report(y_test, y_pred_rand))
# Accuracy:  0.8563218390804598 
# F1 score : 0.8031496062992126