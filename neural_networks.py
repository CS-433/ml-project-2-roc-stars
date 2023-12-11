import numpy as np
import pandas as pd 
import time
from helper import *
 
from sklearn.model_selection import train_test_split

# Import tuning models
from sklearn.model_selection import GridSearchCV

# Import scoring methods
from sklearn.metrics import classification_report

# Import NN related libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the features (recommended for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP
mlp_classifier = MLPClassifier()

# Define the hyperparameter grid to search

param_grid = {
    'hidden_layer_sizes': [(50, 50), (55, 55)],
    'activation': ['relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
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
print("Best Hyperparameters:", grid_search.best_params_) # {'activation': 'relu', 'alpha': 0.5, 'batch_size': 32, 
                                                         # 'beta_1': 0.9, 'beta_2': 0.99, 'hidden_layer_sizes': (50, 50), 
                                                         # 'learning_rate': 'constant', 'max_iter': 1000, 'momentum': 0.95, 
                                                         # 'solver': 'sgd'}

# Make predictions on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
print("MLP\n","Accuracy: ", accuracy, "\n", "F1 score :", f1score)

# Evaluate using kfold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metric 
f1_scorer = make_scorer(f1_score, average='weighted')
# Perform cross-validation and calculate average F1 scores
new_mlp_classifier = MLPClassifier(hidden_layer_sizes=(55, 55), max_iter=1000, random_state=42, beta_2=0.9, beta_1=0.99, solver='adam', activation='relu', batch_size=32, momentum=0.99)

mlp_f1 = cross_val_score(new_mlp_classifier, X_train, y_train, cv=kf, scoring=f1_scorer)

# Calculate average F1 scores
avg_f1_mlp = np.mean(mlp_f1)
print(avg_f1_mlp)


new_mlp_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = new_mlp_classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f1_score(y_test, y_pred))