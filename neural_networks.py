# < -------------------------------Import libraries-------------------------------------- >
import pandas as pd 
import time
from helper import *
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

# < ---------------------------------Load dataset---------------------------------------- >
df = pd.read_csv('Datasets/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < ---------------Multilayer Perceptron tuning using GRIDSEARCH CV------------------------- >
# Define model
mlp_classifier = MLPClassifier()

# Hyperparameters to try
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

# Grid search CV
scoring_metrics = ['accuracy', 'f1_weighted']
grid_search = GridSearchCV(mlp_classifier, params_mlp, cv=5, scoring=scoring_metrics, refit='f1_weighted')

start_time = time.time()
grid_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)
print("Best Hyperparameters:", grid_search.best_params_) # {'activation': 'tanh', 'batch_size': 64, 'beta_1': 0, 'beta_2': 0.99, 'hidden_layer_sizes': (50, 50), 'max_iter': 1000, 'solver': 'adam'}

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)
print("MLP grid search\n","Accuracy: ", accuracy, "\n", "F1 score:", f1score)

# < ---------------Multilayer Perceptron tuning using RANDOMIZEDSEARCH CV------------------- >
# Random search CV
random_search = RandomizedSearchCV(mlp_classifier, params_mlp, cv=5, scoring=scoring_metrics, refit='f1_weighted', n_iter=10)

start_time = time.time()
random_search.fit(X_train, y_train)
stop_time = time.time()
fitting_time = stop_time-start_time
print("Fitting time: ", fitting_time)
print("Best Hyperparameters:", random_search.best_params_) # {'solver': 'sgd', 'max_iter': 1000, 'hidden_layer_sizes': (55, 55), 'beta_2': 0.7, 'beta_1': 0.9, 'batch_size': 32, 'activation': 'relu'}

y_pred_rand = random_search.predict(X_test)
accuracy_rand = accuracy_score(y_test, y_pred_rand)
f1score_rand = f1_score(y_test, y_pred_rand)
print("MLP random search\n","Accuracy: ", accuracy_rand, "\n", "F1 score:", f1score_rand)

# < ---------------Multilayer Perceptron tuning using manual tuning---------------------- >
mlp_man = MLPClassifier(hidden_layer_sizes=(55, 55), max_iter=1000, random_state=42, beta_2=0.8, beta_1=0.8, solver='sgd', activation='relu', batch_size=32)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores = []

# Perform 5-fold cross-validation
for train_index, val_index in cv.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    mlp_man.fit(X_train_fold, y_train_fold)

    y_pred_val = mlp_man.predict(X_val_fold)
    f1_fold = f1_score(y_val_fold, y_pred_val, average='micro')
    f1_scores.append(f1_fold)

# Print the F1 scores for each fold
print("F1 Scores for each fold:", f1_scores)

# Print the mean F1 score
print("Mean F1 Score:", np.mean(f1_scores))

# Assess performance
y_pred_man = mlp_man.predict(X_test)
accuracy_man = accuracy_score(y_test, y_pred_man)
f1score_man = f1_score(y_test, y_pred_man)
print("MLP manual tuning\n","Accuracy: ", accuracy_man, "\n", "F1 score:", f1score_man)
# BEST:
# (hidden_layer_sizes=(55, 55), max_iter=1000, random_state=42, beta_2=0.8, beta_1=0.8, solver='sgd', activation='relu', batch_size=32)