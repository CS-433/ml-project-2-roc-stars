from helper import *
import pandas as pd
from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display best accuracy and F1 scores
mlp = (MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42, beta_2=0.7, beta_1=0.9, solver='adam', activation='relu', batch_size=64))
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_pred_mlp, y_test)
f1_mlp = f1_score(y_pred_mlp, y_test)
print("MLP\n","Accuracy: ", acc_mlp, "\n", "F1 score :", f1_mlp) 
# Accuracy = 0.8908045977011494 
# F1 score : 0.8527131782945736