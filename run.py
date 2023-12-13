from helper import *
import pandas as pd
from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning
from sklearn.neural_network import MLPClassifier


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# MLP
new_mlp_classifier = MLPClassifier(hidden_layer_sizes=(55, 55), max_iter=1000, random_state=42, beta_2=0.99, beta_1=0, solver='adam', activation='relu', batch_size=256, momentum=0.99)
new_mlp_classifier.fit(X_train, y_train)
y_pred_mlp = new_mlp_classifier.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1score_mlp = f1_score(y_test, y_pred_mlp)
print("MLP\n","Accuracy: ", accuracy_mlp, "\n", "F1 score :", f1score_mlp)
# Accuracy:  0.8793103448275862 
# F1 score : 0.8292682926829269