from helper import *
import pandas as pd

from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display average F1 scores
f1_logreg, acc_logreg = performance(LogisticRegression(), X_train, y_train)
f1_knn, acc_knn = performance(KNeighborsClassifier(), X_train, y_train)
f1_svm, acc_svm = performance(SVC(), X_train, y_train)
f1_mlp, acc_mlp = performance(MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42, beta_2=0.9, beta_1=0.99, solver='sgd', activation='relu', batch_size=128, momentum=0.99), X_train, y_train)

print("Logistic regression\n","Accuracy: ", acc_logreg, "\n", "F1 score :", f1_logreg) # f1 = 0.8654894678589372
print("KNN\n","Accuracy: ", acc_knn, "\n", "F1 score :", f1_knn) # f1 = 0.7929342986844565
print("SVM\n","Accuracy: ", acc_svm, "\n", "F1 score :", f1_svm) # f1 = 0.8239125211396205
print("MLP\n","Accuracy: ", acc_mlp, "\n", "F1 score :", f1_mlp) # f1 = 0.8769064263368993