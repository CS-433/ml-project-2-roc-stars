import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# LOGISTIC REGRESSION
logreg = LogisticRegression(C = 1, penalty='l2')
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1score_lr = f1_score(y_test, y_pred_lr)
print("LR\n","Accuracy: ", accuracy_lr, "\n", "F1 score :", f1score_lr)
# Accuracy:  0.867816091954023 
# F1 score : 0.816


# K NEAREST NEIGHBORS
knn = KNeighborsClassifier(metric='manhattan', n_neighbors=9)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1score_knn = f1_score(y_test, y_pred_knn)
print("KNN\n","Accuracy: ", accuracy_knn, "\n", "F1 score :", f1score_knn)
# Accuracy:  0.8103448275862069 
# F1 score : 0.7027027027027027


# SUPPORT VECTOR MACHINE
svm = SVC(C=1, gamma='auto', kernel='poly')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
f1score_svm = f1_score(y_test, y_pred_svm)
print("SVM\n","Accuracy: ", accuracy_svm, "\n", "F1 score :", f1score_svm)
#Accuracy:  0.8390804597701149 
# F1 score : 0.7741935483870969


# MUTILAYER PERCEPTRON
mlp = MLPClassifier(hidden_layer_sizes=(55, 55), max_iter=1000, random_state=42, beta_2=0.99, beta_1=0, solver='adam', activation='relu', batch_size=256, momentum=0.99)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1score_mlp = f1_score(y_test, y_pred_mlp)
print("MLP\n","Accuracy: ", accuracy_mlp, "\n", "F1 score :", f1score_mlp)
# Accuracy:  0.8793103448275862 
# F1 score : 0.8292682926829269


# GAUSSIAN MATRIX MODEL
gmm = GaussianMixture(covariance_type='diag', n_components=2)
gmm.fit(X_train, y_train)
y_pred_gmm = gmm.predict(X_test)
accuracy_gmm = accuracy_score(y_test, y_pred_gmm)
f1score_gmm = f1_score(y_test, y_pred_gmm)
print("GMM\n","Accuracy: ", accuracy_gmm, "\n", "F1 score :", f1score_gmm)
# Accuracy:  0.7988505747126436 
# F1 score : 0.6846846846846846