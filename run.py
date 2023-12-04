import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
print(df)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic regression without penalty
logreg = LogisticRegression(penalty = None)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('Accuracy: ', acc)
print("F1 Score:", f1)

# Logistic regression with L2 penalty
logreg_l2 = LogisticRegression(penalty = 'l2', max_iter = 500, fit_intercept = True, solver = 'lbfgs')
logreg_l2.fit(X_train, y_train)
y_pred_l2 = logreg_l2.predict(X_test)
acc_l2 = accuracy_score(y_test, y_pred_l2)
f1_l2 = f1_score(y_test, y_pred_l2)
print('Accuracy: ', acc_l2)
print("F1 Score:", f1_l2)


