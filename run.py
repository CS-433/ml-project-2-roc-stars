from helper import *
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display average F1 scores
avg_f1_knn, avg_f1_svm, avg_f1_logreg = average_f1_scores(X_train, y_train)
print("Average F1 Score (KNN):", avg_f1_knn)
print("Average F1 Score (SVM):", avg_f1_svm)
print("Average F1 Score (Logistic Regression):", avg_f1_logreg)

