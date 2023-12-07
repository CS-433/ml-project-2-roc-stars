from helper import *
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split #ptr que ce step on pourra le faire dans le datacleaning

# Import tuning models
from sklearn.model_selection import GridSearchCV

# Import scoring methods
from sklearn.metrics import classification_report



# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Display average F1 scores
avg_f1_knn, avg_f1_svm, avg_f1_logreg = average_f1_scores(X_train, y_train)
print("Average F1 Score (KNN):", avg_f1_knn) #0.7929342986844565
print("Average F1 Score (SVM):", avg_f1_svm) # 0.8239125211396205
print("Average F1 Score (Logistic Regression):", avg_f1_logreg) # 0.8654894678589372


# Model Performance

#Create the model
logreg_model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred = logreg_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
