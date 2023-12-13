import pandas as pd 
from helper import *
from sklearn.model_selection import train_test_split
# Import Models
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
# Import Evaluation methods
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic Regression

#Create the model
logreg_model = LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred_logreg))

f1score = f1_score(y_test, y_pred_logreg)
print("Logistic Regression f1 score: ", f1score) # 0.816

#Gaussian Mixture Matrix

# Create the model
gmm_model = GaussianMixture(n_components=2, random_state=42)

# TRain the model
gmm_model.fit(X_train)

# Make predictions
y_pred_gmm = gmm_model.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred_gmm))

f1score = f1_score(y_test, y_pred_gmm)
print("GMM f1 score: ", f1score) # 0.14184397163120568


