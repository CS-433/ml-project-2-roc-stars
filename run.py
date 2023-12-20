# < --------------------------------Import Libraries------------------------------------- >
from helper import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# < -----------------------------------Load Data----------------------------------------- >
# Load dataset
df = pd.read_csv('Datasets/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < -------------------------------Logistic Regression------------------------------------ >
logistic_regression = LogisticRegression(random_state=42, penalty='l2', C=1.0, max_iter=2000)
logistic_regression.fit(X_train, y_train)
model_performance(logistic_regression, X_test, y_test, CV=False)