# < -------------------------------Import libraries-------------------------------------- >
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from helper import *
from sklearn.svm import SVC

# < -----------------------------------Load data----------------------------------------- >
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# < -------------------------------Logistic Regression----------------------------------- >
# < --------------------------------WITHOUT OUTLIERS------------------------------------- >
X_train_nout = remove_outliers(X_train)
X_test_nout = remove_outliers(X_test)
logistic(X_train_nout, X_test_nout, y_train, y_test)
