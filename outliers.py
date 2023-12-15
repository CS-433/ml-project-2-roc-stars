import pandas as pd 
from sklearn.model_selection import train_test_split
from helper import *

df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Logistic regression ====================================================================>

# WITH OUTLIERS
logistic(X_train, X_test, y_train, y_test)
# Accuracy:  0.867816091954023 
# F1 score : 0.816

# WITHOUT OUTLIERS
X_train_nout = remove_outliers(X_train)
X_test_nout = remove_outliers(X_test)

logistic(X_train_nout, X_test_nout, y_train, y_test)
# Accuracy:  0.7873563218390804
# F1 score : 0.7040000000000001



    