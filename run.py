import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('data/final_data.csv', sep=";", header=0, index_col=0)
print(df)

X = df.drop(columns=['SURVEY_NAME'])
y = df['SURVEY_NAME']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))

f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)


confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
