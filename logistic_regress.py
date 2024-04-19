import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from con_matrix import confusionMatrix

def logistic_regression(x_train, x_test, y_train, y_test):
    model = LogisticRegression()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy Score for Logistic Regression Model is: {accuracy}')

    confusionMatrix(y_test, y_pred, model_name='Logistic Regression')