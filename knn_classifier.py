import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from con_matrix import confusionMatrix

def knn(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier(weights='uniform', algorithm='auto', leaf_size=10)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy Score for KNN Classifier Model is: {accuracy}')

    confusionMatrix(y_test, y_pred, model_name='KNN Classifier')