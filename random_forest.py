import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from con_matrix import confusionMatrix

def random_classifier(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=20, min_samples_leaf=3)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'Accuracy Score for Randome Forest Classifier Model is: {accuracy}')

    confusionMatrix(y_test, y_pred, model_name='Random Forest Classifier')