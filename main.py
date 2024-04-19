import pandas as pd
from sklearn.model_selection import train_test_split

from logistic_regress import logistic_regression
from random_forest import random_classifier
from knn_classifier import knn

def preprocess():
    dataset = pd.read_csv(rf'C:\Users\Aakash\Downloads\dataset.csv')

    dataset.drop(['hash','millisecond','usage_counter','normal_prio','policy','vm_pgoff','task_size','cached_hole_size',
                'hiwater_rss','nr_ptes','lock','cgtime','signal_nvcsw'], axis = 1, inplace = True)

    print(dataset.shape)
    print(dataset.head())

    print(dataset.isna().sum())
    print(dataset.describe())

    classification_num = {'benign': 0, 'malware': 1}
    dataset['classification'] = dataset['classification'].map(classification_num)

    labels = dataset['classification']
    features = dataset.drop('classification', axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.3, random_state = 2000)

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = preprocess()

logistic_regression(x_train, x_test, y_train, y_test)
random_classifier(x_train, x_test, y_train, y_test)
knn(x_train, x_test, y_train, y_test)





