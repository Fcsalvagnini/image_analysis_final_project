from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import numpy as np


def sort_index(l):
    return sorted(range(len(l)), key=lambda k: l[k])


def avaliable_KNN(list_features_valid, labels):

    x_test = []
    y_test = []
    index_list = []

    neigh = KNeighborsClassifier(n_neighbors=1)
    list_features_valid = list_features_valid.numpy()

    labels = list(labels)

    index_sort = sort_index(labels)

    X = [list_features_valid[x, :] for x in index_sort]
    Y = [labels[x] for x in index_sort]
    #train and test
    Y = [[Y[3*i], Y[3*i + 1], Y[3*i + 2]] for i in range(len(Y)//3)]
    X = [[X[3*i], X[3*i + 1], X[3*i + 2]] for i in range(len(X)//3)]

    for i in range(len(Y)):
        y_test.append(Y[i].pop(2))
        x_test.append(X[i].pop(2))

    X = sum(X, [])
    Y = sum(Y, [])

    x_train = X
    y_train = Y

    neigh.fit(x_train, y_train)

    y_pred = neigh.predict(x_test)
    y_true = y_test

    return accuracy_score(y_true, y_pred)
