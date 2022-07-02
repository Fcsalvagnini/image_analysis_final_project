from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random
import numpy as np


def sort_index(l):
    return sorted(range(len(l)), key=lambda k: l[k])


def avaliable_KNN(list_features_valid, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)

    list_features_valid = list_features_valid.numpy()

    labels = list(labels)

    index_sort = sort_index(labels)

    X = [list_features_valid[x, :] for x in index_sort]
    Y = [labels[x] for x in index_sort]

    length = len(Y)
    slice = 0.8
    slice_train = int(length * slice)
    slice_test = int(length * (slice - 1))

    x_train = X[:slice_train]
    y_train = Y[:slice_train]

    x_test = X[slice_train:]
    y_test = Y[slice_train:]

    neigh.fit(x_train, y_train)

    y_pred = neigh.predict(x_test)
    y_true = y_test

    return accuracy_score(y_true, y_pred)
