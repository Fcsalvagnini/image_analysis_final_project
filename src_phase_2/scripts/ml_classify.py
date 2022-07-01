from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np

def avaliable_KNN(list_features_valid, labels):
    neigh = KNeighborsClassifier(n_neighbors=1)
    print(labels)
    raise('pause')
    neigh.fit(X, y)