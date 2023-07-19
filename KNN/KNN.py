import numpy as np

class KNN:
    def __init__(self, k=3):
        """
        Parameters:
        k (int): The number of neighbors to consider for classification. Default is 3.
        """
        self.k = k
        self.X = None  # Training data features
        self.y = None  # Training data labels

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self, x):
        distances = self.calculate_distances(x)
        dist_idxs = np.argsort(distances)[:self.k]
        knn = np.unique(self.y[dist_idxs], return_counts=True)
        return knn[0][np.argmax(knn[1])]

    def calculate_distances(self, x):
        distances = np.array([x - x_ for x_ in self.X]) ** 2
        return np.array([np.sqrt(sample.sum()) for sample in distances])
