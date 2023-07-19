import numpy as np
from DecisionTree import DecisionTree # Make sure to import it correctly

# Defining the RandomForest class
class RandomForest:
    def __init__(self, max_depth=10, number_of_features_to_use=None, min_samples_split=2, number_of_trees=10):
        self.max_depth = max_depth  # Maximum depth of each decision tree in the forest
        self.n_features = number_of_features_to_use  # Number of features to be used at each split in a tree
        self.min_samples = min_samples_split  # Minimum number of samples required to split a node in a tree

        self.trees = []  # List to store the individual DecisionTree objects in the forest
        self.n_trees = number_of_trees  # Number of decision trees in the RandomForest

    def fit(self, X, y):

        n_dim, _ = X.shape  # Getting the number of samples

        # Creating the specified number of decision trees and adding them to the RandomForest
        for _ in range(self.n_trees):
            # Randomly selecting rows from the input data with replacement (bootstrap sampling)
            new_idx = np.random.choice(n_dim, n_dim, replace=True)
            new_X, new_y = X[new_idx, :], y[new_idx]

            # Creating a new DecisionTree object with specified hyperparameters and training it on the new data
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples,
                                n_features_to_use=self.n_features)
            tree.fit(new_X, new_y)  # Training the decision tree on the bootstrap sampled data
            self.trees.append(tree)  # Adding the trained tree to the list of trees in the RandomForest

    def predict(self, X):

        # Making predictions for each tree in the RandomForest and storing them in an array
        preds = np.array([tree.predict(X) for tree in self.trees])

        # Combining the predictions of individual trees to get the final prediction for each data point
        # by choosing the most frequent prediction among all trees for each data point
        preds = np.array([np.unique(pred, return_counts=True) for pred in preds.T], dtype=object)
        return np.array([pred[0][np.argmax(pred[1])] for pred in preds])
