import numpy as np

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, min_gain=0.01):
        """
        max_depth: maximum depth allowed for the tree
        min_samples_split: minimum samples required to further split a node
        min_gain: minimum information gain required to accept a split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.tree = None   # this will store the final trained tree

    #entropy to measure impurity 
    def _entropy(self, p):
        """
        p = probability of class 1 at a node
        entropy = 0 means node is pure (all 0 or all 1)
        """
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    # splits data sample according to the feature
    def _split_indices(self, X, feature):
        """
        For a given feature index:
        - left  : samples where feature == 1
        - right : samples where feature == 0
        """
        left, right = [], []

        for idx, x in enumerate(X):
            if x[feature] == 1:
                left.append(idx)
            else:
                right.append(idx)

        return left, right

    #calculated weighted entropy
    def _weighted_entropy(self, X, y, left, right):
        """
        Calculates how impure the children nodes are
        after a split, weighted by their sizes.
        """
        n = len(X)

        # entropy of left child
        if len(left) == 0:
            H_left = 0
        else:
            p_left = np.mean(y[left])
            H_left = self._entropy(p_left)

        # entropy of right child
        if len(right) == 0:
            H_right = 0
        else:
            p_right = np.mean(y[right])
            H_right = self._entropy(p_right)

        # weighted sum
        return (len(left) / n) * H_left + (len(right) / n) * H_right

    #information gain
    def _information_gain(self, X, y, left, right):
        """
        Measures how much a split reduces impurity.
        Higher gain = better split.
        """
        p_node = np.mean(y)  # impurity before split
        return self._entropy(p_node) - self._weighted_entropy(X, y, left, right)

    # finds the best feature to split on
    def _best_split(self, X, y, features):
        """
        Try every feature and pick the one
        with the highest information gain.
        """
        best_gain = -1
        best_feature = None
        best_split = None

        for feature in features:
            left, right = self._split_indices(X, feature)

            # ignore useless splits (all data on one side)
            if len(left) == 0 or len(right) == 0:
                continue

            gain = self._information_gain(X, y, left, right)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_split = (left, right)

        return best_feature, best_split, best_gain

    # a recurive function to build decision tree
    def _build_tree(self, X, y, indices, features, depth):
        """
        Core recursive function that builds the tree.
        """
        y_node = y[indices]

        # stopping conditions 
        if (
            depth == self.max_depth or              # reached max depth
            len(indices) < self.min_samples_split or # too few samples
            len(set(y_node)) == 1                   # node already pure
        ):
            # make a leaf node
            return {
                "type": "leaf",
                "prediction": int(np.round(np.mean(y_node)))
            }

        # data belonging to current node
        X_node = X[indices]

        best_feature, split, gain = self._best_split(X_node, y_node, features)

        # if no meaningful split exists, stop
        if best_feature is None or gain < self.min_gain:
            return {
                "type": "leaf",
                "prediction": int(np.round(np.mean(y_node)))
            }

        # map local indices back to original indices
        left_local, right_local = split
        left_indices = [indices[i] for i in left_local]
        right_indices = [indices[i] for i in right_local]

        # create decision node
        return {
            "type": "node",
            "feature": best_feature,
            "left": self._build_tree(X, y, left_indices, features, depth + 1),
            "right": self._build_tree(X, y, right_indices, features, depth + 1)
        }

    #train the model
    def fit(self, X, y):
        """
        Entry point for training.
        """
        features = list(range(X.shape[1]))
        indices = list(range(len(X)))
        self.tree = self._build_tree(X, y, indices, features, depth=0)

    #predict result for single sample
    def _predict_one(self, x, tree):
        """
        Traverse the tree until a leaf is reached.
        """
        if tree["type"] == "leaf":
            return tree["prediction"]

        if x[tree["feature"]] == 1:
            return self._predict_one(x, tree["left"])
        else:
            return self._predict_one(x, tree["right"])

    #predict result for multiple samples
    def predict(self, X):
        """
        Predict labels for all samples.
        """
        return np.array([self._predict_one(x, self.tree) for x in X])

    #accuracy score
    def accuracy(self, X, y):
        """
        Percentage of correct predictions.
        """
        return np.mean(self.predict(X) == y) * 100

    # visualize tree (text form)
    def visualize(self, tree=None, depth=0):
        """
        Prints the tree structure in a readable way.
        """
        if tree is None:
            tree = self.tree

        indent = "   " * depth

        if tree["type"] == "leaf":
            print(f"{indent}Leaf → predict {tree['prediction']}")
        else:
            print(f"{indent}Split on feature {tree['feature']}")
            print(f"{indent}Left:")
            self.visualize(tree["left"], depth + 1)
            print(f"{indent}Right:")
            self.visualize(tree["right"], depth + 1)
