import numpy as np

from algorithms.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Random forest classifier based on custom decision trees."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.estimators_ = []

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            bootstrap_indices = rng.choice(
                len(X),
                size=len(X),
                replace=True,
            )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=rng.integers(0, 1_000_000),
            )

            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.estimators_.append(tree)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict_proba.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        probabilities = np.zeros((len(X), len(self.classes_)))

        for tree in self.estimators_:
            probabilities += self._align_tree_probabilities(tree, X)

        return probabilities / len(self.estimators_)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _align_tree_probabilities(self, tree, X):
        tree_probabilities = tree.predict_proba(X)
        aligned = np.zeros((len(X), len(self.classes_)))

        for tree_class_index, tree_class in enumerate(tree.classes_):
            class_index = np.where(self.classes_ == tree_class)[0][0]
            aligned[:, class_index] += tree_probabilities[:, tree_class_index]

        return aligned