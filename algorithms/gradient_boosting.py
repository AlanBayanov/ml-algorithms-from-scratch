import numpy as np

from algorithms.decision_tree import DecisionTreeRegressor


class GradientBoostingClassifier:
    """Binary gradient boosting classifier based on regression trees."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)

        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.classes_ = np.unique(y)

        if not np.array_equal(self.classes_, [0, 1]):
            raise ValueError(
                "GradientBoostingClassifier supports binary labels 0 and 1."
            )

        self.n_features_in_ = X.shape[1]

        positive_rate = np.clip(y.mean(), 1e-15, 1 - 1e-15)
        self.initial_prediction_ = np.log(
            positive_rate / (1 - positive_rate)
        )

        current_prediction = np.full(
            len(y),
            self.initial_prediction_,
            dtype=float,
        )
        self.estimators_ = []
        self.loss_history_ = []

        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.n_estimators):
            probabilities = self._sigmoid(current_prediction)
            residuals = y - probabilities

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.integers(0, 1_000_000),
            )
            tree.fit(X, residuals)

            current_prediction += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)

            probabilities = self._sigmoid(current_prediction)
            epsilon = 1e-15
            clipped = np.clip(probabilities, epsilon, 1 - epsilon)

            loss = -np.mean(
                y * np.log(clipped)
                + (1 - y) * np.log(1 - clipped)
            )
            self.loss_history_.append(loss)

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit before predict_proba.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        prediction = np.full(
            len(X),
            self.initial_prediction_,
            dtype=float,
        )

        for tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(X)

        positive_probability = self._sigmoid(prediction)
        return np.c_[1 - positive_probability, positive_probability]

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]