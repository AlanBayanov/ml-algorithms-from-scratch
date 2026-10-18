import numpy as np


class _Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        probabilities=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probabilities = probabilities

    @property
    def is_leaf(self):
        return self.feature is None


class DecisionTreeClassifier:
    """Decision tree classifier with Gini or entropy criterion."""

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be gini or entropy.")

        self.criterion = criterion
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
        self.root_ = self._build_tree(X, y, depth=0)

        return self

    def predict_proba(self, X):
        X = self._check_predict_input(X)

        return np.array([
            self._find_leaf(sample).probabilities
            for sample in X
        ])

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _build_tree(self, X, y, depth):
        if (
            self.max_depth is not None
            and depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self._make_leaf(y)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return self._make_leaf(y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return self._make_leaf(y)

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _best_split(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        feature_indices = self._get_feature_indices(
            rng,
            n_features,
        )

        best_feature = None
        best_threshold = None
        best_loss = np.inf

        for feature in feature_indices:
            unique_values = np.unique(X[:, feature])

            if len(unique_values) < 2:
                continue

            thresholds = (
                unique_values[:-1] + unique_values[1:]
            ) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                if self.criterion == "gini":
                    loss = (
                        len(y_left) * self._gini(y_left)
                        + len(y_right) * self._gini(y_right)
                    ) / n_samples
                else:
                    loss = (
                        len(y_left) * self._entropy(y_left)
                        + len(y_right) * self._entropy(y_right)
                    ) / n_samples

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _make_leaf(self, y):
        counts = np.array([
            np.sum(y == class_value)
            for class_value in self.classes_
        ])

        probabilities = counts / counts.sum()
        value = self.classes_[np.argmax(probabilities)]

        return _Node(value=value, probabilities=probabilities)

    def _find_leaf(self, sample):
        node = self.root_

        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node

    def _get_feature_indices(self, rng, n_features):
        if self.max_features is None:
            return range(n_features)

        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            size = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            size = min(self.max_features, n_features)
        else:
            raise ValueError("Unsupported max_features value.")

        return rng.choice(
            n_features,
            size=size,
            replace=False,
        )

    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return 1.0 - np.sum(probabilities ** 2)

    @staticmethod
    def _entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        return -np.sum(probabilities * np.log2(probabilities))

    def _check_predict_input(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        return X


class DecisionTreeRegressor:
    """Decision tree regressor with MSE splitting criterion."""

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional array.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.n_features_in_ = X.shape[1]
        self.root_ = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit before prediction.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        return np.array([
            self._predict_sample(sample, self.root_)
            for sample in X
        ])

    def _build_tree(self, X, y, depth):
        if (
            self.max_depth is not None
            and depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return _Node(value=np.mean(y))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return _Node(value=np.mean(y))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if (
            left_mask.sum() < self.min_samples_leaf
            or right_mask.sum() < self.min_samples_leaf
        ):
            return _Node(value=np.mean(y))

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return _Node(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
        )

    def _best_split(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        feature_indices = self._get_feature_indices(
            rng,
            n_features,
        )

        best_feature = None
        best_threshold = None
        best_loss = np.inf

        for feature in feature_indices:
            unique_values = np.unique(X[:, feature])

            if len(unique_values) < 2:
                continue

            thresholds = (
                unique_values[:-1] + unique_values[1:]
            ) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                loss = (
                    len(y_left) * self._mse(y_left)
                    + len(y_right) * self._mse(y_right)
                ) / n_samples

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _predict_sample(self, sample, node):
        while not node.is_leaf:
            if sample[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.value

    def _get_feature_indices(self, rng, n_features):
        if self.max_features is None:
            return range(n_features)

        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            size = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            size = min(self.max_features, n_features)
        else:
            raise ValueError("Unsupported max_features value.")

        return rng.choice(
            n_features,
            size=size,
            replace=False,
        )

    @staticmethod
    def _mse(y):
        return np.mean((y - np.mean(y)) ** 2)