import numpy as np


class LogisticRegression:
    """Binary logistic regression trained with mini-batch SGD."""

    def __init__(
        self,
        learning_rate=0.01,
        max_iter=1000,
        batch_size=64,
        l2=0.0,
        tol=1e-6,
        random_state=None,
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l2 = l2
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _sigmoid(z):
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)

        positive = z >= 0
        result[positive] = 1.0 / (1.0 + np.exp(-z[positive]))

        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1.0 + exp_z)

        return result

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

        if len(self.classes_) != 2:
            raise ValueError(
                "LogisticRegression supports exactly two classes."
            )

        self.n_features_in_ = X.shape[1]
        y_binary = (y == self.classes_[1]).astype(float)

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0

        rng = np.random.default_rng(self.random_state)

        for iteration in range(self.max_iter):
            indices = rng.choice(
                len(X),
                size=min(self.batch_size, len(X)),
                replace=False,
            )
            batch_X = X[indices]
            batch_y = y_binary[indices]

            probabilities = self._sigmoid(
                batch_X @ self.coef_ + self.intercept_
            )
            errors = probabilities - batch_y

            gradient_w = (
                batch_X.T @ errors / len(batch_X)
                + self.l2 * self.coef_
            )
            gradient_b = errors.mean()

            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

            full_probabilities = self._sigmoid(
                X @ self.coef_ + self.intercept_
            )
            epsilon = 1e-15
            clipped = np.clip(full_probabilities, epsilon, 1 - epsilon)

            loss = -np.mean(
                y_binary * np.log(clipped)
                + (1 - y_binary) * np.log(1 - clipped)
            )
            loss += self.l2 * np.sum(self.coef_ ** 2) / 2

            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if len(self.loss_history_) > 1:
                if self.loss_history_[-2] - self.loss_history_[-1] < self.tol:
                    break

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict_proba.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        positive_probability = self._sigmoid(
            X @ self.coef_ + self.intercept_
        )
        return np.c_[1 - positive_probability, positive_probability]

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]