import numpy as np


class LinearRegression:
    """Linear regression with analytic, GD or mini-batch SGD solvers."""

    def __init__(
        self,
        solver="analytic",
        learning_rate=0.01,
        max_iter=1000,
        batch_size=64,
        l1=0.0,
        l2=0.0,
        tol=1e-6,
        random_state=None,
    ):
        if solver not in {"analytic", "gd", "mini_batch"}:
            raise ValueError("solver must be analytic, gd or mini_batch.")

        if l1 < 0 or l2 < 0:
            raise ValueError(
                "Regularization coefficients must be non-negative."
            )

        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.l1 = l1
        self.l2 = l2
        self.tol = tol
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

        if self.solver == "analytic" and (self.l1 or self.l2):
            raise ValueError(
                "The analytic solver supports only l1=0 and l2=0."
            )

        self.n_features_in_ = X.shape[1]
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0

        if self.solver == "analytic":
            self._fit_analytic(X, y)
        else:
            self._fit_iterative(X, y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "coef_"):
            raise RuntimeError("Call fit before predict.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        return X @ self.coef_ + self.intercept_

    def _fit_analytic(self, X, y):
        X_with_bias = np.c_[np.ones(len(X)), X]
        coefficients = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]

        self.intercept_ = coefficients[0]
        self.coef_ = coefficients[1:]

    def _fit_iterative(self, X, y):
        rng = np.random.default_rng(self.random_state)
        n_samples = len(X)

        for iteration in range(self.max_iter):
            if self.solver == "gd":
                batch_X, batch_y = X, y
            else:
                indices = rng.choice(
                    n_samples,
                    size=min(self.batch_size, n_samples),
                    replace=False,
                )
                batch_X, batch_y = X[indices], y[indices]

            predictions = batch_X @ self.coef_ + self.intercept_
            errors = predictions - batch_y

            gradient_w = (
                batch_X.T @ errors / len(batch_X)
                + self.l1 * np.sign(self.coef_)
                + self.l2 * self.coef_
            )
            gradient_b = errors.mean()

            self.coef_ -= self.learning_rate * gradient_w
            self.intercept_ -= self.learning_rate * gradient_b

            full_predictions = X @ self.coef_ + self.intercept_
            loss = np.mean((full_predictions - y) ** 2) / 2
            loss += self.l1 * np.sum(np.abs(self.coef_))
            loss += self.l2 * np.sum(self.coef_ ** 2) / 2

            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if len(self.loss_history_) > 1:
                improvement = self.loss_history_[-2] - self.loss_history_[-1]

                if improvement < self.tol:
                    break   