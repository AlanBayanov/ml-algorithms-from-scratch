import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes classifier."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

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

        self.class_prior_ = np.zeros(len(self.classes_))
        self.theta_ = np.zeros((len(self.classes_), X.shape[1]))
        self.var_ = np.zeros_like(self.theta_)

        for class_index, class_value in enumerate(self.classes_):
            class_X = X[y == class_value]

            self.class_prior_[class_index] = len(class_X) / len(X)
            self.theta_[class_index] = class_X.mean(axis=0)
            self.var_[class_index] = class_X.var(axis=0)

        self.var_ += (
            self.var_smoothing * self.var_.max(axis=1, keepdims=True)
        )

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)

        if not hasattr(self, "class_prior_"):
            raise RuntimeError("Call fit before predict_proba.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features.")

        log_probabilities = []

        for class_index in range(len(self.classes_)):
            log_prior = np.log(self.class_prior_[class_index])
            log_likelihood = self._log_gaussian_pdf(
                X,
                self.theta_[class_index],
                self.var_[class_index],
            ).sum(axis=1)

            log_probabilities.append(log_prior + log_likelihood)

        log_probabilities = np.array(log_probabilities).T
        log_probabilities -= log_probabilities.max(axis=1, keepdims=True)

        probabilities = np.exp(log_probabilities)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    @staticmethod
    def _log_gaussian_pdf(X, mean, variance):
        return (
            -0.5 * np.log(2 * np.pi * variance)
            - (X - mean) ** 2 / (2 * variance)
        )