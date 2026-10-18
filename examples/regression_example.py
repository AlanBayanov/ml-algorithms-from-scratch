import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from algorithms.linear_regression import LinearRegression


def main():
    rng = np.random.default_rng(21)

    X = rng.normal(size=(300, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.1, size=300)

    X_train, X_test = X[:240], X[240:]
    y_train, y_test = y[:240], y[240:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression(solver="analytic")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.4f}")


if __name__ == "__main__":
    main()