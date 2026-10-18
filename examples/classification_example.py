import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from algorithms.gradient_boosting import GradientBoostingClassifier


def main():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        weights=[0.8, 0.2],
        random_state=21,
    )

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=21,
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")


if __name__ == "__main__":
    main()