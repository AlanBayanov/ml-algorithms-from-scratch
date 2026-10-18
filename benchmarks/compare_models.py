import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from algorithms import (
    DecisionTreeClassifier,
    GaussianNB,
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
)

RANDOM_STATE = 21
RESULTS_PATH = Path("benchmarks/results/classification_benchmark.csv")


def main():
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.8, 0.2],
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []

    for name, model in models.items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        probabilities = model.predict_proba(X_test)[:, 1]
        predict_time = time.perf_counter() - start

        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(
                y_test,
                probabilities,
            ),
            "fit_seconds": fit_time,
            "predict_seconds": predict_time,
        })

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    print(results.sort_values("roc_auc", ascending=False))


if __name__ == "__main__":
    main()