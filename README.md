# ML Algorithms from Scratch

Учебный репозиторий с реализациями классических алгоритмов машинного
обучения на NumPy. Основная цель — разобраться в внутреннем устройстве
алгоритмов и сравнить собственные реализации со scikit-learn.

## Реализованные алгоритмы

- Linear Regression: аналитическое решение, градиентный спуск, mini-batch SGD
- L1/L2-регуляризация
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree Classifier и Regressor
- Random Forest Classifier
- Gradient Boosting Classifier

## Стек

- Python
- NumPy
- Pandas
- scikit-learn — для сравнения и метрик
- Matplotlib — для визуализаций
- Jupyter Notebook — для демонстраций

## Установка

```bash
git clone [https://github.com/AlanBayanov/ml-algorithms-from-scratch.git](https://github.com/AlanBayanov/ml-algorithms-from-scratch.git)
cd ml-algorithms-from-scratch

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\Activate.ps1  # Windows PowerShell

python -m pip install -r requirements.txt
```

## Запуск примеров

```bash
python examples/regression_example.py
python examples/classification_example.py
```

## Запуск сравнения моделей

```bash
python benchmarks/compare_models.py
```

## Пример использования

```python
from algorithms.linear_regression import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression(solver="analytic")
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
```

## Структура репозитория

```text
algorithms/       # реализации ML-алгоритмов
examples/         # короткие примеры запуска
benchmarks/       # сравнение моделей
notebooks/        # демонстрации и визуализации
```

## Ограничения

- Поддерживаются плотные числовые массивы без пропусков.
- Logistic Regression и Gradient Boosting — бинарная классификация.
- Масштабирование и метрики используются из scikit-learn.
- Реализации учебные и не оптимизированы по скорости.
- Проект не является заменой scikit-learn.

## Происхождение кода

Реализации созданы в ходе обучения и доработаны для этого репозитория.