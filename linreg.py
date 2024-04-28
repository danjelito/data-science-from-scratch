from linealg import Vector, Matrix
from gradient import gradient_step
import random
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

n_samples = 100
n_outliers = 50


xs, ys = make_regression(
    n_samples=n_samples,
    n_features=1,
    n_informative=1,
    noise=10,
    random_state=0,
)


class LinearRegression:
    def __init__(self, lr: float = 0.005, epochs: int = 1000) -> None:
        self.m = 0.0
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def fit(self, xs: Matrix, ys: Vector) -> None:
        self.num_training_samples = len(xs)
        for _ in range(self.epochs):
            dm = self._grad_error_to_m(xs, ys)
            db = self._grad_error_to_b(xs, ys)
            self.m, self.b = gradient_step([self.m, self.b], [dm, db], -self.lr)

    def _grad_error_to_m(self, xs: Matrix, ys: Vector) -> float:
        preds = self.predict(xs)
        errors = [
            self._error_single(y_true, y_pred) for y_true, y_pred in zip(ys, preds)
        ]
        return (
            2
            / self.num_training_samples
            * sum(x * error for x, error in zip(xs, errors))
        )

    def _grad_error_to_b(self, xs: Matrix, ys: Vector) -> float:
        preds = self.predict(xs)
        errors = [
            self._error_single(y_true, y_pred) for y_true, y_pred in zip(ys, preds)
        ]
        return 2 / self.num_training_samples * sum(errors)

    def _error_single(self, y_true: float, y_pred: float) -> float:
        return y_pred - y_true

    def _squared_error_single(self, y_true: float, y_pred: float) -> float:
        return self._error_single(y_true, y_pred) ** 2

    def _mean_squared_error(self, y_trues: Vector, y_preds: Vector) -> float:
        return sum(
            self._squared_error_single(y_true, y_pred)
            for y_true, y_pred in zip(y_trues, y_preds)
        )

    def _predict_single(self, x: Vector) -> float:
        return self.m * x + self.b

    def predict(self, xs: Matrix) -> Vector:
        return [self._predict_single(x) for x in xs]


def flatten(xss):
    return [x for xs in xss for x in xs]


lr = LinearRegression()
lr.fit(xs, ys)
y_preds = lr.predict(xs)

plt.scatter(xs, ys, c="blue")
plt.scatter(xs, y_preds, c="red")
plt.show()
