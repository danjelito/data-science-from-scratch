from typing import List

import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

import linealg
from linealg import Vector


# create random samples
x, y = make_regression(
    n_samples=100, n_features=1, n_informative=1, noise=20, random_state=42
)
x += 20
y -= 50
inputs = [[x_i, y_i] for x_i, y_i in zip(x.flatten(), y)]


# see the data
# plt.scatter(x, y)
# plt.show()
# print(inputs)


# make each dimension has mean = 0
def de_mean(data: List[Vector]) -> List[Vector]:
    means = linealg.vector_mean(data)
    return [linealg.subtract(v, means) for v in data]


inputs = de_mean(inputs)
assert -0.1 < linealg.vector_mean(inputs)[0] < 0.1
assert -0.1 < linealg.vector_mean(inputs)[1] < 0.1


# see the data
# plt.scatter([x for (x, y) in inputs], [y for (x, y) in inputs])
# plt.show()


def direction(w: Vector) -> Vector:
    magnitude = linealg.magnitude(w)
    return [w_i / magnitude for w_i in w]


def directional_variance(data: List[Vector], w: Vector) -> float:
    """Calculates the variance of x in the direction of w."""
    w_dir = direction(w)
    return sum(linealg.dot(v, w_dir) ** 2 for v in data)


def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
    """The gradient of directional variance with respect to w."""
    w_dir = direction(w)
    return [sum(2 * linealg.dot(v, w_dir) * v[i] for v in data) for i in range(len(w))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Move step_size in the direction of gradient direction from v"""
    assert len(v) == len(gradient), "v and gradient must be of the same size."
    step = linealg.scalar_multiply(step_size, gradient)
    return linealg.add(v, step)


def first_principal_component(data: List[Vector], n: int, step_size: float) -> Vector:
    direction_guess = [1.0 for _ in data[0]]
    for step in range(n):
        dv = directional_variance(data, direction_guess)
        gradient = directional_variance_gradient(data, direction_guess)
        direction_guess = gradient_step(direction_guess, gradient, -step_size)

    return direction(direction_guess)


def project(v: Vector, w: Vector) -> Vector:
    """Return the projection of v onto the direction of w."""
    projection_length = linealg.dot(v, w)
    return linealg.scalar_multiply(projection_length, w)


def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
    return linealg.subtract(v, project(v, w))


def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
    return [remove_projection_from_vector(v, w) for v in data]


# from scratch
direction_guess = first_principal_component(inputs, 100, 0.1)
inputs_projected_scratch = remove_projection(inputs, direction_guess)


# see the data
# plt.scatter([x for (x, y) in inputs], [y for (x, y) in inputs])
# plt.scatter(
#     [x for (x, y) in inputs_projected_scratch],
#     [y for (x, y) in inputs_projected_scratch],
#     color="blue",
# )
# plt.arrow(
#     0,
#     0,
#     direction_guess[0] * 100,
#     direction_guess[1] * 100,
#     color="red",
# )
# plt.show()
