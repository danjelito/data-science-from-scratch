from typing import Callable
from linealg import Vector
import linealg
import random
import matplotlib.pyplot as plt

random.seed(0)


def difference_quotient(f: Callable[[float], float], x: float, h: float = 0.00001):
    """
    Calculates the estimated derivative of function f at point x using finite difference approximation.

    Parameters:
    f (Callable[[float], float]): The function to differentiate.
    x (float): The point at which to estimate the derivative.
    h (float, optional): The step size for the finite difference approximation. Default is 0.00001.

    Returns:
    float: The estimated derivative of f at x.
    """
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(
    f: Callable[[Vector], float], v: Vector, i: float, h: float = 0.00001
):
    """
    Calculates the estimated partial derivative of function f with respect to v[i],
    where v is a vector of values.

    Parameters:
    f (Callable[[Vector], float]): The function to differentiate.
    v (Vector): The vector of values at which to estimate the partial derivative.
    i (int): The index of the element in v with respect to which to calculate the partial derivative.
    h (float, optional): The step size for the finite difference approximation. Default is 0.00001.

    Returns:
    float: The estimated partial derivative of f with respect to v[i].
    """
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float], v: Vector, h: float = 0.00001):
    """
    Calculates the estimated gradient of function f at point v using finite difference approximation.

    Parameters:
    f (Callable[[Vector], float]): The function to differentiate.
    v (Vector): The point at which to estimate the gradient.
    h (float, optional): The step size for the finite difference approximation. Default is 0.00001.

    Returns:
    Vector: The estimated gradient of f at v.
    """
    return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def assertion():

    def square(x: int | float):
        return x * x

    derivative_square = difference_quotient(square, 5)
    assert round(derivative_square) == 10

    def arbitrary_function(v: Vector):
        x, y, z = v
        return (4 * x**3 * y * z**2) + (2 * x * z**3) + (x * y * z)

    derivative_arbitrary_function = estimate_gradient(arbitrary_function, [1, -1, -1])
    assert [round(x) for x in derivative_arbitrary_function] == [-13, 3, 13]


assertion()


if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # derivative of a univariable function
    def square(x: int | float):
        return x * x

    derivative_square = difference_quotient(square, 5)
    print(f"The slope of x ** 2 at x = 5 is {derivative_square: .3f}")

    # ---------------------------------------------------------------------
    # partial derivative of multivarialble function f with respect to x
    def arbitrary_function(v: Vector):
        x, y, z = v
        return (4 * x**3 * y * z**2) + (2 * x * z**3) + (x * y * z)

    f = "4x3yz2+2xz3+xyz"
    sample_v = [1, -1, -1]
    partial_der_to_x = partial_difference_quotient(
        f=arbitrary_function, v=sample_v, i=0
    )
    print(
        f"The partial derivative of {f} with respect to x at {sample_v} is {round(partial_der_to_x, 6)}"
    )

    # ---------------------------------------------------------------------
    # gradient of a multivariable function
    derivative_arbitrary_function = estimate_gradient(arbitrary_function, [1, -1, -1])
    print(
        f"The slope of {f} at {sample_v} is {[round(x, 6) for x in derivative_arbitrary_function]}"
    )

    # ---------------------------------------------------------------------
    # using gradient, let's find a point where sum_of_squares function has the smallest output
    def sum_of_squares_gradient(v: Vector) -> Vector:
        return [2 * v_i for v_i in v]

    def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
        """Move step_size in the direction of gradient direction from v"""
        assert len(v) == len(gradient), "v and gradient must be of the same size."
        step = linealg.scalar_multiply(step_size, gradient)
        return linealg.add(v, step)

    # pick a random starting point
    v = [random.uniform(-10, 10) for _ in range(3)]
    epochs = 1000
    # step size should be negative because we are moving away from the increase
    # play around with this to see the effect of step size
    step_size = -0.005
    for epoch in range(epochs):
        gradient = sum_of_squares_gradient(v)
        v = gradient_step(v, gradient, step_size)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, v = {v}")
    print(f"Distance to [0, 0, 0] == {linealg.distance(v, [0, 0, 0]): .8f}")

    # ---------------------------------------------------------------------
    # using gradient, let's fit a model
    def f(x) -> tuple:
        return x, 20 * x + 5

    inputs = [f(x) for x in range(-50, 50)]  # [(x1, y1), (x2, y2), ...]
    # print(inputs)
    # plt.scatter(x=[x for (x, y) in inputs], y=[y for (x, y) in inputs])
    # plt.show()

    def linear_gradient(x: float, y: float, theta: Vector):

        def calculate_error(y_true, y_pred):
            """Calculates error of the prediction."""
            return y_true - y_pred

        def calculate_squared_error(y_true, y_pred):
            """Calculates squared error of the prediction."""
            return calculate_error(y_true, y_pred) ** 2

        def calculate_squared_error_derivative(x, error):
            """
            Partial derivative of squared error
            with respect to slope and intercept.
            """
            # return [der loss to slope, der loss to intercept]
            return [2 * error * x, 2 * error]

        slope, intercept = theta
        predicted = slope * x + intercept # create prediction using linear eq
        error = calculate_error(predicted, y)
        squared_error = calculate_squared_error(predicted, y)
        gradient = calculate_squared_error_derivative(x, error)
        return gradient

    # start with random theta
    theta = [random.uniform(-1, 1) for _ in range(2)]
    learning_rate = 0.001
    epochs = 5_000
    for epoch in range(epochs):
        # compute the mean of gradients accross all data points
        gradients = linealg.vector_mean([linear_gradient(x, y, theta) for (x, y) in inputs])
        # take a step in the opposite direction
        theta = gradient_step(theta, gradients, - learning_rate)
        # print log
        if (epoch % 1000 == 0) or (epoch == epochs - 1):
            print(f"Epoch {epoch}, (slope, intercept) = {theta}")

    slope, intercept = theta
    assert 19.9 < slope < 20.1, "Slope should be around 20."
    assert 4.9 < intercept < 5.1, "Intercept should be around 5."
























