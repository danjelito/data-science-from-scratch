from typing import Callable
from linealg import Vector
import linealg


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
