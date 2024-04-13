from typing import List, Tuple, Optional, Union, Callable
import math

Vector = List[float]
Matrix = List[List[float]]


def add(v: Vector, w: Vector) -> Vector:
    """
    Add corresponding element.

    Example:
    add([1, 2, 3], [4, 5, 6]) => [5, 7, 9]
    """
    assert len(v) == len(w), "Vector must be of the same length."
    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    Subtract corresponding element.

    Example:
    subtract([4, 5, 6], [1, 2, 3]) => [3, 3, 3]
    """
    assert len(v) == len(w), "Vector must be of the same length."
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors=List[Vector]) -> Vector:
    """
    Like add but add arbitrary number of vectors.

    Example:
    vector_sum([[1, 2], [3, 4], [5, 6]]) => [9, 12]
    """
    num_elements = len(vectors[0])
    assert vectors, "No vectors provided."
    assert all(
        num_elements == len(vector) for vector in vectors
    ), "Vectors are of different lengths!"
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def scalar_multiply(c: float | int, v: Vector) -> Vector:
    """
    Multiply every element of v by c.

    Example:
    scalar_multiply(2, [1, 2, 3]) => [2, 4, 6]
    """
    return [c * v_i for v_i in v]


def vector_mean(vectors=List[Vector]) -> Vector:
    """
    Element-wise average.

    Example:
    vector_mean([[1, 2], [3, 4], [5, 6]]) => [3.0, 4.0]
    """
    num_vectors = len(vectors)
    return scalar_multiply(1 / num_vectors, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> int | float:
    """
    Sum of component-wise product.

    Example:
    dot([1, 2, 3], [4, 5, 6]) => 32
    """
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> int | float:
    """
    Dot product of vector with itself.

    Example:
    sum_of_squares([1, 2, 3]) => 14
    """
    return dot(v, v)


def magnitude(v: Vector) -> int | float:
    """
    Squareroot of sum_of_squares. Length of a vector.

    Example:
    magnitude([3, 4]) => 5.0
    """
    return math.sqrt(dot(v, v))


def squared_distance(v: Vector, w: Vector) -> int | float:
    """
    (v_1 - w_1)**2 + ... + (v_n - w_n)**2

    Example:
    squared_distance([1, 2], [4, 6]) => 25
    """
    assert len(v) == len(w), "Vector must be of the same length."
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> int | float:
    """
    sqrt((v_1 - w_1)**2 + ... + (v_n - w_n)**2)

    Example:
    distance([1, 2], [4, 6]) => 5.0
    """
    assert len(v) == len(w), "Vector must be of the same length."
    return math.sqrt(squared_distance(v, w))


def shape(A: Matrix) -> Tuple[int, int]:
    """
    Shape of a matrix.

    Example:
    shape([[1, 2], [3, 4], [5, 6]]) => (3, 2)
    """
    assertion_msg = "Vector inside matrix must be of the same length."
    for row in range(len(A)):
        assert len(A[0]) == len(A[row]), assertion_msg
    num_rows = len(A)
    num_cols = len(A[0])
    return (num_rows, num_cols)


def get_row(A: Matrix, i: int) -> Vector:
    """
    Return i-th row as vector.

    Example:
    get_row([[1, 2], [3, 4], [5, 6]], 1) => [3, 4]
    """
    assertion_msg = "Vector inside matrix must be of the same length."
    for row in range(len(A)):
        assert len(A[0]) == len(A[row]), assertion_msg
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    """
    Return j-th column as vector.

    Example:
    get_column([[1, 2], [3, 4], [5, 6]], 1) => [2, 4, 6]
    """
    assertion_msg = "Vector inside matrix must be of the same length."
    for row in range(len(A)):
        assert len(A[0]) == len(A[row]), assertion_msg
    return [A_i[j] for A_i in A]


def make_matrix(
    num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]
) -> Matrix:
    """
    Returns num_rows x num_cols matrix
    whose (i, j)-th element is entry_fn(i, j).

    Example:
    make_matrix(2, 3, lambda i, j: i * j) => [[0, 0, 0], [0, 1, 2]]
    """
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    """
    Return nxn identity matrix.

    Example:
    identity_matrix(3) => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert add([1, 2], [2, 3]) == [3, 5]
assert subtract([1, 2], [2, 3]) == [-1, -1]
assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]
assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]
assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]
assert dot([1, 2, 3], [4, 5, 6]) == 32
assert sum_of_squares([1, 2, 3]) == 14
assert magnitude([3, 4]) == 5
assert squared_distance([1, 2, 3], [4, 5, 6]) == 27
assert distance([1, 2, 3], [4, 5, 6]) == math.sqrt(27)
assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)
assert get_row([[1, 2, 3], [4, 5, 6]], 0) == [1, 2, 3]
assert get_column([[1, 2, 3], [4, 5, 6]], 0) == [1, 4]
assert make_matrix(3, 3, lambda i, j: 0) == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
assert identity_matrix(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
