import numpy as np


def determinant(arr: np.ndarray, j: int = 0) -> float:
    """Laplace expansion along row j

    Args:
        arr (np.ndarray): Array for which to calculate determinant
        j (int, optional): The row to expand along. Defaults to 0.

    Raises:
        ValueError: If number of rows is not equal to number of columns
        ValueError: If j is not between 0 and nrows.

    Returns:
        float: The determinant of the array
    """

    nrows, ncols = arr.shape
    if nrows != ncols:
        raise ValueError(
            f"Cannot calculate determinant for matrix of shape ({nrows},{ncols})."
        )
    if not (0 <= j < nrows):
        raise ValueError(f"The row index to expand along is not in range.")

    if nrows == 2:
        return arr[0, 0] * arr[1, 1] - arr[0, 1] * arr[1, 0]

    s = 0.0
    for k in range(nrows):
        s += (
            (-1) ** (k + j)
            * arr[j, k]
            * determinant(np.delete(np.delete(arr, j, 0), k, 1))
        )

    return s


ex44 = np.array([[0, -1, 1, 1], [-1, 1, -2, 3], [2, -1, 0, 0], [1, -1, 1, 0]])

eigenvalues, eigenvectors = np.linalg.eig(ex44)
print(eigenvalues)

for lam in range(-10, 10):
    m = ex44 - np.eye(ex44.shape[0]) * lam
    det = determinant(m)
    if np.isclose(det, 0):
        print(lam)


# 4.7 b

ex47 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
p = np.array([[1, -1, -1], [1, 1, 0], [1, 0, 1]])
d = np.array([[3, 0, 0], [0, 0, 0], [0, 0, 0]])
print(p @ d @ np.linalg.inv(p))

# eigenvector with 0 eigenvalue
print(ex47 @ np.array([-1, 1, 0]))
print(ex47 @ np.array([-1, 0, 1]))

# 4.7 d

ex47d = np.array([[5, -6, -6], [-1, 4, 2], [3, -6, -4]])
ex47d_p = np.array([[2, 2, 3], [1, 0, -1], [0, 1, 3]])
ex47d_d = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
print(ex47d_p @ ex47d_d @ np.linalg.inv(ex47d_p))
