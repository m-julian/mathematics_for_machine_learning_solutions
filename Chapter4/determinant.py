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
        raise ValueError(f"Cannot calculate determinant for matrix of shape ({nrows},{ncols}).")
    if not (0 <= j < nrows):
        raise ValueError(f"The row index to expand along is not in range.")

    if nrows == 2:
        return arr[0, 0] * arr[1, 1] - arr[0, 1] * arr[1, 0]
    
    s = 0.0
    for k in range(nrows):
        s += (-1)**(k+j) * arr[j, k] * determinant(np.delete(np.delete(arr, j, 0), k, 1))

    return s

ex44 = np.array([[0, -1, 1, 1], [-1, 1, -2, 3], [2, -1, 0, 0], [1 ,-1, 1 ,0]])

eigenvalues, eigenvectors = np.linalg.eig(ex44)
print(eigenvalues)

for lam in range(-10, 10):
    m = ex44 - np.eye(ex44.shape[0]) * lam
    det = determinant(m)
    if np.isclose(det, 0):
        print(lam)