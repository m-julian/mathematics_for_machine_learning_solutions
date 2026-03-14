from typing import Callable
import numpy as np
from itertools import pairwise
from functools import partial


def numerical_integration_1d(
    func: Callable, lower_bound: float, upper_bound: float, n: int = 1000
) -> float:
    total = 0.0
    x = np.linspace(lower_bound, upper_bound, n)
    for x1, x2 in pairwise(x):
        func_val = func(x1)
        dx = x2 - x1
        total += func_val * dx
    return total


def numerical_integration_2d(
    func: Callable,
    lb_x1: float,
    ub_x1: float,
    lb_x2: float,
    ub_x2: float,
    n1: int = 300,
    n2: int = 300,
) -> float:

    total = 0.0
    x1 = np.linspace(lb_x1, ub_x1, n1)
    x2 = np.linspace(lb_x2, ub_x2, n2)

    for x11, x12 in pairwise(x1):
        for x21, x22 in pairwise(x2):
            midpoint_x1 = (x11 + x12) / 2
            midpoint_x2 = (x21 + x22) / 2
            dx1 = x12 - x11
            dx2 = x22 - x21
            func_val = func(np.array([midpoint_x1, midpoint_x2]))
            total += func_val * dx1 * dx2

    return total


def gaussian_density_1d(x: float, mu: float, sigma: float):

    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def gaussian_density_2d(
    x_vec: np.ndarray, mu_vec: float, covar_matrix: np.ndarray
) -> float:

    ndim = len(x_vec)
    covar_matrix_determinant = np.linalg.det(covar_matrix)
    covar_matrix_inv = np.linalg.inv(covar_matrix)

    return (
        (2 * np.pi) ** (-ndim / 2)
        * covar_matrix_determinant**-0.5
        * np.exp(
            -0.5
            * (x_vec - mu_vec).reshape(1, -1)
            @ covar_matrix_inv
            @ (x_vec - mu_vec).reshape(-1, 1)
        ).item()
    )


def find_median_1d(func: Callable, lb=-20, dx=1e-03) -> float:

    total = 0.0
    counter = 0
    while total < 0.5:
        func_val = func(lb + counter * dx)
        total += func_val * dx
        counter += 1
    return lb + counter * dx


def find_modes_1d(func: Callable, lb=-1, ub=15, n=1000000) -> list[float]:

    modes = []
    lins = np.linspace(lb, ub, n)

    prev = np.inf
    found_mode = False

    for x in lins:
        current = func(x)
        if current < prev and found_mode:
            found_mode = False
            modes.append(x)
        elif current > prev and not found_mode:
            found_mode = True

        prev = current

    return modes


def find_modes_2d(func: Callable, lb=-5, ub=15, n=500) -> list[float]:

    # could remove redundant calculations here

    modes = []
    lins1 = np.linspace(lb, ub, n)
    lins2 = np.linspace(lb, ub, n)

    dx = (ub - lb) / (n - 1)

    for x1 in lins1:
        for x2 in lins2:
            current = func(np.array([x1, x2]))
            eval_minusx1 = func(np.array([x1 - dx, x2]))
            eval_plusx1 = func(np.array([x1 + dx, x2]))
            eval_minusx2 = func(np.array([x1, x2 - dx]))
            eval_plusx2 = func(np.array([x1, x2 + dx]))
            eval_minusx1_minusx2 = func(np.array([x1 - dx, x2 - dx]))
            eval_plusx1_plusx2 = func(np.array([x1 + dx, x2 + dx]))
            eval_minusx1_plusx2 = func(np.array([x1 - dx, x2 + dx]))
            eval_plusx1_minusx2 = func(np.array([x1 + dx, x2 - dx]))

            if np.all(
                current
                > np.array(
                    [
                        eval_minusx1,
                        eval_plusx1,
                        eval_minusx2,
                        eval_plusx2,
                        eval_minusx1_minusx2,
                        eval_plusx1_plusx2,
                        eval_minusx1_plusx2,
                        eval_plusx1_minusx2,
                    ]
                )
            ):
                modes.append([x1, x2])

    return modes


# ex 6.2
def marginal_x1(x: float) -> float:

    return 0.4 * gaussian_density_1d(x, 10, 1) + 0.6 * gaussian_density_1d(
        x, 0, 8.4**0.5
    )


def marginal_x2(x: float) -> float:

    return 0.4 * gaussian_density_1d(x, 2, 1) + 0.6 * gaussian_density_1d(
        x, 0, 1.7**0.5
    )


def gaussian_sum(x: np.ndarray) -> float:

    return 0.4 * gaussian_density_2d(
        x, np.array([10, 2]), np.eye(2)
    ) + 0.6 * gaussian_density_2d(
        x, np.array([0, 0]), np.array([[8.4, 2.0], [2.0, 1.7]])
    )


# integrating marginal_x1 to get expected value of x1
# mean_x1 = numerical_integration_1d(lambda x: x * marginal_x1(x), -30, 30)
# mean_x2 = numerical_integration_1d(lambda x: x * marginal_x2(x), -30, 30)
# print(mean_x1)
# print(mean_x2)

# # medians marginal
# print(find_median_1d(marginal_x1))
# print(find_median_1d(marginal_x2))

# modes marginal
print(find_modes_1d(marginal_x1))
print(find_modes_1d(marginal_x2))

# modes 2d
# print(find_modes_2d(gaussian_sum))


# integrating the pdfs
# lower_bound, upper_bound = -10.0, 10.0
# numerical_integral_1d = numerical_integration_1d(
#     partial(gaussian_density_1d, mu=0.0, sigma=1.0), lower_bound, upper_bound
# )
# print(numerical_integral_1d)

# # adjust these if mean / covariance matrix change
# lb_x1, ub_x1, lb_x2, ub_x2 = -5, 5, -5, 5
# numerical_integral_2d = numerical_integration_2d(
#     partial(gaussian_density_2d, mu_vec=np.array([0.0, 0.0]), covar_matrix=np.eye(2)),
#     lb_x1,
#     ub_x1,
#     lb_x2,
#     ub_x2,
# )
# print(numerical_integral_2d)
