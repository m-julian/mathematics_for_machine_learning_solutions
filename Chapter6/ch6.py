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


lower_bound, upper_bound = -10.0, 10.0
numerical_integral_1d = numerical_integration_1d(
    partial(gaussian_density_1d, mu=0.0, sigma=1.0), lower_bound, upper_bound
)
print(numerical_integral_1d)

# adjust these if mean / covariance matrix change
lb_x1, ub_x1, lb_x2, ub_x2 = -5, 5, -5, 5
numerical_integral_2d = numerical_integration_2d(
    partial(gaussian_density_2d, mu_vec=np.array([0.0, 0.0]), covar_matrix=np.eye(2)),
    lb_x1,
    ub_x1,
    lb_x2,
    ub_x2,
)
print(numerical_integral_2d)
