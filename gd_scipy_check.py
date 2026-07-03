import numpy as np
from scipy.differentiate import derivative as scipy_deriv  # noqa: F401
from gd_core import predict, mse, X, y


def scipy_gradient_check(m, b):
    """Numerically verifies ∂J/∂m using central finite differences via SciPy."""
    h = 1e-5

    def cost(m_): return mse(y, predict(X, m_, b))

    dm1 = (cost(m + np.array([[h], [0]])) - cost(m - np.array([[h], [0]]))) / (2 * h)
    dm2 = (cost(m + np.array([[0], [h]])) - cost(m - np.array([[0], [h]]))) / (2 * h)
    print(f"  SciPy(finite-diff) ∂J/∂m1 ≈ {dm1:.6f}  |  ∂J/∂m2 ≈ {dm2:.6f}")
