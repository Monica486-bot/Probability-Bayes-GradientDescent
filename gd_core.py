import numpy as np

# Given data
X = np.array([[1, 3],
              [4, 10]], dtype=float)   # shape (2, 2)
y = np.array([[5],
              [6]], dtype=float)       # shape (2, 1)

LEARNING_RATE = 0.01
ITERATIONS    = 4                      # one per group member


def predict(X, m, b):
    """ŷ = X·m + b  (matrix form)"""
    return X @ m + b                   # (2,1)


def mse(y_true, y_pred):
    n = len(y_true)
    return float(np.sum((y_true - y_pred) ** 2) / n)


def gradients(X, y, m, b):
    """
    ∂J/∂m = -(2/n) · Xᵀ · (y - ŷ)
    ∂J/∂b = -(2/n) · 1ᵀ · (y - ŷ)
    """
    n     = len(y)
    error = y - predict(X, m, b)       # (2,1)
    dJ_dm = -(2 / n) * (X.T @ error)  # (2,1)
    dJ_db = -(2 / n) * np.sum(error, axis=0, keepdims=True)  # (1,1)
    return dJ_dm, dJ_db
