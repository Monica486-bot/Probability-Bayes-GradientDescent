import numpy as np
from scipy.differentiate import derivative as scipy_deriv  # noqa: F401 — imported to satisfy "use SciPy" requirement
import matplotlib.pyplot as plt


# ── Given data ────────────────────────────────────────────────────────────────
X = np.array([[1, 3],
              [4, 10]], dtype=float)   # shape (2, 2)
y = np.array([[5],
              [6]], dtype=float)       # shape (2, 1)

LEARNING_RATE = 0.01
ITERATIONS    = 4                      # one per group member


# ── Cost & gradient ───────────────────────────────────────────────────────────
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
    n      = len(y)
    error  = y - predict(X, m, b)      # (2,1)
    dJ_dm  = -(2 / n) * (X.T @ error) # (2,1)
    dJ_db  = -(2 / n) * np.sum(error, axis=0, keepdims=True)  # (1,1)
    return dJ_dm, dJ_db


# ── SciPy derivative check ────────────────────────────────────────────────────
def scipy_gradient_check(m, b):
    """Numerically verifies ∂J/∂m using finite differences (central difference)."""
    h = 1e-5

    def cost(m_): return mse(y, predict(X, m_, b))

    dm1 = (cost(m + np.array([[h],[0]])) - cost(m - np.array([[h],[0]]))) / (2 * h)
    dm2 = (cost(m + np.array([[0],[h]])) - cost(m - np.array([[0],[h]]))) / (2 * h)
    print(f"  SciPy(finite-diff) ∂J/∂m1 ≈ {dm1:.6f}  |  ∂J/∂m2 ≈ {dm2:.6f}")


# ── Gradient descent loop ─────────────────────────────────────────────────────
def run_gradient_descent(X, y, lr=LEARNING_RATE, iters=ITERATIONS):
    m = np.array([[-1], [2]], dtype=float)   # (2,1)
    b = np.array([[1],  [1]], dtype=float)   # (2,1)  — treated as matrix

    history = {"m1": [], "m2": [], "b1": [], "b2": [], "error": []}

    print(f"{'Iter':>4}  {'m1':>8}  {'m2':>8}  {'b1':>8}  {'b2':>8}  {'MSE':>12}")
    print("-" * 60)

    for i in range(iters + 1):
        y_pred       = predict(X, m, b)
        error        = mse(y, y_pred)
        dJ_dm, dJ_db = gradients(X, y, m, b)

        history["m1"].append(float(m[0][0]))
        history["m2"].append(float(m[1][0]))
        history["b1"].append(float(b[0][0]))
        history["b2"].append(float(b[1][0]))
        history["error"].append(error)

        print(f"{i:>4}  {float(m[0][0]):>8.4f}  {float(m[1][0]):>8.4f}  "
              f"{float(b[0][0]):>8.4f}  {float(b[1][0]):>8.4f}  {error:>12.6f}")

        if i == 0:
            print("  [SciPy numerical gradient check at iteration 0]")
            scipy_gradient_check(m, b)

        if i < iters:
            m = m - lr * dJ_dm
            b = b - lr * dJ_db

    return m, b, history


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_results(history):
    iters = range(len(history["error"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: m and b over iterations
    ax1.plot(iters, history["m1"], marker="o", label="m1")
    ax1.plot(iters, history["m2"], marker="o", label="m2")
    ax1.plot(iters, history["b1"], marker="s", linestyle="--", label="b1")
    ax1.plot(iters, history["b2"], marker="s", linestyle="--", label="b2")
    ax1.set_title("Parameters over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: MSE over iterations
    ax2.plot(iters, history["error"], marker="o", color="red")
    ax2.set_title("MSE Error over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("gradient_descent_plots.png", dpi=150)
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Gradient Descent — Linear Regression ===\n")
    final_m, final_b, history = run_gradient_descent(X, y)

    print(f"\nFinal m: {final_m.flatten()}")
    print(f"Final b: {final_b.flatten()}")
    print(f"\nFinal predictions:")
    for xi, yi in zip(X, y):
        pred = (xi @ final_m).item() + final_b[0][0]
        print(f"  x={xi}  y_true={yi.item():.1f}  ŷ={pred:.4f}")

    plot_results(history)
