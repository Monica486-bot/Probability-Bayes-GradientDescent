import numpy as np
import matplotlib.pyplot as plt
from gd_core import predict, mse, gradients, X, y, LEARNING_RATE, ITERATIONS
from gd_scipy_check import scipy_gradient_check


def run_gradient_descent(X, y, lr=LEARNING_RATE, iters=ITERATIONS):
    m = np.array([[-1], [2]], dtype=float)   # (2,1)
    b = np.array([[1],  [1]], dtype=float)   # (2,1)

    history = {"m1": [], "m2": [], "b1": [], "b2": [], "error": []}

    print(f"{'Iter':>4}  {'m1':>8}  {'m2':>8}  {'b1':>8}  {'b2':>8}  {'MSE':>12}")
    print("-" * 60)

    for i in range(iters + 1):
        y_pred        = predict(X, m, b)
        error         = mse(y, y_pred)
        dJ_dm, dJ_db  = gradients(X, y, m, b)

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


def plot_results(history):
    iters = range(len(history["error"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iters, history["m1"], marker="o", label="m1")
    ax1.plot(iters, history["m2"], marker="o", label="m2")
    ax1.plot(iters, history["b1"], marker="s", linestyle="--", label="b1")
    ax1.plot(iters, history["b2"], marker="s", linestyle="--", label="b2")
    ax1.set_title("Parameters over Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(iters, history["error"], marker="o", color="red")
    ax2.set_title("MSE Error over Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("MSE")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("gradient_descent_plots.png", dpi=150)
    plt.show()
