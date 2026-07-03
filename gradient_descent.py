from gd_core import X, y
from gd_train import run_gradient_descent, plot_results

if __name__ == "__main__":
    print("=== Gradient Descent — Linear Regression ===\n")
    final_m, final_b, history = run_gradient_descent(X, y)

    print(f"\nFinal m: {final_m.flatten()}")
    print(f"Final b: {final_b.flatten()}")
    print("\nFinal predictions:")
    for xi, yi in zip(X, y):
        pred = (xi @ final_m).item() + final_b[0][0]
        print(f"  x={xi}  y_true={yi.item():.1f}  ŷ={pred:.4f}")

    plot_results(history)
