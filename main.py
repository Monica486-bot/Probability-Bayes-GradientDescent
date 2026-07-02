from data_prep import load_heights, initialize_parameters
from em_model import run_em, classify

CSV_PATH = "GaltonFamilies.csv"      
PARENT_COL = None             
CHILD_COL = None              
SEED = 0
MAX_ITERS = 50
TOL = 1e-6


def main():
    # 1. Load and pool the data (unlabeled)
    data = load_heights(CSV_PATH, parent_col=PARENT_COL, child_col=CHILD_COL)
    print(f"Loaded {len(data)} pooled height values.\n")

    # 2. Initialize parameters
    init_params = initialize_parameters(data, seed=SEED)

    # 3. Run EM, printing the tracking table as it goes
    print("Running EM...\n")
    final_params, history = run_em(data, init_params, max_iters=MAX_ITERS, tol=TOL)

    # 4. Figure out which component is "Children" (lower mean height)
    if final_params["mu1"] < final_params["mu2"]:
        child_label, parent_label = "Child", "Parent"
        mu_child, mu_parent = final_params["mu1"], final_params["mu2"]
    else:
        child_label, parent_label = "Parent", "Child"
        mu_child, mu_parent = final_params["mu2"], final_params["mu1"]

    print("\nFinal fitted parameters:")
    print(f"  {child_label} component mean: {mu_child:.3f}")
    print(f"  {parent_label} component mean: {mu_parent:.3f}")

    # 5. Live classification demo -- coach gives a test height
    test_height = float(input("\nEnter a test height to classify: "))
    result = classify(
        test_height,
        final_params,
        label1="Child" if final_params["mu1"] < final_params["mu2"] else "Parent",
        label2="Parent" if final_params["mu1"] < final_params["mu2"] else "Child",
    )

    print(f"\nHeight = {test_height}")
    for label, prob in result.items():
        print(f"  P({label}) = {prob:.4f}")


if __name__ == "__main__":
    main()