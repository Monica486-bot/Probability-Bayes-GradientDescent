import numpy as np
import pandas as pd


def load_heights(filepath, compare="mother", child_gender=None):
    df = pd.read_csv(filepath, index_col=0)
    df.columns = df.columns.str.strip().str.lower()

    # deduplicate parents because each parent row repeats per child
    parent_heights = df.drop_duplicates(subset=["family"])[compare].dropna().values

    if child_gender:
        mask = df["gender"].str.lower() == child_gender.lower()
        child_heights = df.loc[mask, "childheight"].dropna().values
    else:
        child_heights = df["childheight"].dropna().values

    heights = np.concatenate([parent_heights, child_heights])

    print(f"Parents: {len(parent_heights)} | Children: {len(child_heights)} | Total: {len(heights)}")
    print(f"Min: {heights.min():.2f}  Max: {heights.max():.2f}  Mean: {heights.mean():.2f}")

    return heights


def initialize_parameters(heights, seed=42):
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(heights), size=2, replace=False)
    mu1, mu2 = float(heights[idx[0]]), float(heights[idx[1]])

    if mu1 > mu2:
        mu1, mu2 = mu2, mu1

    variance = float(np.var(heights))

    params = {
        "mu1": mu1,
        "mu2": mu2,
        "sigma1_sq": variance,
        "sigma2_sq": variance,
        "pi1": 0.5,
        "pi2": 0.5,
    }

    print(f"Init -> mu1={mu1:.2f}  mu2={mu2:.2f}  var={variance:.2f}  pi1=0.5  pi2=0.5")
    return params


def gaussian_pdf(x, mu, sigma_sq):
    sigma_sq = max(sigma_sq, 1e-6)
    coefficient = 1.0 / np.sqrt(2.0 * np.pi * sigma_sq)
    exponent = -((x - mu) ** 2) / (2.0 * sigma_sq)
    return coefficient * np.exp(exponent)


def e_step(heights, params):
    w1 = params["pi1"] * gaussian_pdf(heights, params["mu1"], params["sigma1_sq"])
    w2 = params["pi2"] * gaussian_pdf(heights, params["mu2"], params["sigma2_sq"])

    total = w1 + w2
    total = np.where(total == 0, 1e-300, total)

    R = np.column_stack([w1 / total, w2 / total])
    return R


def compute_log_likelihood(heights, params):
    w1 = params["pi1"] * gaussian_pdf(heights, params["mu1"], params["sigma1_sq"])
    w2 = params["pi2"] * gaussian_pdf(heights, params["mu2"], params["sigma2_sq"])
    mixture = np.clip(w1 + w2, 1e-300, None)
    return float(np.sum(np.log(mixture)))


if __name__ == "__main__":
    import os

    DATASET_PATH = "galton.csv"

    if not os.path.exists(DATASET_PATH):
        print("File not found, using synthetic data\n")
        rng = np.random.default_rng(0)
        heights = np.concatenate([
            rng.normal(64.1, 2.3, 205),
            rng.normal(66.7, 3.5, 934),
        ])
    else:
        heights = load_heights(DATASET_PATH, compare="mother")

    params = initialize_parameters(heights)

    ll = compute_log_likelihood(heights, params)
    print(f"\nLog-Likelihood at iteration 0: {ll:.4f}")

    R = e_step(heights, params)
    print(f"\nResponsibilities shape: {R.shape}")
    print(f"Rows sum to 1: {np.allclose(R.sum(axis=1), 1.0)}")
    print(f"\n{'Height':>8}  {'P(shorter)':>12}  {'P(taller)':>12}")
    for i in range(5):
        print(f"{heights[i]:8.2f}  {R[i,0]:12.6f}  {R[i,1]:12.6f}")
