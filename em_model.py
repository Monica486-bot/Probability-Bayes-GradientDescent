import numpy as np

from data_prep import e_step, compute_log_likelihood, gaussian_pdf

def m_step(data, gamma1, gamma2):
    n = len(data)

    N1 = np.sum(gamma1)
    N2 = np.sum(gamma2)

    mu1 = np.sum(gamma1 * data) / N1
    mu2 = np.sum(gamma2 * data) / N2

    sigma1_sq = np.sum(gamma1 * (data - mu1) ** 2) / N1
    sigma2_sq = np.sum(gamma2 * (data - mu2) ** 2) / N2

    pi1 = N1 / n
    pi2 = N2 / n

    return {
        "mu1": float(mu1),
        "mu2": float(mu2),
        "sigma1_sq": float(sigma1_sq),
        "sigma2_sq": float(sigma2_sq),
        "pi1": float(pi1),
        "pi2": float(pi2),
    }

def run_em(data, init_params, max_iters=50, tol=1e-6, verbose=True):
    params = dict(init_params)
    history = []

    # Iteration 0: the initialization state
    ll0 = log_likelihood(data, params)
    history.append({"iteration": 0, **params, "log_likelihood": ll0})
    if verbose:
        _print_row(history[-1])

    prev_ll = ll0
    for it in range(1, max_iters + 1):
        gamma1, gamma2 = e_step(data, params)
        params = m_step(data, gamma1, gamma2)
        ll = log_likelihood(data, params)

        history.append({"iteration": it, **params, "log_likelihood": ll})
        if verbose:
            _print_row(history[-1])

        if abs(ll - prev_ll) < tol:
            if verbose:
                print(f"Converged after {it} iterations "
                      f"(log-likelihood change < {tol}).")
            break
        prev_ll = ll

    return params, history


def _print_row(row):
    print(
        f"Iter {row['iteration']:>2} | "
        f"mu1={row['mu1']:.4f}  mu2={row['mu2']:.4f} | "
        f"s1^2={row['sigma1_sq']:.4f}  s2^2={row['sigma2_sq']:.4f} | "
        f"pi1={row['pi1']:.4f}  pi2={row['pi2']:.4f} | "
        f"LL={row['log_likelihood']:.4f}"
    )

def classify(height, params, label1="Child", label2="Parent"):
    p1 = params["pi1"] * gaussian_pdf(height, params["mu1"], params["sigma1_sq"])
    p2 = params["pi2"] * gaussian_pdf(height, params["mu2"], params["sigma2_sq"])
    total = p1 + p2

    posterior1 = p1 / total
    posterior2 = p2 / total

    return {label1: float(posterior1), label2: float(posterior2)}