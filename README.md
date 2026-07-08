# Formative 3 - Probability Distributions, Bayesian Probability, and Gradient Descent Implementation

This repository contains our team's complete implementation for Parts 1 through 4 of the assignment. We have built all core machine learning models from scratch using foundational mathematics and pure, native Python code.

## 1. Group Information & Contributions

| Team Member | Core Project Role & Responsibility | Assigned Tasks Covered |
| :--- | :--- | :--- |
| **David Nshuti Ngarambe** | EM Algorithm Framework Lead & Q&A Facilitator | Part 1 Code, Iteration 4 Manual Matrix Calculus |
| **Audric Ntaganda Mihigo** | Mixture Distribution Flow Designer & Live Classifier Engineer | Part 1 Soft-Splitting Logic, Iteration 2 Manual Matrix Calculus |
| **Emma Tiffany Umwari** | Pure Python Bayesian Text Pipeline & Parser Developer | Part 2 Data Stream Setup, Iteration 1 Manual Matrix Calculus |
| **Monica Akoi Dau Ahol** | Matrix Calculus Lead & Automation Engineer | Part 3 Chain Rule Derivations, Part 4 SciPy Code & Plots, Iteration 3 Manual Matrix Calculus |

---

## 2. Part 1: EM Algorithm (Grouping Height Data)

### Core Analysis Question
* **Should you or should you not simply draw a line at the dataset's global mean to split the data into two piles, and then calculate the mean of each pile?**
* **Answer:** **No, you absolutely should not.** Splitting the dataset strictly at a single global mean line fails because real-world biological population data significantly overlaps. A rigid cutoff line misclassifies shorter parents as children and taller children as parents, cutting off the true tails of both underlying Gaussians. This distorts the resulting mean calculations and ignores variance. The Expectation-Maximization (EM) algorithm resolves this by assigning **soft, probabilistic weights (responsibilities)** to data points, accurately capturing the true hidden parameters without artificial boundaries.

### Our Live Test Case Result Example
When the presentation coach inputs a custom validation test height, the system dynamically utilizes our trained GMM variables to compute the exact posterior metrics:
* **Probability it is a Child:** $0.02\%$
* **Probability it is a Parent (Basketball Player):** $99.98\%$
* **Conclusion:** The model correctly and confidently classifies adult/tall individuals into the Parent distribution based on calculated Gaussian densities rather than a naive split.

---

## 3. Part 2: Bayesian Sentiment Inference

### Project Boundaries
Per the explicit assignment constraints, this text analyzer is built using **pure, basic Python math operations**. No external data analysis libraries (such as Pandas) or machine learning toolkits (such as Scikit-Learn) were utilized. Data ingestion was achieved exclusively via the native Python `csv` stream reader to satisfy the core script evaluation criteria.

### Final Keyword Probability Balance
We selected 4 positive and 4 negative keywords to update our perfectly balanced baseline **Prior $P(\text{Positive}) = 0.5000$** across the $50,000$ IMDb review dataset using Bayes' Theorem:

$$P(\text{Positive} \mid \text{keyword}) = \frac{P(\text{keyword} \mid \text{Positive}) \times P(\text{Positive})}{P(\text{keyword})}$$

Our modular architecture evaluated the following exact posterior tracking metrics:
* **Tracking Positive Words:** Keywords like `wonderful` ($0.8203$) and `excellent` ($0.8099$) successfully drive the final posterior probability toward certainty ($1.00$).
* **Tracking Negative Words:** Keywords like `worst` ($0.0927$) and `waste` ($0.0993$) successfully drop the final posterior probability toward impossibility ($0.00$).

---

## 4. Part 3 & 4: Gradient Descent Optimization

### Software Design (The DRY Principle)
To ensure the mathematical execution is clean, unabstracted, and fully visible, we avoided pre-built linear regression libraries. We strictly maintained the **DRY (Don't Repeat Yourself)** principle by packaging our structural matrix multiplications inside a clean, centralized operational execution loop. 

### Trajectory Interpretation
* **Loss Optimization:** Our programmatic implementation tracks the matrix error variables across iterations. The Mean Squared Error (MSE) cost function $J(m,b)$ drops rapidly over successive updates, confirming our partial derivative calculations are correct.
* **Why Code and Manual Steps Differ:** The manual calculations completed in Part 3 were executed step-by-step across individuals to demonstrate personal arithmetic mastery of the matrix calculus and Chain Rule. The automated Python script executes full vectorized Batch Gradient Descent using the `SciPy` derivative extraction wrapper to update parameters seamlessly across the entire matrix system. Both tracks align and converge toward identical minimized error limits.
