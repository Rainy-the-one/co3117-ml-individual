# AI Use Trail

### Entry 1: W05 / 2026-09-29
- **Learning Question:** What are the structural requirements and workflow specifications of the CO3117 longitudinal assignment, how should the repository and data pipeline be initialized to prevent leakage, and what theoretical and mathematical arguments must be synthesized for the W01–W04 pre-release catch-up artifacts?
- **Pre-AI Evidence:** Release-day baseline state; local repository directory initialized; no existing directory structure, data loader, or catch-up documentation.
- **AI Tool:** Gemini
- **Prompt Purpose:** Socratic consulting on assignment specification breakdown, directory scaffolding, data hygiene protocol design under group constraints, and theoretical framing for the catch-up report and baseline diagnostic.
- **Hint/Question Received:** 
  1. Clarified the longitudinal principle: exactly one dataset and use case throughout the semester, paired with a two-state commit progression (first attempt vs. post-reference).
  2. Recommended the official UCI HAR dataset structure and identified the necessity of subject-aware grouping (`subject_train.txt` vs. `subject_test.txt`) to avoid biometric identity leakage.
  3. Outlined mandatory Markdown layouts (`PROGRESS.md`, `AI_USE.md`, `MODEL_LOG.md`, `PRE_RELEASE_CATCHUP.md`) and structural requirements for the handwritten baseline diagnostic PDF.
  4. Formulated core theoretical criteria for Decision Trees: Information Gain flaws on high-cardinality features, C4.5 Gain Ratio penalties, continuous splitting thresholds via sorting midpoints, surrogate splits (CART) vs. fractional weights (C4.5) for missing attributes, and L1 vs. L2 loss derivations for leaf constants (Median vs. Mean).
- **Verification Source:** 
  - CO3117 Machine Learning Course Syllabus & Assignment Specification (HK261, HCMUT).
  - Lecture Slide Chapters 1 (ML Foundations) and Chapter 2 (Decision Trees).
  - Murphy (2022), *Probabilistic Machine Learning: An Introduction*, Chapters 5 & 18.
- **What Changed:** 
  1. Constructed the complete project directory hierarchy according to specification guidelines.
  2. Implemented `src/data.py` enforcing subject-level partitioning (21 train subjects, 9 sealed test subjects) and isolated feature standardization (`StandardScaler` fitted strictly on `X_train`).
  3. Configured environment reproducibility via pinned package versions in `requirements.txt`.
  4. Authored `docs/pre-release/PRE_RELEASE_CATCHUP.md` with explicit bias-variance formulations, continuous split sorting procedures, and a reference dissection of `ML-From-Scratch`.
  5. Formulated and structured the handwritten diagnostic exam sheet (`exercises/release-baseline-w01-w02.pdf`) covering generalization bounds, preprocessing leakage mechanisms, entropy calculations, and leaf prediction criteria.
- **Closed-Book Reproduction:** 
  - Able to independently prove why unconstrained Information Gain degenerates under unique identification keys.
  - Able to derive the mathematical conditions under which leaf constants minimize squared loss (mean) versus absolute loss (median).
  - Able to verbally justify why global preprocessing across train and test partitions invalidates empirical risk bounds.

  ### Entry 2: W05 / 2026-10-02 (Test Environment Setup)
- **Learning Question:** I kept getting a `ModuleNotFoundError` for 'src' when trying to run my test script directly from the `tests/` folder. How do I fix the Python path?
- **Pre-AI Evidence:** First-attempt commit `<insert_first_attempt_hash>`
- **AI Tool:** Gemini
- **Prompt Purpose:** Debugging a Python directory structure issue.
- **Hint/Question Received:** Suggested running the test script as a module from the root directory using the `-m` flag so Python recognizes the `src` folder.
- **Verification Source:** Python official documentation on modules and packages.
- **What Changed:** Switched my terminal command from `python tests/test_perceptron.py` to `python -m tests.test_perceptron`.
- **Closed-Book Reproduction:** Yes. I now understand how `PYTHONPATH` works for root-level modules.

### Entry 3: W05 / 2026-10-02 (Data Pipeline Unpacking)
- **Learning Question:** Hit a `ValueError: too many values to unpack (expected 2, got 3)` when importing `load_har_data()` into my benchmark test. How to resolve this?
- **Pre-AI Evidence:** First-attempt commit `<insert_first_attempt_hash>`
- **AI Tool:** Gemini
- **Prompt Purpose:** Debugging a tuple unpacking mismatch.
- **Hint/Question Received:** Pointed out that my R0 data loader returns subject IDs (to prevent leakage) but my test script wasn't catching them. Advised using underscores (`_`) to drop the unused subject variables.
- **Verification Source:** Checked my own `src/data.py` return statement.
- **What Changed:** Updated the unpacking logic to `(X_train, y_train, _), (X_test, y_test, _), _ = load_har_data()`.
- **Closed-Book Reproduction:** Yes.

### Entry 4: W05 / 2026-10-02 (Vectorization & Code-to-Theory)
- **Learning Question:** How exactly do the matrix dot products in the `ML-From-Scratch` Perceptron code map back to the SGD and Chain Rule math formulas we learned in class?
- **Pre-AI Evidence:** First-attempt commit `<insert_first_attempt_hash>`
- **AI Tool:** Gemini
- **Prompt Purpose:** Socratic code-reading assistant.
- **Hint/Question Received:** Broke down the vectorization: `X.dot(W)` is the batch linear combination, multiplying the loss gradient with the activation gradient is the literal chain rule, and `X.T.dot(err)` accumulates the weight gradients across the entire batch.
- **Verification Source:** CO3117 Lecture Slides (Chapter 3) and `ML-From-Scratch` repository.
- **What Changed:** Used this breakdown to write the Code-to-Theory trace section in my W05 blog. Decided to drop nested loops and use matrix operations for future models.
- **Closed-Book Reproduction:** Yes. I can write out the vectorized forward and backward passes manually now.