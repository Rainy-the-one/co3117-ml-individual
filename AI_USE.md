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