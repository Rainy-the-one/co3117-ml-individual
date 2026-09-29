# Pre-Release Catch-Up: Machine Learning Workflow & Decision Trees (W01–W04)

*Author: To Quoc Tai*  
*StudentID: 2453141*

---

## 1. Machine Learning Foundations & Protocol Setup

### 1.1 General Learning Formulation
In our course framework, supervised learning is about approximating an unknown target function $h: \mathcal{X} \to \mathcal{Y}$ using empirical data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$. Every learner brings an **inductive bias**—a set of built-in assumptions about the hypothesis space $\mathcal{H}$ that allows it to generalize to unseen points rather than just memorizing observations.

### 1.2 Generalization, Underfitting, and Overfitting
The actual goal of machine learning is not driving training error to zero; it is minimizing generalization error on unseen data.

* **Underfitting (High Bias):** The model class is too simple to capture the underlying data patterns.
  * *Concrete Example:* Fitting a simple line ($y = w_1 x + w_0$) to a quadratic trajectory ($y = x^2$). Both training error and validation error stay consistently high.
* **Overfitting (High Variance):** The model has too much capacity and ends up fitting sample-specific noise.
  * *Concrete Example:* Fitting an unregularized polynomial of degree 15 through 10 noisy data points. The curve passes through every single point (training error $\approx 0$), but oscillates wildly between them, causing validation error to explode.

#### Bias-Variance Decomposition (Squared Error)
For a target $y = f(\mathbf{x}) + \epsilon$ where noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected prediction error decomposes into three distinct components:

$$\mathbb{E}_{\mathcal{D}} \big[(y - \hat{f}(\mathbf{x}))^2\big] = \text{Bias}^2\big[\hat{f}(\mathbf{x})\big] + \text{Var}\big[\hat{f}(\mathbf{x})\big] + \sigma^2$$

Where:
$$\text{Bias}\big[\hat{f}(\mathbf{x})\big] = \mathbb{E}_{\mathcal{D}}\big[\hat{f}(\mathbf{x})\big] - f(\mathbf{x})$$
$$\text{Var}\big[\hat{f}(\mathbf{x})\big] = \mathbb{E}_{\mathcal{D}}\Big[\big(\hat{f}(\mathbf{x}) - \mathbb{E}_{\mathcal{D}}[\hat{f}(\mathbf{x})]\big)^2\Big]$$

* $\sigma^2$ is the irreducible noise that no algorithm can eliminate.
* Increasing model complexity lowers bias but increases sensitivity to training variations (variance). Finding the optimal sweet spot requires cross-validation or validation monitoring.

### 1.3 Data Protocol & Leakage Controls on UCI HAR
To maintain experimental validity on the Human Activity Recognition (HAR) dataset, we enforce a strict validation protocol:

1. **Group/Subject-Aware Partitioning:**
   * *The Problem:* The raw dataset collects sequential inertial measurements from 30 subjects.
   * *Counterexample (What goes wrong):* If we shuffle all rows randomly across folds, time windows from the exact same person and walking trial leak into both train and test partitions. The model learns individual biometric quirks rather than general activity dynamics, yielding misleadingly high accuracy that collapses on new people.
   * *Correct Protocol:* We preserve the canonical group split: Subjects 1–21 are reserved for training/validation, while Subjects 22–30 are permanently sealed for final evaluation.

2. **Preprocessing Isolation (No In-Sample Leakage):**
   * *Rule:* Any parameter estimation (mean, standard deviation, imputation statistics) must be calculated exclusively on the training partition.
   * *Formula:*
     $$\mu_{\text{train}} = \frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} \mathbf{x}_i, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{tr}}} \sum_{i=1}^{N_{\text{tr}}} (\mathbf{x}_i - \mu_{\text{train}})^2}$$
   * Test/validation data are transformed strictly using $(\mu_{\text{train}}, \sigma_{\text{train}})$. Calling `.fit_transform()` on the whole dataset prior to splitting is strictly avoided.

3. **Evaluation Metrics:**
   * Because activities differ in sample volume, overall accuracy can be misleading. We adopt **Macro-averaged F1** as our primary metric:
     $$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot \text{Precision}_k \cdot \text{Recall}_k}{\text{Precision}_k + \text{Recall}_k}$$
     This weights each of the 6 physical activities equally.

---

## 2. Decision Tree Mechanics (Depth B Retrospective)

### 2.1 Recursive Partitioning & Impurity Criteria
A Decision Tree recursively divides feature space using axis-aligned orthogonal splits. At each internal node $S$, a split is selected greedily to maximize impurity reduction.

* **Shannon Entropy (Used in ID3 / C4.5):**
  $$H(S) = - \sum_{k=1}^K p_k \log_2 (p_k)$$
  where $p_k$ is the empirical probability of class $k$ in node $S$. For completely pure subsets, $H(S) = 0$; for evenly balanced binary distributions, $H(S) = 1.0$.

* **Gini Impurity (Used in CART):**
  $$\text{Gini}(S) = 1 - \sum_{k=1}^K p_k^2$$
  Gini measures the probability of misclassifying a randomly chosen element if it were labeled randomly according to the class distribution. It is computationally faster than Entropy because it does not require calculating logarithms.

* **Information Gain (IG):**
  $$IG(S, A) = H(S) - \sum_{v \in \text{Children}} \frac{\vert S_v \vert}{\vert S \vert} H(S_v)$$

### 2.2 Counterexample: The High-Cardinality Failure of Information Gain
* **Setup:** Consider an identification attribute such as `Sample_ID` (or `Timestamp`), where every training sample has a unique ID ($\vert S_v \vert = 1$ for all children).
* **Failure Mode:** Every resulting leaf contains exactly one sample, so $H(S_v) = 0$ for all branches. This produces:
  $$IG(S, \text{Sample\_ID}) = H(S) - 0 = H(S)$$
  Pure Information Gain ranks this attribute as the best possible split. In reality, the split has zero predictive power and completely fails on unseen data.
* **Resolution (C4.5 Gain Ratio):** C4.5 addresses this by normalizing $IG$ using intrinsic split entropy:
  $$SplitInfo(S, A) = - \sum_{v \in \text{Values}(A)} \frac{\vert S_v \vert}{\vert S \vert} \log_2 \left(\frac{\vert S_v \vert}{\vert S \vert}\right)$$
  $$GR(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$
  For `Sample_ID`, $SplitInfo$ explodes to $\log_2(\vert S \vert)$, heavily penalizing fragmented splits.

### 2.3 Vectorized NumPy Implementation of Entropy
Below is a standalone vectorized routine to compute Entropy and Information Gain without high-level library dependencies:

```python
import numpy as np

def compute_entropy(y: np.ndarray) -> float:
    """Computes Shannon entropy for discrete class labels."""
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / y.size
    # Add epsilon to prevent log2(0) runtime warnings
    return -float(np.sum(probabilities * np.log2(probabilities + 1e-12)))

def compute_information_gain(
    y_parent: np.ndarray, 
    y_left: np.ndarray, 
    y_right: np.ndarray
) -> float:
    """Calculates Information Gain for a candidate binary split."""
    n_total = y_parent.size
    if n_total == 0:
        return 0.0
    
    h_parent = compute_entropy(y_parent)
    weight_left = y_left.size / n_total
    weight_right = y_right.size / n_total
    
    h_children = (weight_left * compute_entropy(y_left) + 
                  weight_right * compute_entropy(y_right))
    return float(h_parent - h_children)
```

### 2.4 Continuous Attributes & Overfitting Control

#### Handling Continuous Features
In datasets with continuous features (such as the 561 inertial features in HAR), we sort unique feature values:
$$x_{(1)} < x_{(2)} < \dots < x_{(m)}$$

We evaluate candidate binary thresholds at the midpoints:
$$T_j = \frac{x_{(j)} + x_{(j+1)}}{2}$$


The threshold that produces the maximum impurity drop is selected.

#### Mitigating Overfitting via Pruning
* **Pre-pruning (Early Stopping):** We halt tree growth if node sample count falls below `min_samples_split`, or if the tree reaches `max_depth`.
* **Post-pruning:** Allow the tree to expand fully, then prune subtrees bottom-up if the validation accuracy does not degrade. In CART, this is formalized via Cost-Complexity pruning:
  $$R_\alpha(T) = R(T) + \alpha \vert T \vert$$
  where $R(T)$ is training error, $\vert T \vert$ is the leaf count, and $\alpha$ is tuned via cross-validation.

#### Reference Dissection (ML-From-Scratch)
In `ML-From-Scratch/mlfromscratch/supervised_learning/decision_tree.py`, the core divide-and-conquer logic is implemented in the `_build_tree()` method.

The implementation computes continuous threshold boundaries by testing every distinct value and measuring variance reduction (for regression) or information gain (for classification). A notable limitation of that reference code is that it re-sorts and evaluates all distinct threshold points at every split, giving it $\mathcal{O}(N \cdot D \log N)$ complexity per node—which runs slowly on a 561-dimensional dataset like HAR unless feature subsampling or max-depth constraints are applied.

---

## 3. Retrospective Reflection
* **Key Takeaway:** High training accuracy is straightforward to obtain with deep trees, but real model reliability requires careful data isolation, group-aware cross-validation, and well-chosen pruning thresholds.
* **Next Step (W05):** Transition from non-parametric partition trees to parametric linear models (Perceptron, Delta rule) and multi-layer perceptrons trained via backpropagation.