# W05: Perceptron, Delta Rule, and Backpropagation

### A. Code-to-Theory Trace
After committing my first-attempt from scratch, I reviewed the `ML-From-Scratch` repository to see how a proper implementation is structured. The biggest takeaway was their use of matrix vectorization instead of the clunky nested `for` loops I wrote. Here is how their code maps to the math we learned in class:

*   **Vectorized Forward Pass:** `linear_output = X.dot(self.W) + self.w0`. Instead of updating row by row, this computes the linear combination for the entire batch at once. It perfectly matches the mathematical model $Y = XW^\top + \mathbf{1}b^\top$.
*   **The Chain Rule in Action:** `error_gradient = self.loss.gradient(y, y_pred) * self.activation_func.gradient(linear_output)`. This single line is the literal implementation of the backward pass chain rule $\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z}$. It multiplies the loss derivative with the local derivative of the activation function element-wise to propagate the error backward.
*   **SGD Weight Update:** `grad_wrt_w = X.T.dot(error_gradient)` followed by `self.W -= self.learning_rate * grad_wrt_w`. The dot product here basically accumulates the gradients across all samples in the batch, calculating $\nabla_W L = X^T \delta$. Then, it takes a step in the opposite direction of the gradient to minimize the loss.

### B. Controlled Experiment: Linear Separability Limits
Following the assignment protocol, I ran a benchmark comparing my NumPy Perceptron with `scikit-learn`'s implementation on the canonical UCI Human Activity Recognition Using Smartphones dataset[cite: 7]. As required, the evaluation uses Macro-F1 and Test Accuracy[cite: 7].

**Scenario 1: WALKING vs. LAYING (Perfectly Separable)**
| Model | Test Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| My Perceptron (Scratch) | 100.00% | 1.0000 |git add docs/weekly/w05-perceptron-delta.md tests/test_perceptron.py
| Sklearn Perceptron      | 100.00% | 1.0000 |

*Observation:* Both hit 100%. This happens because a dynamic activity (Walking) and a static one (Laying) have wildly different sensor variances. In a 561-dimensional space, they are perfectly linearly separable, so the Perceptron converges without a hitch.

**Scenario 2: SITTING vs. STANDING (Overlapping Classes)**
| Model | Test Accuracy | Macro-F1 |
| :--- | :--- | :--- |
| My Perceptron (Scratch) | 83.97% | 0.8339 |
| Sklearn Perceptron      | 93.06% | 0.9304 |

*Observation:* When classifying Sitting (class 3) versus Standing (class 4), accuracy drops significantly compared to the 100% baseline. My scratch implementation achieved roughly 84%, while Sklearn hit 93% (likely due to internal optimizations like shuffling, early stopping criteria, or adaptive learning rates). Because these two static postures share nearly identical accelerometer and gyroscope profiles, they overlap heavily in the feature space. A single-layer Perceptron only draws a flat hyper-plane, so it physically cannot separate them perfectly without non-linear transformations.

### C. Failure & Misconception (Section E)
*   **The Failure:** My sanity check script yielded exactly 50.0% accuracy when trying to train the Perceptron on an XOR logic gate.
*   **The Misconception:** I initially assumed a Perceptron could learn any basic logic gate as long as I gave it enough epochs. I realized that the points for XOR are positioned on opposite diagonal corners of a square. It is mathematically impossible to draw a single straight line that separates these diagonals. To solve this, a Multilayer Perceptron (MLP) is required, as the hidden layers act as non-linear transformers that warp the feature space before classification.

### D. Written-Exam Capsule (Section F)
The discrete Perceptron is a linear binary classifier that updates its weights strictly when a hard misclassification occurs. Geometrically, the update rule adds a fraction of the input vector to the weight vector, rotating the decision boundary's normal vector toward the misclassified sample to correct the error. However, because it relies entirely on a discrete step-function error, it cannot converge on non-linearly separable datasets. In contrast, the Delta rule (LMS) performs continuous gradient descent on a Mean Squared Error surface, allowing the model to make fine-grained weight adjustments even when the thresholded classification is already correct. To handle non-linear problems, a Multilayer Perceptron (MLP) uses continuous activation functions (like ReLU) so that error signals can flow backward from the output layer to update hidden weights via the Chain Rule.

### E. Reflection (Section G)
Writing the code manually really highlighted the difference between discrete updates and continuous gradients. I can now clearly explain why my first-attempt model failed on XOR but aced the AND gate. Moving forward, I realize how crucial vectorized matrix operations are for performance, and I'll be abandoning nested loops for all future neural network implementations. Next week, I plan to test how different activation functions (like ReLU vs. Sigmoid) actually affect the vanishing gradient problem during training.

### F. Inquiry Trail (Section H)
As required by the assignment rules for logging AI assistance:
*   **Debugging the Test Script:** I got a `ModuleNotFoundError` because I was running my test script directly from inside the `tests/` folder. AI pointed out I needed to run it as a module (`python -m tests.test_perceptron`) from the root directory. 
*   **Fixing Data Pipeline Unpacking:** Hit a `ValueError` because my R0 `load_har_data()` returned subject IDs to prevent data leakage, but my test script only expected `X` and `y`. Used AI to figure out the correct tuple unpacking syntax with underscores (`_`) to ignore the subject IDs in this specific benchmark.
*   **Code-to-Theory Translation:** I understood the math from the lecture, but I struggled to see how it mapped to the numpy code in the reference repo. I used AI as a Socratic tutor to help me trace the matrix dot products directly back to the mini-batch gradient descent formulas. I can now reconstruct the vectorized forward and backward passes on my own.