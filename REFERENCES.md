# Academic References

1. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press.
2. Müller, A. C., & Guido, S. (2017). *Introduction to Machine Learning with Python*. O'Reilly Media.
3. Mitchell, T. M. (1997). *Machine Learning*. McGraw-Hill.
4. Marsland, S. (2009). *Machine Learning: An Algorithmic Perspective*. Chapman & Hall/CRC.
5. Linder-Norén, E. (2019). *ML-From-Scratch*. GitHub repository: `https://github.com/eriklindernoren/ML-From-Scratch`.
6. Anguita, D., et al. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN.

---

### Usage & Attribution Log

**W05: Perceptron & Backpropagation**
*   **Theory:** Verified discrete update rules and gradient descent limitations against Mitchell [3] and Murphy [1].
*   **Code Adaptation:** Used Linder-Norén's repository [5] as a structural reference to understand matrix vectorization. Specifically, I mapped their `X.dot(W)` implementation back to the mini-batch math equations to replace my initial, inefficient nested `for` loops during the post-reference phase.
*   **Data:** Used the standard UCI HAR dataset [6] to run the W05 benchmark testing linear separability constraints.