# Model Performance Log

| Week | Date | Model Architecture | Dataset (Task) | Test Accuracy | Macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W05 | 2026-10-02 | NumPy Perceptron (Scratch) | UCI HAR (Sitting vs Standing) | 83.97% | 0.8339 | Single-layer linear boundary. Proves limited capacity on overlapping, non-linearly separable classes. |
| W05 | 2026-10-02 | Sklearn Perceptron | UCI HAR (Sitting vs Standing) | 93.06% | 0.9304 | Baseline reference. Higher accuracy likely due to internal data shuffling and adaptive optimizations. |