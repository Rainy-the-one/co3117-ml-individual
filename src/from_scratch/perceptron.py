import numpy as np

class Perceptron:
    def __init__(self, learning_rate = 0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Ensure targets are strictly 1 and -1
        y_encoded = np.where(y <= 0, -1, 1)

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1
                
                # Perceptron update rule
                update = self.lr * (y_encoded[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)
