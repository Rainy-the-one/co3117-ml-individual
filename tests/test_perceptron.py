import numpy as np
from sklearn.linear_model import Perceptron as SklearnPerceptron
from sklearn.metrics import accuracy_score, f1_score

from src.from_scratch.perceptron import Perceptron
from src.data import load_har_data


def test_linearly_separable_logic_gates():
    print("=== 1. TEST LINEARLY SEPARABLE LOGIC GATES (AND / OR) ===")
    # Test AND Gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([-1, -1, -1, 1])

    clf = Perceptron(learning_rate=0.1, epochs=50)
    clf.fit(X_and, y_and)
    preds = clf.predict(X_and)

    acc = np.mean(preds == y_and)
    print(f"AND Gate - Predict: {preds} | True Labels: {y_and}")
    print(f"Accuracy: {acc * 100:.1f}%\n")
    assert acc == 1.0, "Perceptron must learn the AND gate perfectly!"


def test_xor_limitation():
    print("=== 2. TEST COUNTEREXAMPLE: XOR GATE (NOT LINEARLY SEPARABLE) ===")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([-1, 1, 1, -1])  # Diagonal elements are -1, off-diagonal are 1

    clf = Perceptron(learning_rate=0.1, epochs=100)
    clf.fit(X_xor, y_xor)
    preds = clf.predict(X_xor)

    acc = np.mean(preds == y_xor)
    print(f"XOR Gate - Predict: {preds} | True Labels: {y_xor}")
    print(f"Accuracy: {acc * 100:.1f}% \n")


def test_uci_har_binary_benchmark():
    print("=== 3. BENCHMARK in UCI HAR (WALKING vs. LAYING) ===")
    # Load data UCI HAR
    (X_train, y_train, _), (X_test, y_test, _), _ = load_har_data()

    classes = np.unique(y_train)
    c1, c2 = classes[3], classes[4]  # choose two classes for binary classification

    train_mask = np.isin(y_train, [c1, c2])
    test_mask = np.isin(y_test, [c1, c2])

    X_tr_sub, y_tr_sub = X_train[train_mask], y_train[train_mask]
    X_te_sub, y_te_sub = X_test[test_mask], y_test[test_mask]

    # Normalize labels to {-1, 1}
    y_tr_bin = np.where(y_tr_sub == c1, 1, -1)
    y_te_bin = np.where(y_te_sub == c1, 1, -1)

    # 1. Run Perceptron From-Scratch
    my_p = Perceptron(learning_rate=0.01, epochs=100)
    my_p.fit(X_tr_sub, y_tr_bin)
    my_preds = my_p.predict(X_te_sub)
    my_acc = accuracy_score(y_te_bin, my_preds)
    my_f1 = f1_score(y_te_bin, my_preds, average="macro")

    # 2. Run Scikit-Learn Benchmark
    sk_p = SklearnPerceptron(eta0=0.01, max_iter=100, random_state=42)
    sk_p.fit(X_tr_sub, y_tr_bin)
    sk_preds = sk_p.predict(X_te_sub)
    sk_acc = accuracy_score(y_te_bin, sk_preds)
    sk_f1 = f1_score(y_te_bin, sk_preds, average="macro")

    print(f"| Model | Test Accuracy | Macro-F1 |")
    print(f"| :--- | :--- | :--- |")
    print(f"| My Perceptron (Scratch) | {my_acc * 100:.2f}% | {my_f1:.4f} |")
    print(f"| Sklearn Perceptron      | {sk_acc * 100:.2f}% | {sk_f1:.4f} |")


if __name__ == "__main__":
    test_linearly_separable_logic_gates()
    test_xor_limitation()
    test_uci_har_binary_benchmark()