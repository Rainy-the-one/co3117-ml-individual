import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "UCI HAR Dataset")


def load_har_data(data_dir=DATA_DIR):
    """
    Load the UCI HAR dataset.
    Preserves the subject-aware split to prevent data leakage.
    """
    labels = pd.read_csv(
        os.path.join(data_dir, "activity_labels.txt"),
        sep=r"\s+",
        header=None,
        names=["idx", "activity"],
    )

    # Train data
    X_train = np.loadtxt(os.path.join(data_dir, "train", "X_train.txt"))
    y_train = np.loadtxt(os.path.join(data_dir, "train", "y_train.txt"), dtype=int) - 1
    subj_train = np.loadtxt(os.path.join(data_dir, "train", "subject_train.txt"), dtype=int)

    # Test data (SEALED TEST SET)
    X_test = np.loadtxt(os.path.join(data_dir, "test", "X_test.txt"))
    y_test = np.loadtxt(os.path.join(data_dir, "test", "y_test.txt"), dtype=int) - 1
    subj_test = np.loadtxt(os.path.join(data_dir, "test", "subject_test.txt"), dtype=int)

    return (X_train, y_train, subj_train), (X_test, y_test, subj_test), labels["activity"].values


def preprocess_data(X_train, X_test):
    """
    Fit the scaler ONLY on the training set to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    (X_tr, y_tr, s_tr), (X_te, y_te, s_te), classes = load_har_data()
    print(f"X_train shape: {X_tr.shape} | Subjects: {np.unique(s_tr)}")
    print(f"X_test shape:  {X_te.shape}  | Subjects: {np.unique(s_te)}")
    print(f"Classes: {classes}")