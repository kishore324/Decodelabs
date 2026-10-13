"""
DecodeLabs - Artificial Intelligence | Project 2: Data Classification Using AI
=============================================================================
Pipeline (IPO framework from the training kit):

    INPUT   : Load the Iris dataset, inspect it, apply feature scaling
    PROCESS : Shuffled 80/20 train-test split, K-Nearest Neighbors (KNN)
    OUTPUT  : Confusion matrix, precision / recall / F1 score

Run:
    python classify_iris.py
Outputs are saved to the ./outputs folder.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20          # 80% train / 20% test
DEFAULT_K = 5             # K=5 as shown in the workflow slide
K_RANGE = range(1, 31)    # range searched for the "elbow"
OUT_DIR = Path(__file__).parent / "outputs"


# --------------------------------------------------------------------------
# INPUT: load and understand the dataset
# --------------------------------------------------------------------------
def load_data():
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    class_names = [str(name) for name in iris.target_names]

    print("=" * 70)
    print("STEP 1 - LOAD & UNDERSTAND THE DATASET (Iris Benchmark)")
    print("=" * 70)
    print(f"Samples    : {X.shape[0]}")
    print(f"Dimensions : {X.shape[1]} -> {list(X.columns)}")
    print(f"Classes    : {len(class_names)} -> {class_names}")
    print("\nClass balance:")
    print(y.map(dict(enumerate(class_names))).value_counts().to_string())
    print("\nFirst 5 rows:")
    print(pd.concat([X, y.rename('target')], axis=1).head().to_string())
    print("\nSummary statistics (raw):")
    print(X.describe().round(2).to_string())
    print("\nMissing values:", int(X.isna().sum().sum()))
    return X, y, class_names


# --------------------------------------------------------------------------
# PROCESS: split -> scale -> tune K -> train
# --------------------------------------------------------------------------
def split_data(X, y):
    """Shuffle, then split 80/20. Stratify keeps class proportions equal."""
    print("\n" + "=" * 70)
    print("STEP 2 - TRAIN / TEST SPLIT (shuffled, 80/20)")
    print("=" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        shuffle=True,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Training set : {len(X_train)} samples")
    print(f"Test set     : {len(X_test)} samples (locked away until final validation)")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    The Gatekeeper Rule: StandardScaler -> mean = 0, variance = 1.
    Fit ONLY on the training data, then apply to the test data, so no
    information from the test set leaks into training.
    """
    print("\n" + "=" * 70)
    print("STEP 3 - FEATURE SCALING (StandardScaler)")
    print("=" * 70)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("Train mean after scaling :", np.round(X_train_s.mean(axis=0), 3))
    print("Train std  after scaling :", np.round(X_train_s.std(axis=0), 3))
    return X_train_s, X_test_s, scaler


def tune_k(X_train_s, y_train):
    """
    Choose K with 5-fold cross-validation on the TRAINING set only
    (never touching the test set). Plots error rate vs K - the "elbow".
    """
    from sklearn.model_selection import cross_val_score

    print("\n" + "=" * 70)
    print("STEP 4 - TUNING THE ENGINE: CHOOSING K")
    print("=" * 70)
    errors = []
    for k in K_RANGE:
        model = KNeighborsClassifier(n_neighbors=k)
        acc = cross_val_score(model, X_train_s, y_train, cv=5, scoring="accuracy").mean()
        errors.append(1 - acc)

    best_k = list(K_RANGE)[int(np.argmin(errors))]
    print(f"Lowest cross-validated error at K = {best_k} "
          f"(error = {min(errors):.4f})")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_RANGE), errors, marker="o", color="#003a70")
    ax.scatter([best_k], [min(errors)], s=300, facecolors="none",
               edgecolors="#ff4f00", linewidths=3, label=f"Elbow (K={best_k})")
    ax.set_xlabel("K value")
    ax.set_ylabel("Error rate (5-fold CV)")
    ax.set_title("Tuning the Engine: Choosing K")
    ax.annotate("K=1: noise / overfitting", (1, errors[0]),
                textcoords="offset points", xytext=(10, 10))
    ax.annotate("Large K: generic / underfitting", (K_RANGE[-1], errors[-1]),
                textcoords="offset points", xytext=(-150, 10))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "k_selection.png", dpi=150)
    plt.close(fig)
    return best_k


def train_model(X_train_s, y_train, k):
    """Workflow: INSTANTIATE -> FIT -> (PREDICT happens in evaluate)."""
    print("\n" + "=" * 70)
    print(f"STEP 5 - TRAIN KNN MODEL (n_neighbors={k})")
    print("=" * 70)
    model = KNeighborsClassifier(n_neighbors=k)   # INSTANTIATE (build the frame)
    model.fit(X_train_s, y_train)                  # FIT (memorize the map)
    print("Model trained.")
    return model


# --------------------------------------------------------------------------
# OUTPUT: validate with confusion matrix and F1
# --------------------------------------------------------------------------
def evaluate(model, X_test_s, y_test, class_names, label="Model"):
    predictions = model.predict(X_test_s)          # PREDICT (apply logic)

    print("\n" + "=" * 70)
    print(f"STEP 6 - OUTPUT VALIDATION ({label})")
    print("=" * 70)
    acc = accuracy_score(y_test, predictions)
    f1_macro = f1_score(y_test, predictions, average="macro")
    print(f"Accuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")

    cm = confusion_matrix(y_test, predictions)
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=class_names, columns=class_names).to_string())

    print("\nClassification report (precision / recall / F1 per class):")
    print(classification_report(y_test, predictions, target_names=class_names))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - KNN on Iris")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    return predictions, acc, f1_macro


def compare_with_other_models(X_train_s, X_test_s, y_train, y_test):
    """Conclusion slide: 'experiment ... like comparing different algorithms'."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    print("\n" + "=" * 70)
    print("BONUS - COMPARING ALGORITHMS")
    print("=" * 70)
    models = {
        "KNN (K=5)": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }
    rows = []
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        p = m.predict(X_test_s)
        rows.append({"model": name,
                     "accuracy": round(accuracy_score(y_test, p), 4),
                     "f1_macro": round(f1_score(y_test, p, average="macro"), 4)})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


def predict_new_sample(model, scaler, class_names):
    """Test the model with completely new data (scale it the same way!)."""
    print("\n" + "=" * 70)
    print("BONUS - PREDICT A BRAND-NEW FLOWER")
    print("=" * 70)
    new_flower = pd.DataFrame(
        [[5.9, 3.0, 5.1, 1.8]],
        columns=["sepal length (cm)", "sepal width (cm)",
                 "petal length (cm)", "petal width (cm)"],
    )
    pred = model.predict(scaler.transform(new_flower))[0]
    print(f"Measurements: {new_flower.values.tolist()[0]}")
    print(f"Predicted species: {class_names[pred]}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # INPUT
    X, y, class_names = load_data()

    # PROCESS
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    best_k = tune_k(X_train_s, y_train)

    baseline = train_model(X_train_s, y_train, DEFAULT_K)
    evaluate(baseline, X_test_s, y_test, class_names, label=f"Baseline K={DEFAULT_K}")

    if best_k != DEFAULT_K:
        tuned = train_model(X_train_s, y_train, best_k)
        evaluate(tuned, X_test_s, y_test, class_names, label=f"Tuned K={best_k}")
        final_model = tuned
    else:
        final_model = baseline

    # Extras suggested by the Conclusion slide
    comparison = compare_with_other_models(X_train_s, X_test_s, y_train, y_test)
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    predict_new_sample(final_model, scaler, class_names)

    print(f"\nDone. Plots and tables saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
