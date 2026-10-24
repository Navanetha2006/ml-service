"""Train and compare models, then save the best one.

Leakage prevention (see README):
  1. Split into train/test FIRST, before looking at any statistics.
  2. Scaling lives inside a Pipeline, so it is fitted on training data only.
  3. Models are compared with cross-validation on the TRAIN set only.
  4. The test set is used exactly once, for the final chosen model.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
]


def load_data():
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    df.columns = [c.replace(" ", "_") for c in df.columns]
    X = df[FEATURES]
    y = df["target"]  # 0 = malignant, 1 = benign
    return X, y


def run(out_dir="artifacts"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_data()

    # --- data understanding ---
    print("Rows:", len(X))
    print("Missing values:", int(X.isna().sum().sum()))
    print("Duplicate rows:", int(X.duplicated().sum()))
    print("Class balance:", y.value_counts(normalize=True).round(2).to_dict())

    # --- split BEFORE anything is fitted ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "knn": KNeighborsClassifier(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # --- compare with CV on training data only ---
    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1")
        rows.append({"model": name, "cv_f1_mean": scores.mean(), "cv_f1_std": scores.std()})
    results = pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False)
    print(results.round(4).to_string(index=False))
    results.to_csv(out / "comparison.csv", index=False)

    # --- pick winner, fit on train, test ONCE ---
    best_name = results.iloc[0]["model"]
    best = Pipeline([("scaler", StandardScaler()), ("model", models[best_name])])
    best.fit(X_train, y_train)
    pred = best.predict(X_test)
    test_metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
    }
    print("Best model:", best_name)
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    joblib.dump(best, out / "model.joblib")
    meta = {
        "model_name": best_name,
        "features": FEATURES,
        "classes": {"0": "malignant", "1": "benign"},
        "sklearn_version": sklearn.__version__,
        "test_metrics": test_metrics,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
