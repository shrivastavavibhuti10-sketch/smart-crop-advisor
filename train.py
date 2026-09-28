import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv(
    "data/crop_data.csv"
)


# ============================================================
# INPUT FEATURES
# ============================================================

X = data[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]


# ============================================================
# TARGET
# ============================================================

y = data["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST MODEL
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/crop_model.pkl"
)


# ============================================================
# SAVE PERFORMANCE RESULTS
# ============================================================

performance = {
    "accuracy": accuracy,
    "precision": report["weighted avg"]["precision"],
    "recall": report["weighted avg"]["recall"],
    "f1_score": report["weighted avg"]["f1-score"],
    "classes": list(model.classes_),
    "confusion_matrix": cm.tolist()
}


joblib.dump(
    performance,
    "models/model_performance.pkl"
)


print(
    "\nModel saved successfully!"
)

print(
    "Performance metrics saved successfully!"
)