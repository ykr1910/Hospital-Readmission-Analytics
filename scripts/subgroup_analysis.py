
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(PROJECT_DIR, "data", "ml")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "best_model.joblib")
THRESHOLD_PATH = os.path.join(
    PROJECT_DIR, "models", "decision_threshold.json"
)
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

os.makedirs(REPORT_DIR, exist_ok=True)

# Small groups need cautious interpretation.
MIN_GROUP_SIZE = 100
MIN_CLASS_COUNT = 20

# --------------------------------------------------
# 2. Load model, threshold and datasets
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_info = json.load(file)

threshold = float(threshold_info["threshold"])

X = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
y = pd.read_csv(os.path.join(DATA_DIR, "target.csv"))["target"]
groups = pd.read_csv(
    os.path.join(DATA_DIR, "patient_groups.csv")
)["patient_key"]

if not (len(X) == len(y) == len(groups)):
    raise ValueError(
        "Features, target and patient groups have different lengths."
    )

if not set(y.dropna().unique()).issubset({0, 1}):
    raise ValueError("Target must contain binary labels 0 and 1.")

print("Loaded model:", type(model.named_steps["classifier"]).__name__)
print("Model name:", threshold_info.get("model_name", "Unknown"))
print("Decision threshold:", threshold)
print("Total encounters:", len(X))

# --------------------------------------------------
# 3. Reproduce the exact patient-aware test split
# --------------------------------------------------

split1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42,
)

train_idx, temp_idx = next(
    split1.split(X, y, groups)
)

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42,
)

val_rel_idx, test_rel_idx = next(
    split2.split(X_temp, y_temp, groups_temp)
)

X_test = X_temp.iloc[test_rel_idx].copy().reset_index(drop=True)
y_test = y_temp.iloc[test_rel_idx].reset_index(drop=True)
groups_test = (
    groups_temp.iloc[test_rel_idx].reset_index(drop=True)
)

# Verify there is no patient overlap with train/validation.
groups_train = set(groups.iloc[train_idx])
groups_val = set(groups_temp.iloc[val_rel_idx])
groups_test_set = set(groups_test)

assert groups_train.isdisjoint(groups_val)
assert groups_train.isdisjoint(groups_test_set)
assert groups_val.isdisjoint(groups_test_set)

print("\nTest-set size:", len(X_test))
print("Unique test patients:", groups_test.nunique())
print("Patient leakage check: passed")

# --------------------------------------------------
# 4. Generate predictions on the test set
# --------------------------------------------------

probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

evaluation = X_test.copy()
evaluation["actual"] = y_test.to_numpy()
evaluation["probability"] = probabilities
evaluation["predicted"] = predictions

# --------------------------------------------------
# 5. Calculate metrics for each subgroup
# --------------------------------------------------

def evaluate_subgroup(frame, dimension, subgroup):
    actual = frame["actual"].to_numpy(dtype=int)
    predicted = frame["predicted"].to_numpy(dtype=int)
    probability = frame["probability"].to_numpy(dtype=float)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    positives = int(actual.sum())
    negatives = int(len(actual) - positives)

    warnings = []

    if len(frame) < MIN_GROUP_SIZE:
        warnings.append(
            f"Small sample: fewer than {MIN_GROUP_SIZE} encounters"
        )

    if positives < MIN_CLASS_COUNT:
        warnings.append(
            f"Few positive cases: fewer than {MIN_CLASS_COUNT}"
        )

    if negatives < MIN_CLASS_COUNT:
        warnings.append(
            f"Few negative cases: fewer than {MIN_CLASS_COUNT}"
        )

    if len(np.unique(actual)) == 2:
        auc = roc_auc_score(actual, probability)
    else:
        auc = np.nan
        warnings.append(
            "ROC-AUC unavailable: only one actual class present"
        )

    return {
        "dimension": dimension,
        "subgroup": str(subgroup),
        "sample_count": len(frame),
        "unique_patients": groups_test.iloc[
            frame.index
        ].nunique(),
        "actual_readmissions": positives,
        "readmission_prevalence": (
            positives / len(frame) if len(frame) else np.nan
        ),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
        "roc_auc": auc,
        "brier_score": brier_score_loss(actual, probability),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "specificity": (
            tn / (tn + fp) if (tn + fp) else np.nan
        ),
        "warnings": "; ".join(warnings) if warnings else "None",
    }


results = []

# Overall test-set metrics as a baseline.
results.append(
    evaluate_subgroup(evaluation, "overall", "All test encounters")
)

# Evaluate age and gender separately.
for dimension in ["age_group", "gender"]:
    if dimension not in evaluation.columns:
        print(f"Warning: {dimension} not found; skipping.")
        continue

    evaluation[dimension] = (
        evaluation[dimension]
        .fillna("Missing")
        .astype(str)
    )

    for subgroup, frame in evaluation.groupby(
        dimension,
        sort=True,
        dropna=False,
    ):
        results.append(
            evaluate_subgroup(frame, dimension, subgroup)
        )

# --------------------------------------------------
# 6. Save and display results
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = os.path.join(
    REPORT_DIR, "subgroup_performance.csv"
)

results_df.to_csv(output_path, index=False)

display_columns = [
    "dimension",
    "subgroup",
    "sample_count",
    "unique_patients",
    "actual_readmissions",
    "readmission_prevalence",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "brier_score",
    "specificity",
    "warnings",
]

print("\nSubgroup performance results:")
print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "readmission_prevalence": lambda x: f"{x:.4f}",
            "precision": lambda x: f"{x:.4f}",
            "recall": lambda x: f"{x:.4f}",
            "f1": lambda x: f"{x:.4f}",
            "roc_auc": lambda x: (
                f"{x:.4f}" if pd.notna(x) else "N/A"
            ),
            "brier_score": lambda x: f"{x:.4f}",
            "specificity": lambda x: (
                f"{x:.4f}" if pd.notna(x) else "N/A"
            ),
        },
    )
)

print("\nConfusion matrix counts by subgroup:")
print(
    results_df[
        [
            "dimension",
            "subgroup",
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
        ]
    ].to_string(index=False)
)

print("\nSaved:", output_path)
print("Subgroup analysis completed.")
print(
    "Reminder: subgroup results are descriptive, "
    "not proof of bias or clinical validity."
)
