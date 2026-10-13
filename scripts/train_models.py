
import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
)

from xgboost import XGBClassifier

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

load_dotenv()

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

X = pd.read_csv("data/ml/features.csv")
y = pd.read_csv("data/ml/target.csv")["target"]
groups = pd.read_csv("data/ml/patient_groups.csv")["patient_key"]

# --------------------------------------------------
# 2. Patient-aware train / validation / test split
# --------------------------------------------------

split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, temp_idx = next(split1.split(X, y, groups))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
groups_train = groups.iloc[train_idx]

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)
val_rel_idx, test_rel_idx = next(
    split2.split(X_temp, y_temp, groups_temp)
)

X_val = X_temp.iloc[val_rel_idx]
y_val = y_temp.iloc[val_rel_idx]

X_test = X_temp.iloc[test_rel_idx]
y_test = y_temp.iloc[test_rel_idx]

groups_val = groups_temp.iloc[val_rel_idx]
groups_test = groups_temp.iloc[test_rel_idx]

assert set(groups_train).isdisjoint(set(groups_val))
assert set(groups_train).isdisjoint(set(groups_test))
assert set(groups_val).isdisjoint(set(groups_test))

print("Split sizes:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))
print("Patient leakage check: passed")

# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

categorical = [
    "age_group",
    "gender",
    "medical_specialty",
]

numerical = [
    "admission_type_id",
    "time_in_hospital",
    "prior_visits",
    "diagnosis_count",
    "medication_count",
]

def create_preprocessor():
    return ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),

        ("numerical", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
    ])

# --------------------------------------------------
# 4. Define models
# --------------------------------------------------

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    ),
}

# --------------------------------------------------
# 5. Train and select thresholds using validation
# --------------------------------------------------

mlflow.set_experiment("Hospital_Readmission_Phase_II")

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(
        y_val, probabilities
    )

    # Select threshold maximizing F1 on validation data.
    f1_values = (
        2 * precisions[:-1] * recalls[:-1]
        / np.maximum(precisions[:-1] + recalls[:-1], 1e-12)
    )

    best_idx = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_idx])

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    metrics = {
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_val, probabilities),
        "brier_score": brier_score_loss(y_val, probabilities),
    }

    trained_models[name] = {
        "pipeline": pipeline,
        "threshold": best_threshold,
        "validation_metrics": metrics,
    }

    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_param("split_strategy", "patient_group_aware")
        mlflow.log_param("threshold_selection", "validation_max_f1")
        mlflow.log_param("classification_threshold", best_threshold)
        mlflow.log_metrics({
            f"validation_{key}": value
            for key, value in metrics.items()
        })

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format="cloudpickle",
        )

    results.append({
        "model": name,
        "threshold": best_threshold,
        **metrics,
    })

    print("Selected threshold:", round(best_threshold, 4))
    for metric, value in metrics.items():
        print(f"Validation {metric}: {value:.4f}")

# --------------------------------------------------
# 6. Compare models and select by validation F1
# --------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    "f1", ascending=False
)

results_df.to_csv(
    "reports/model_comparison.csv", index=False
)

print("\nValidation comparison:")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
best_info = trained_models[best_name]

best_pipeline = best_info["pipeline"]
best_threshold = best_info["threshold"]

print("\nSelected model:", best_name)
print("Selected threshold:", round(best_threshold, 4))

# --------------------------------------------------
# 7. Final evaluation on the untouched test set
# --------------------------------------------------

test_probabilities = best_pipeline.predict_proba(X_test)[:, 1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = {
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "roc_auc": roc_auc_score(y_test, test_probabilities),
    "brier_score": brier_score_loss(
        y_test, test_probabilities
    ),
}

print("\nFinal test results:")
print("Model:", best_name)
print("Threshold:", round(best_threshold, 4))

for metric, value in test_metrics.items():
    print(f"Test {metric}: {value:.4f}")

with mlflow.start_run(run_name=f"{best_name}_final_test"):
    mlflow.log_param("selected_model", best_name)
    mlflow.log_param("classification_threshold", best_threshold)
    mlflow.log_param("evaluation_set", "untouched_test")
    mlflow.log_metrics({
        f"test_{key}": value
        for key, value in test_metrics.items()
    })

# --------------------------------------------------
# 8. Save model, threshold and test metrics
# --------------------------------------------------

joblib.dump(
    best_pipeline,
    "models/best_model.joblib",
)

with open("models/decision_threshold.json", "w") as file:
    json.dump({
        "model_name": best_name,
        "threshold": best_threshold,
        "positive_class": "Readmitted within 30 days",
        "target_mapping": {
            "<30": 1,
            ">30": 0,
            "NO": 0,
        },
    }, file, indent=4)

with open("reports/test_metrics.json", "w") as file:
    json.dump({
        "model_name": best_name,
        "threshold": best_threshold,
        **test_metrics,
    }, file, indent=4)

print("\nSaved:")
print("models/best_model.joblib")
print("models/decision_threshold.json")
print("reports/model_comparison.csv")
print("reports/test_metrics.json")
print("\nTraining pipeline completed successfully.")
