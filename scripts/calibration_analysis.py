
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Load saved model and selected threshold
model = joblib.load("models/best_model.joblib")

# Reproduce the same patient-aware split used in training
from sklearn.model_selection import GroupShuffleSplit

X = pd.read_csv("data/ml/features.csv")
y = pd.read_csv("data/ml/target.csv")["target"]
groups = pd.read_csv("data/ml/patient_groups.csv")["patient_key"]

split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)
train_idx, temp_idx = next(split1.split(X, y, groups))

X_temp = X.iloc[temp_idx]
y_temp = y.iloc[temp_idx]
groups_temp = groups.iloc[temp_idx]

split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)
val_rel_idx, test_rel_idx = next(
    split2.split(X_temp, y_temp, groups_temp)
)

X_test = X_temp.iloc[test_rel_idx]
y_test = y_temp.iloc[test_rel_idx]

# Generate probabilities on the held-out test set
probabilities = model.predict_proba(X_test)[:, 1]

# Compute calibration curve
observed_rate, mean_probability = calibration_curve(
    y_test,
    probabilities,
    n_bins=10,
    strategy="quantile",
)

brier = brier_score_loss(y_test, probabilities)

print("Test-set calibration analysis")
print("Brier score:", round(brier, 4))
print("\nCalibration bins:")
for i, (predicted, observed) in enumerate(
    zip(mean_probability, observed_rate), start=1
):
    print(
        f"Bin {i}: mean predicted={predicted:.4f}, "
        f"observed rate={observed:.4f}"
    )

# Plot observed rates against predicted probabilities
os.makedirs("reports", exist_ok=True)

plt.figure(figsize=(7, 6))
plt.plot(
    mean_probability,
    observed_rate,
    marker="o",
    label="XGBoost",
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)
plt.xlabel("Mean predicted probability")
plt.ylabel("Observed readmission rate")
plt.title("XGBoost Calibration Curve — Test Set")
plt.legend()
plt.tight_layout()

plt.savefig(
    "reports/calibration_curve.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("\nSaved reports/calibration_curve.png")
