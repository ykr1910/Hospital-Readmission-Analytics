
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from scipy import sparse

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "best_model.joblib"
)

FEATURES_PATH = os.path.join(
    PROJECT_DIR, "data", "ml", "features.csv"
)

REPORT_DIR = os.path.join(PROJECT_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load trained pipeline
# --------------------------------------------------

pipeline = joblib.load(MODEL_PATH)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

feature_names = np.asarray(
    preprocessor.get_feature_names_out()
)

print("Loaded model:", type(classifier).__name__)
print("Transformed feature count:", len(feature_names))

# --------------------------------------------------
# 3. XGBoost built-in feature importance
# --------------------------------------------------

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": classifier.feature_importances_,
}).sort_values("importance", ascending=False)

importance_df.to_csv(
    os.path.join(REPORT_DIR, "feature_importance.csv"),
    index=False,
)

top_features = importance_df.head(15).sort_values(
    "importance", ascending=True
)

fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(top_features["feature"], top_features["importance"])
ax.set_xlabel("XGBoost Feature Importance")
ax.set_ylabel("Feature")
ax.set_title("Top 15 Features for Readmission Prediction")
fig.tight_layout()

fig.savefig(
    os.path.join(REPORT_DIR, "feature_importance.png"),
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved built-in feature importance reports.")

# --------------------------------------------------
# 4. Load and transform data
# --------------------------------------------------

X = pd.read_csv(FEATURES_PATH)

sample_size = min(1000, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42,
)

X_transformed = preprocessor.transform(X_sample)

if sparse.issparse(X_transformed):
    X_transformed = X_transformed.toarray()

X_transformed = np.asarray(X_transformed)

if X_transformed.shape[1] != len(feature_names):
    raise ValueError(
        "Transformed data and feature names do not match."
    )

print("Rows selected for SHAP:", len(X_transformed))

# --------------------------------------------------
# 5. Calculate SHAP values
# --------------------------------------------------

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

if isinstance(shap_values, list):
    shap_values = shap_values[-1]

shap_values = np.asarray(shap_values)

if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]

if shap_values.shape != X_transformed.shape:
    raise ValueError(
        f"Unexpected SHAP shape {shap_values.shape}; "
        f"expected {X_transformed.shape}."
    )

# Save SHAP values for each sampled record.
pd.DataFrame(
    shap_values,
    columns=feature_names,
).to_csv(
    os.path.join(REPORT_DIR, "shap_values_sample.csv"),
    index=False,
)

# --------------------------------------------------
# 6. Calculate global SHAP importance
# --------------------------------------------------

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

shap_importance.to_csv(
    os.path.join(REPORT_DIR, "shap_importance.csv"),
    index=False,
)

print("\nTop 15 SHAP features:")
print(shap_importance.head(15).to_string(index=False))

# --------------------------------------------------
# 7. SHAP summary plot using Matplotlib
# --------------------------------------------------

# Display the 20 most important features.
top_indices = (
    shap_importance.head(20)["feature"]
    .map({name: i for i, name in enumerate(feature_names)})
    .to_numpy()
)

# Order features from least to most important.
top_indices = top_indices[::-1]

fig, ax = plt.subplots(figsize=(12, 9))

rng = np.random.default_rng(42)

for position, feature_index in enumerate(top_indices):
    values = shap_values[:, feature_index]
    feature_data = X_transformed[:, feature_index]

    # Small vertical jitter separates overlapping observations.
    jitter = rng.uniform(-0.22, 0.22, size=len(values))

    ax.scatter(
        values,
        position + jitter,
        c=feature_data,
        cmap="coolwarm",
        s=10,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )

ax.axvline(0, color="gray", linewidth=1)
ax.set_yticks(range(len(top_indices)))
ax.set_yticklabels(feature_names[top_indices])
ax.set_xlabel("SHAP value (impact on model output)")
ax.set_ylabel("Feature")
ax.set_title("SHAP Summary: Readmission Prediction")
ax.grid(axis="x", linestyle="--", alpha=0.25)

fig.tight_layout()

summary_path = os.path.join(
    REPORT_DIR, "shap_summary.png"
)

fig.savefig(
    summary_path,
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved SHAP summary:", summary_path)

# --------------------------------------------------
# 8. SHAP global importance plot using Matplotlib
# --------------------------------------------------

top_importance = (
    shap_importance.head(15)
    .sort_values("mean_absolute_shap", ascending=True)
)

fig, ax = plt.subplots(figsize=(11, 8))

ax.barh(
    top_importance["feature"],
    top_importance["mean_absolute_shap"],
)

ax.set_xlabel("Mean absolute SHAP value")
ax.set_ylabel("Feature")
ax.set_title("Mean Absolute SHAP Importance")
ax.grid(axis="x", linestyle="--", alpha=0.25)

fig.tight_layout()

importance_path = os.path.join(
    REPORT_DIR, "shap_importance.png"
)

fig.savefig(
    importance_path,
    dpi=250,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved SHAP importance:", importance_path)

# --------------------------------------------------
# 9. Completion
# --------------------------------------------------

print("\nExplainability analysis completed.")
print("Generated files:")

for filename in [
    "feature_importance.csv",
    "feature_importance.png",
    "shap_importance.csv",
    "shap_summary.png",
    "shap_importance.png",
    "shap_values_sample.csv",
]:
    print(" -", os.path.join(REPORT_DIR, filename))
