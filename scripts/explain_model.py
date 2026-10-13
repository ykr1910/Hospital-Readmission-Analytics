
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the selected model
model_path = "models/best_model.joblib"
pipeline = joblib.load(model_path)

# Extract preprocessing and trained XGBoost model
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# Get feature names after one-hot encoding
feature_names = preprocessor.get_feature_names_out()

# Get XGBoost feature importance scores
importance_scores = classifier.feature_importances_

if len(feature_names) != len(importance_scores):
    raise ValueError(
        "Feature names and importance scores have different lengths."
    )

# Create a ranked feature-importance table
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_scores,
}).sort_values("importance", ascending=False)

# Save results
os.makedirs("reports", exist_ok=True)

importance_df.to_csv(
    "reports/feature_importance.csv",
    index=False,
)

print("\nTop 15 features:")
print(importance_df.head(15).to_string(index=False))

# Plot the top 15 features
top_features = importance_df.head(15).sort_values(
    "importance", ascending=True
)

plt.figure(figsize=(10, 7))
plt.barh(
    top_features["feature"],
    top_features["importance"],
)
plt.xlabel("XGBoost Feature Importance")
plt.ylabel("Feature")
plt.title("Top 15 Features for 30-Day Readmission Prediction")
plt.tight_layout()

plt.savefig(
    "reports/feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved:")
print("reports/feature_importance.csv")
print("reports/feature_importance.png")
