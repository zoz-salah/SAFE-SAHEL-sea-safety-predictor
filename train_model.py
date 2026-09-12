import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

df = pd.read_csv("/home/claude/safe_sahel/data/swim_safety_data.csv")

df["sea_roughness_score"] = (
    df["wave_height_m"] * 2 + df["wind_speed_kmh"] / 10 + df["current_strength"]
)

df["recent_incident_flag"] = (df["historical_incidents"] > 0).astype(int)

numeric_features = [
    "sea_temp_c", "wave_height_m", "wind_speed_kmh", "current_strength",
    "uv_index", "historical_incidents", "sea_roughness_score", "recent_incident_flag"
]
categorical_features = ["weather_condition", "crowd_level"]

X = df[numeric_features + categorical_features]
y = df["is_safe"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Model Evaluation on Test Set ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall   : {recall_score(y_test, y_pred):.2%}")
print(f"F1 Score : {f1_score(y_test, y_pred):.2%}")
print("\nConfusion Matrix (rows=actual, cols=predicted) [Not Safe, Safe]:")
print(confusion_matrix(y_test, y_pred))
print("\nFull Report:")
print(classification_report(y_test, y_pred, target_names=["Not Safe", "Safe"]))

feature_names = (
    numeric_features +
    list(model.named_steps["preprocessor"]
         .named_transformers_["cat"]
         .get_feature_names_out(categorical_features))
)
coefs = model.named_steps["classifier"].coef_[0]
importance = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
importance["abs_coef"] = importance["coefficient"].abs()
importance = importance.sort_values("abs_coef", ascending=False)
print("\n=== Feature Importance (higher |coefficient| = more influence) ===")
print(importance[["feature", "coefficient"]].to_string(index=False))

joblib.dump(model, "/home/claude/safe_sahel/model/safe_sahel_model.pkl")
print("\nModel saved to model/safe_sahel_model.pkl")
