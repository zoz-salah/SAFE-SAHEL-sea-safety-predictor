"""
Safe Sahel - Step 2: Exploratory Data Analysis (EDA)
=====================================================
Simple, beginner-friendly EDA: look at the data, check for missing values,
and plot a few charts to understand what makes swimming unsafe.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/home/claude/safe_sahel/data/swim_safety_data.csv")

print("=== Basic Info ===")
print(df.info())
print("\n=== Missing values ===")
print(df.isnull().sum())
print("\n=== Summary statistics ===")
print(df.describe())

# Map 0/1 label to words for nicer charts
df["safety_label"] = df["is_safe"].map({1: "Safe", 0: "Not Safe"})

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1. Wave height vs safety
sns.boxplot(data=df, x="safety_label", y="wave_height_m", ax=axes[0, 0], palette="Blues")
axes[0, 0].set_title("Wave Height by Safety Label")

# 2. Current strength vs safety
sns.boxplot(data=df, x="safety_label", y="current_strength", ax=axes[0, 1], palette="Oranges")
axes[0, 1].set_title("Current Strength by Safety Label")

# 3. Wind speed vs safety
sns.boxplot(data=df, x="safety_label", y="wind_speed_kmh", ax=axes[1, 0], palette="Greens")
axes[1, 0].set_title("Wind Speed by Safety Label")

# 4. Safety proportion by month
month_safety = df.groupby("month")["is_safe"].mean().reset_index()
month_safety["month_name"] = month_safety["month"].map({7: "July", 8: "August", 9: "September"})
sns.barplot(data=month_safety, x="month_name", y="is_safe", ax=axes[1, 1], palette="Purples")
axes[1, 1].set_title("Proportion of Safe Days by Month")
axes[1, 1].set_ylabel("Proportion Safe")

plt.tight_layout()
plt.savefig("/home/claude/safe_sahel/data/eda_charts.png", dpi=120)
print("\nSaved charts to data/eda_charts.png")

# Correlation heatmap of numeric features
plt.figure(figsize=(7, 5))
numeric_cols = ["sea_temp_c", "wave_height_m", "wind_speed_kmh", "current_strength",
                 "uv_index", "historical_incidents", "is_safe"]
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("/home/claude/safe_sahel/data/correlation_heatmap.png", dpi=120)
print("Saved correlation heatmap to data/correlation_heatmap.png")
