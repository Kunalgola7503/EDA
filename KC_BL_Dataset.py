# ---- Import required libraries ----
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Configure display options ----
pd.set_option('display.max_columns', None)   # Show all columns when printing DataFrames
sns.set_style("whitegrid")                   # Set Seaborn style for plots

# ---- Load dataset ----
# NOTE: Update the file path as needed (currently points to local path)
df = pd.read_csv(r"C:\Users\user\Downloads\KC_BL_DATASET.csv")

# ---- Quick look at dataset ----
print(df.head())                             # First 5 rows
print("Shape:", df.shape)                    # Dataset dimensions
print("\nInfo:")
print(df.info())                             # Column data types and non-null counts
print("\nSummary Statistics:")
print(df.describe(include="all"))            # Summary stats for all columns

# ---- Missing values & duplicates check ----
print("Missing Values:\n", df.isnull().sum())
print("\nMissing %:\n", (df.isnull().mean()*100).round(2))
print("\nDuplicate Rows:", df.duplicated().sum())

# ---- Numerical feature distributions ----
df.hist(figsize=(12,8), bins=30)
plt.suptitle("Numerical Feature Distributions", fontsize=14)
plt.show()

# ---- Categorical feature distributions ----
for col in df.select_dtypes(include=["object"]).columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# ---- Correlation heatmap for numeric features ----
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ---- Boxplots: categorical vs numerical features ----
categorical_cols = df.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    for num_col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(6,4))
        sns.boxplot(x=col, y=num_col, data=df, palette="Set3")
        plt.title(f"{num_col} by {col}")
        plt.xticks(rotation=45)
        plt.show()

# ---- Boxplots: univariate analysis for numerical columns ----
for col in df.select_dtypes(include=np.number).columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col], color="orange")
    plt.title(f"Boxplot of {col}")
    plt.show()

# ---- Final EDA Insights ----
print("\n========== Final EDA Insights ==========")
print(f"1. Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
print(f"2. Missing values detected in {df.isnull().any().sum()} columns.")
print(f"3. Duplicate rows: {df.duplicated().sum()}")
print(f"4. Numerical features show varied distributions (see histograms).")
print(f"5. Some categorical features are imbalanced (count plots).")
print(f"6. Correlations highlight relationships between numeric variables.")
print(f"7. Outliers present in certain numerical columns (boxplots).")
