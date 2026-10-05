"""
Week 2 Internship Project: Titanic EDA and Visualization
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = sns.load_dataset("titanic")

print("Shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDescriptive statistics:")
print(df.describe(include="all").T)

# Cleaning and transformations
df["age_filled"] = df["age"].fillna(df["age"].median())
df["embarked_filled"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["survival_status"] = df["survived"].map({0:"Not Survived",1:"Survived"})

print("\nOverall survival rate:", df["survived"].mean())
print("\nSurvival rate by gender:")
print(df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:")
print(df.groupby("class", observed=False)["survived"].mean())

# 1. Survival distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="survival_status")
plt.title("Titanic Survival Distribution")
plt.xlabel("Outcome"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/01_survival_distribution.png",dpi=180); plt.close()

# 2. Gender distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="sex")
plt.title("Passenger Distribution by Gender")
plt.xlabel("Gender"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/02_gender_distribution.png",dpi=180); plt.close()

# 3. Survival by gender
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="sex", hue="survival_status")
plt.title("Survival Outcome by Gender")
plt.xlabel("Gender"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/03_survival_by_gender.png",dpi=180); plt.close()

# 4. Class distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="class", order=["First","Second","Third"])
plt.title("Passenger Distribution by Passenger Class")
plt.xlabel("Passenger Class"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/04_class_distribution.png",dpi=180); plt.close()

# 5. Survival by class
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="class", hue="survival_status", order=["First","Second","Third"])
plt.title("Survival Outcome by Passenger Class")
plt.xlabel("Passenger Class"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/05_survival_by_class.png",dpi=180); plt.close()

# 6. Age distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="age_filled", bins=20, kde=True)
plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/06_age_distribution.png",dpi=180); plt.close()

# 7. Fare distribution
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="fare", bins=30, kde=True)
plt.title("Fare Distribution")
plt.xlabel("Fare"); plt.ylabel("Number of Passengers")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/07_fare_distribution.png",dpi=180); plt.close()

# 8. Age vs fare
plt.figure(figsize=(9,6))
sns.scatterplot(data=df, x="age_filled", y="fare", hue="survival_status", alpha=0.65)
plt.title("Age vs Fare by Survival Status")
plt.xlabel("Age"); plt.ylabel("Fare")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/08_age_vs_fare.png",dpi=180); plt.close()

# 9. Correlation heatmap
numeric_cols=["survived","pclass","age","sibsp","parch","fare"]
plt.figure(figsize=(9,7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Titanic Numeric Feature Correlation Matrix")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/09_correlation_heatmap.png",dpi=180); plt.close()

# 10. Survival rate by gender
gender_survival=df.groupby("sex")["survived"].mean().mul(100)
plt.figure(figsize=(8,5))
sns.barplot(x=gender_survival.index,y=gender_survival.values)
plt.title("Survival Rate by Gender")
plt.xlabel("Gender"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/10_survival_rate_gender.png",dpi=180); plt.close()

# 11. Survival rate by class
class_survival=df.groupby("class",observed=False)["survived"].mean().mul(100)
plt.figure(figsize=(8,5))
sns.barplot(x=class_survival.index,y=class_survival.values,order=["First","Second","Third"])
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class"); plt.ylabel("Survival Rate (%)")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/11_survival_rate_class.png",dpi=180); plt.close()

print("\nEDA completed. Charts saved in:", OUTPUT_DIR)
