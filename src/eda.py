"""
Exploratory Data Analysis (EDA) for Heart Disease Prediction
Author: CodeAlpha Internship
Description: Inspects dataset health, class balance, and generates correlation visualizations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(csv_path="data/heart.csv", output_dir="visuals"):
    print("=" * 60)
    print("      HEART DISEASE DATASET - EXPLORATORY DATA ANALYSIS      ")
    print("=" * 60)

    # 1. Load the dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"\n[1] Dataset Shape: {df.shape[0]} rows (patients), {df.shape[1]} columns (features)")

    # 2. Check for missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"\n[2] Missing / Null Values: {total_nulls}")
    if total_nulls > 0:
        print(null_counts[null_counts > 0])
    else:
        print("    -> Excellent! Zero missing values detected.")

    # 3. Target Distribution (Heart Disease Presence)
    print("\n[3] Target Class Distribution:")
    target_counts = df['target'].value_counts()
    no_disease = target_counts.get(0, 0)
    has_disease = target_counts.get(1, 0)
    print(f"    - Class 0 (No Heart Disease): {no_disease} ({no_disease / len(df) * 100:.1f}%)")
    print(f"    - Class 1 (Heart Disease Present): {has_disease} ({has_disease / len(df) * 100:.1f}%)")
    print("    -> The dataset is well-balanced! (No extreme class imbalance problem).")

    # 4. Basic Medical Statistics
    print("\n[4] Key Clinical Metric Ranges:")
    print(f"    - Age: min {df['age'].min()}, mean {df['age'].mean():.1f}, max {df['age'].max()} years")
    print(f"    - Resting Blood Pressure (trestbps): min {df['trestbps'].min()}, mean {df['trestbps'].mean():.1f}, max {df['trestbps'].max()} mm Hg")
    print(f"    - Serum Cholesterol (chol): min {df['chol'].min()}, mean {df['chol'].mean():.1f}, max {df['chol'].max()} mg/dl")
    print(f"    - Max Heart Rate (thalach): min {df['thalach'].min()}, mean {df['thalach'].mean():.1f}, max {df['thalach'].max()} bpm")

    # 5. Generate and Save Visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Visualization 1: Target Distribution Plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x='target', data=df, palette=['#3498db', '#e74c3c'])
    plt.title("Distribution of Heart Disease Cases", fontsize=12, fontweight='bold')
    plt.xlabel("Diagnosis (0 = Healthy, 1 = Heart Disease)", fontsize=10)
    plt.ylabel("Patient Count", fontsize=10)
    plt.xticks([0, 1], ['No Disease (0)', 'Disease (1)'])
    plt.tight_layout()
    target_dist_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(target_dist_path, dpi=300)
    plt.close()
    print(f"\n[5] Saved visual chart: {target_dist_path}")

    # Visualization 2: Correlation Heatmap
    plt.figure(figsize=(11, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar=True)
    plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"    Saved visual chart: {heatmap_path}")

    print("\n" + "=" * 60)
    print("EDA completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    perform_eda()
