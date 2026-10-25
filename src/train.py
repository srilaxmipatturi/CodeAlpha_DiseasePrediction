"""
Model Training and Evaluation Pipeline for Heart Disease Prediction
Author: CodeAlpha Internship
Description: Trains and compares 4 classification models (Logistic Regression,
             SVM, Random Forest, XGBoost), evaluates metrics, and exports the
             best model for deployment.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def train_and_evaluate(data_path="data/heart.csv", models_dir="models", visuals_dir="visuals"):
    print("=" * 65)
    print("     DISEASE PREDICTION - MODEL TRAINING & COMPARISON PIPELINE    ")
    print("=" * 65)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    # 1. Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    feature_names = list(X.columns)

    print(f"\n[1] Features: {len(feature_names)} features identified: {feature_names}")
    print(f"    Target: 'target' (0 = Healthy, 1 = Heart Disease)")

    # 2. Train-Test Split (80% Train, 20% Test)
    # Using stratify=y preserves the healthy vs disease ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[2] Data Split Completed:")
    print(f"    - Training Set: {X_train.shape[0]} patients")
    print(f"    - Testing Set:  {X_test.shape[0]} patients (unseen during training)")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("\n[3] Feature Scaling (StandardScaler) applied successfully.")

    # 4. Define the 4 CodeAlpha Classification Algorithms
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine": SVC(probability=True, random_state=42, kernel='rbf'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }

    results = {}
    fitted_models = {}

    print("\n[4] Training and Evaluating Models on Unseen Test Data:")
    print("-" * 65)
    print(f"{'Algorithm':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 65)

    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        fitted_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc,
            "Predictions": y_pred
        }

        print(f"{name:<25} | {acc * 100:>8.2f}% | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}%")

    print("-" * 65)

    # 5. Select Best Model by Test Accuracy
    best_model_name = max(results, key=lambda k: results[k]["Accuracy"])
    best_accuracy = results[best_model_name]["Accuracy"]
    best_model = fitted_models[best_model_name]

    print(f"\n[5] Best Performing Model: >>> {best_model_name} <<< with {best_accuracy * 100:.2f}% Test Accuracy!")

    # 6. Save the Scaler and Best Model
    model_path = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "metadata.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "best_model_name": best_model_name,
        "test_accuracy": round(float(best_accuracy), 4),
        "features": feature_names,
        "all_results": {k: {m: round(float(v), 4) for m, v in metrics.items() if m != "Predictions"} for k, metrics in results.items()}
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\n[6] Saved Artifacts:")
    print(f"    - Best Model saved to:  {model_path}")
    print(f"    - Scaler saved to:      {scaler_path}")
    print(f"    - Metadata saved to:    {meta_path}")

    # 7. Generate Visual Comparison Chart
    plt.figure(figsize=(9, 5))
    algos = list(results.keys())
    accuracies = [results[a]["Accuracy"] * 100 for a in algos]
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']

    bars = plt.bar(algos, accuracies, color=colors, width=0.55)
    plt.title("Model Accuracy Comparison (CodeAlpha Task)", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.ylim(50, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{acc:.2f}%", ha='center', fontweight='bold')

    plt.tight_layout()
    comparison_chart_path = os.path.join(visuals_dir, "model_comparison.png")
    plt.savefig(comparison_chart_path, dpi=300)
    plt.close()
    print(f"\n[7] Visual Charts Saved:")
    print(f"    - Comparison Bar Chart: {comparison_chart_path}")

    # 8. Confusion Matrix for Best Model
    plt.figure(figsize=(6, 4.5))
    cm = confusion_matrix(y_test, results[best_model_name]["Predictions"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy (0)', 'Disease (1)'],
                yticklabels=['Healthy (0)', 'Disease (1)'])
    plt.title(f"Confusion Matrix: {best_model_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=10)
    plt.ylabel("Actual Label", fontsize=10)
    plt.tight_layout()
    cm_path = os.path.join(visuals_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"    - Confusion Matrix:     {cm_path}")

    print("\n" + "=" * 65)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    train_and_evaluate()
