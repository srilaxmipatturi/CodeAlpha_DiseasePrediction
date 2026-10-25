"""
Command-Line Prediction Script for Heart Disease
Author: CodeAlpha Internship
Description: Loads the exported model and scaler to perform inference
             on sample patient profiles from the terminal.
"""

import os
import joblib
import numpy as np
import pandas as pd

def predict_heart_disease(patient_data):
    """
    Takes a dictionary or list of 13 features:
    [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    """
    model_path = os.path.join("models", "best_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Model or Scaler not found! Run 'src/train.py' first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    sample_df = pd.DataFrame([patient_data], columns=feature_names)
    sample_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_scaled)[0]
    probability = model.predict_proba(sample_scaled)[0][1]

    return prediction, probability

def test_samples():
    print("=" * 60)
    print("       TESTING REAL-TIME INFERENCE ON PATIENT SAMPLES       ")
    print("=" * 60)

    # Sample Patient 1: Low-Risk Profile
    # Younger, female, no chest pain, normal BP and cholesterol, high heart rate, no angina
    patient_low_risk = [38, 0, 0, 115, 180, 0, 0, 175, 0, 0.2, 2, 0, 2]

    # Sample Patient 2: High-Risk Profile
    # Older, male, asymptomatic chest pain, high BP, high cholesterol, low max heart rate, angina present
    patient_high_risk = [67, 1, 3, 160, 286, 1, 0, 108, 1, 1.8, 1, 3, 3]

    print("\n[Test 1] Testing Low-Risk Patient Profile:")
    pred1, prob1 = predict_heart_disease(patient_low_risk)
    diagnosis1 = "Heart Disease Indicated" if pred1 == 1 else "Healthy / Low Risk"
    print(f"    - Input: Age 38, Female, Normal BP (115), Normal Chol (180)")
    print(f"    - Result: {diagnosis1} (Risk Probability: {prob1 * 100:.1f}%)")

    print("\n[Test 2] Testing High-Risk Patient Profile:")
    pred2, prob2 = predict_heart_disease(patient_high_risk)
    diagnosis2 = "Heart Disease Indicated" if pred2 == 1 else "Healthy / Low Risk"
    print(f"    - Input: Age 67, Male, High BP (160), High Chol (286), Angina Present")
    print(f"    - Result: {diagnosis2} (Risk Probability: {prob2 * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("Inference test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    test_samples()
