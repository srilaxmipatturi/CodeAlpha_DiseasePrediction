"""
Heart Disease Prediction & Clinical Decision Support System
Author: CodeAlpha Internship
Description: Advanced clinical dashboard with real-time patient diagnosis,
             quick preset profiles, explainable AI risk factors, batch screening,
             AHA guideline indicators, and downloadable medical reports.
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioPredict AI | CodeAlpha",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .status-badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD ARTIFACTS
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "metadata.json")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals")
SAMPLE_BATCH_PATH = os.path.join(BASE_DIR, "data", "sample_patients_batch.csv")

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FEATURE_LABELS = {
    'cp': 'Chest Pain Type',
    'thalach': 'Max Heart Rate',
    'oldpeak': 'ST Depression',
    'thal': 'Thalassemia',
    'chol': 'Cholesterol',
    'ca': 'Major Vessels',
    'age': 'Age',
    'trestbps': 'Resting Blood Pressure',
    'exang': 'Exercise Angina',
    'slope': 'Peak ST Slope',
    'sex': 'Sex',
    'restecg': 'Resting ECG',
    'fbs': 'Fasting Blood Sugar'
}

@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            metadata = json.load(f)
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()

# Header Banner
st.markdown("""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; color: #ffffff;">❤️ CardioPredict AI</h1>
            <p style="margin: 6px 0 0 0; opacity: 0.9; font-size: 1.05rem; color: #cbd5e1;">
                Clinical Decision Support System | CodeAlpha Data Science & ML Internship
            </p>
        </div>
        <div style="text-align: right; background: rgba(255, 255, 255, 0.1); padding: 10px 18px; border-radius: 8px;">
            <span style="font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">Active Model</span><br>
            <span style="font-size: 1.15rem; font-weight: bold; color: #38bdf8;">Random Forest (83.6% Acc)</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Navigation")
page = st.sidebar.radio(
    "Choose Module",
    [
        "🏥 Patient Risk Assessment",
        "📂 Batch Screening (CSV)",
        "📊 Model Comparison & Analytics",
        "🩺 Clinical Guidelines (AHA)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Model Health & Specs")
if metadata:
    st.sidebar.markdown(f"**Algorithm:** `{metadata.get('best_model_name', 'Random Forest')}`")
    st.sidebar.markdown(f"**Test Accuracy:** `{metadata.get('test_accuracy', 0.8361) * 100:.2f}%`")
    st.sidebar.markdown(f"**Test Recall:** `96.97%` *(High sensitivity)*")
st.sidebar.markdown("---")
st.sidebar.caption("Built for CodeAlpha Data Science Internship")

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def get_bp_category(bp):
    if bp < 120:
        return "Normal (< 120)", "status-badge-low"
    elif 120 <= bp <= 129:
        return "Elevated (120-129)", "status-badge-warning"
    elif 130 <= bp <= 139:
        return "Stage 1 Hypertension (130-139)", "status-badge-high"
    else:
        return "Stage 2 Hypertension (>= 140)", "status-badge-high"

def get_chol_category(chol):
    if chol < 200:
        return "Desirable (< 200 mg/dl)", "status-badge-low"
    elif 200 <= chol <= 239:
        return "Borderline High (200-239)", "status-badge-warning"
    else:
        return "High Cholesterol (>= 240)", "status-badge-high"

# -----------------------------------------------------------------------------
# MODULE 1: INDIVIDUAL PATIENT RISK ASSESSMENT
# -----------------------------------------------------------------------------
if page == "🏥 Patient Risk Assessment":
    st.subheader("Patient Clinical Evaluation")
    st.write("Fill in patient vitals or load a preset sample profile to evaluate heart disease risk.")

    # Preset Profile Loader
    st.markdown("##### ⚡ Quick Clinical Presets (One-Click Demo):")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)

    # Initialize session state for default values
    defaults = {
        "age": 52, "sex": 1, "cp": 2, "trestbps": 125, "chol": 212,
        "fbs": 0, "restecg": 0, "thalach": 168, "exang": 0, "oldpeak": 1.0,
        "slope": 1, "ca": 0, "thal": 2
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    with p_col1:
        if st.button("🧑 Load Healthy Adult", use_container_width=True):
            st.session_state.update({
                "age": 36, "sex": 0, "cp": 0, "trestbps": 115, "chol": 175,
                "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0, "oldpeak": 0.0,
                "slope": 2, "ca": 0, "thal": 2
            })
            st.rerun()

    with p_col2:
        if st.button("👴 Load High-Risk Cardiac Patient", use_container_width=True):
            st.session_state.update({
                "age": 67, "sex": 1, "cp": 0, "trestbps": 160, "chol": 290,
                "fbs": 1, "restecg": 0, "thalach": 105, "exang": 1, "oldpeak": 3.6,
                "slope": 0, "ca": 2, "thal": 3
            })
            st.rerun()

    with p_col3:
        if st.button("🧑‍💼 Load Borderline Patient", use_container_width=True):
            st.session_state.update({
                "age": 54, "sex": 1, "cp": 1, "trestbps": 138, "chol": 242,
                "fbs": 0, "restecg": 1, "thalach": 145, "exang": 0, "oldpeak": 1.2,
                "slope": 1, "ca": 1, "thal": 2
            })
            st.rerun()

    with p_col4:
        if st.button("🔄 Reset Defaults", use_container_width=True):
            st.session_state.update(defaults)
            st.rerun()

    st.markdown("---")

    # Form with Live Vitals Feedback
    with st.form("patient_assessment_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 1. Demographics & Symptoms")
            age = st.slider("Age (years)", 18, 100, st.session_state.age)
            sex = st.selectbox(
                "Biological Sex",
                options=[("Male", 1), ("Female", 0)],
                index=0 if st.session_state.sex == 1 else 1,
                format_func=lambda x: x[0]
            )[1]
            cp = st.selectbox(
                "Chest Pain Type (cp)",
                options=[
                    ("Typical Angina (0)", 0),
                    ("Atypical Angina (1)", 1),
                    ("Non-anginal Pain (2)", 2),
                    ("Asymptomatic (3)", 3)
                ],
                index=st.session_state.cp,
                format_func=lambda x: x[0]
            )[1]
            exang = st.selectbox(
                "Exercise-Induced Angina (exang)",
                options=[("No (0)", 0), ("Yes (1)", 1)],
                index=st.session_state.exang,
                format_func=lambda x: x[0]
            )[1]

        with col2:
            st.markdown("#### 2. Vitals & Blood Tests")
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, st.session_state.trestbps)
            bp_label, bp_class = get_bp_category(trestbps)
            st.markdown(f"**AHA BP Status:** <span class='{bp_class}'>{bp_label}</span>", unsafe_allow_html=True)

            chol = st.slider("Serum Cholesterol (mg/dl)", 100, 580, st.session_state.chol)
            chol_label, chol_class = get_chol_category(chol)
            st.markdown(f"**Lipid Status:** <span class='{chol_class}'>{chol_label}</span>", unsafe_allow_html=True)

            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl (fbs)",
                options=[("Normal / <= 120 mg/dl (0)", 0), ("Elevated / > 120 mg/dl (1)", 1)],
                index=st.session_state.fbs,
                format_func=lambda x: x[0]
            )[1]
            thalach = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, st.session_state.thalach)

        with col3:
            st.markdown("#### 3. Cardiac Diagnostics")
            restecg = st.selectbox(
                "Resting ECG (restecg)",
                options=[
                    ("Normal (0)", 0),
                    ("ST-T Wave Abnormality (1)", 1),
                    ("Left Ventricular Hypertrophy (2)", 2)
                ],
                index=st.session_state.restecg,
                format_func=lambda x: x[0]
            )[1]
            oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, float(st.session_state.oldpeak), step=0.1)
            slope = st.selectbox(
                "Peak Exercise ST Slope",
                options=[("Upsloping (0)", 0), ("Flat (1)", 1), ("Downsloping (2)", 2)],
                index=st.session_state.slope,
                format_func=lambda x: x[0]
            )[1]
            ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-3)", options=[0, 1, 2, 3], index=st.session_state.ca)
            thal = st.selectbox(
                "Thalassemia Status (thal)",
                options=[
                    ("Normal (1)", 1),
                    ("Fixed Defect (2)", 2),
                    ("Reversible Defect (3)", 3)
                ],
                index=min(max(st.session_state.thal - 1, 0), 2),
                format_func=lambda x: x[0]
            )[1]

        submitted = st.form_submit_button("🔍 Run Diagnostic Assessment", use_container_width=True)

    # Prediction Outcome
    if submitted:
        input_df = pd.DataFrame([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]], columns=FEATURE_NAMES)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")
        st.subheader("📋 Diagnostic Assessment Result")

        res_col1, res_col2 = st.columns([1.4, 1])

        with res_col1:
            if prediction == 1:
                st.error("### ⚠️ Result: High Risk of Heart Disease Indicated")
                st.markdown(f"""
                * **Predicted Class:** **Positive for Heart Disease (Class 1)**
                * **Model Confidence:** **`{probability * 100:.1f}%` Probability**
                * **Risk Severity:** **CRITICAL**
                
                #### 👨‍⚕️ Clinical Recommendations:
                1. **Cardiologist Consultation:** Priority referral for specialized diagnostic workup.
                2. **Confirmatory Tests:** Coronary angiography or Myocardial Perfusion Imaging (MPI).
                3. **Targeted Intervention:** Strict management of resting blood pressure, blood glucose, and statin therapy if indicated.
                """)
            else:
                st.success("### ✅ Result: Low Risk (Healthy Profile)")
                st.markdown(f"""
                * **Predicted Class:** **No Heart Disease Detected (Class 0)**
                * **Confidence:** **`{(1 - probability) * 100:.1f}%` Confidence**
                * **Risk Severity:** **LOW / OPTIMAL**
                
                #### 🌿 Preventative Guidance:
                1. **Routine Surveillance:** Continue annual preventative health checkups.
                2. **Cardiovascular Wellness:** 150 minutes of moderate aerobic activity weekly.
                3. **Dietary Standards:** Mediterranean or DASH diet low in saturated fats.
                """)

        with res_col2:
            st.markdown("#### Risk Gauge & Metrics")
            st.metric("Cardiovascular Risk Score", f"{probability * 100:.1f}%")
            st.progress(float(probability))
            
            if probability > 0.65:
                st.markdown("<span class='status-badge-high'>CRITICAL CARDIAC RISK</span>", unsafe_allow_html=True)
            elif probability > 0.40:
                st.markdown("<span class='status-badge-warning'>MODERATE RISK / BORDERLINE</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='status-badge-low'>LOW CARDIAC RISK</span>", unsafe_allow_html=True)

            st.caption(f"Evaluated by: **{metadata.get('best_model_name', 'Random Forest')}** (83.61% Test Accuracy)")

        # Explainable AI: Feature Importance Breakdown
        st.markdown("---")
        st.subheader("🧠 Explainable AI: Key Influencing Factors")
        st.write("Understand which clinical biomarkers contributed most to this assessment.")

        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_NAMES],
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(importance_df["Feature"][-6:], importance_df["Importance"][-6:], color="#3b82f6")
        ax.set_xlabel("Relative Importance Weight", fontsize=9)
        ax.set_title("Top 6 Predictive Biomarkers (Model Decision Weights)", fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # Downloadable Diagnostic Report
        report_text = f"""======================================================================
              CARDIOVASCULAR DIAGNOSTIC HEALTH REPORT
======================================================================
Generated By: CardioPredict AI (CodeAlpha Project)
Model Used:   Random Forest Classifier (Test Accuracy: 83.61%)
Date:         {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT CLINICAL VITALS:
----------------------------------------------------------------------
Age:                     {age} years
Biological Sex:          {'Male' if sex == 1 else 'Female'}
Chest Pain Type:         Code {cp}
Resting Blood Pressure:  {trestbps} mm Hg ({bp_label})
Serum Cholesterol:       {chol} mg/dl ({chol_label})
Fasting Blood Sugar:     {'> 120 mg/dl' if fbs == 1 else '<= 120 mg/dl'}
Resting ECG:             Code {restecg}
Max Heart Rate:          {thalach} bpm
Exercise Angina:         {'Yes' if exang == 1 else 'No'}
ST Depression (oldpeak): {oldpeak}
Peak ST Slope:           Code {slope}
Major Vessels Colored:   {ca}
Thalassemia:             Code {thal}

ASSESSMENT RESULT:
----------------------------------------------------------------------
Diagnosis:               {'HEART DISEASE INDICATED (HIGH RISK)' if prediction == 1 else 'NO HEART DISEASE (LOW RISK)'}
Heart Disease Probability: {probability * 100:.2f}%
Assessment Status:       {'CRITICAL ATTENTION REQUIRED' if prediction == 1 else 'HEALTHY / ROUTINE MONITORING'}

MEDICAL DISCLAIMER:
----------------------------------------------------------------------
This report was generated by a machine learning model built for the
CodeAlpha Internship. It is intended for educational purposes and 
does not replace a licensed physician's clinical judgment.
======================================================================
"""
        st.download_button(
            label="📥 Download Clinical Diagnostic Report (.txt)",
            data=report_text,
            file_name=f"cardiac_report_{age}_{'M' if sex == 1 else 'F'}.txt",
            mime="text/plain",
            use_container_width=True
        )

# -----------------------------------------------------------------------------
# MODULE 2: BATCH PATIENT SCREENING (CSV)
# -----------------------------------------------------------------------------
elif page == "📂 Batch Screening (CSV)":
    st.subheader("📂 Batch Patient Health Screening")
    st.write("Upload a batch CSV file containing multiple patient records to perform automated screening in seconds.")

    b_col1, b_col2 = st.columns([1, 1])
    with b_col1:
        uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])
    with b_col2:
        st.markdown("##### Or Test with Sample Batch Data:")
        if st.button("📁 Load Provided Sample Batch (10 Patients)"):
            if os.path.exists(SAMPLE_BATCH_PATH):
                uploaded_file = SAMPLE_BATCH_PATH
                st.session_state["use_sample_batch"] = True

    batch_df = None
    if uploaded_file is not None:
        if isinstance(uploaded_file, str):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        st.success(f"Loaded {len(batch_df)} patient records successfully.")
        
        # Verify required features exist
        missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            # Predict
            X_batch = batch_df[FEATURE_NAMES]
            X_scaled = scaler.transform(X_batch)
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)[:, 1]

            results_df = batch_df.copy()
            results_df["Risk Score (%)"] = np.round(probabilities * 100, 1)
            results_df["Diagnosis"] = ["⚠️ High Risk" if p == 1 else "✅ Low Risk" for p in predictions]

            # High level metrics
            total = len(results_df)
            high_risk_count = int(np.sum(predictions == 1))
            low_risk_count = total - high_risk_count

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Screened", total)
            m2.metric("High Risk Cases Flagged", high_risk_count, delta=f"{high_risk_count/total*100:.1f}% of cohort", delta_color="inverse")
            m3.metric("Low Risk / Healthy Cases", low_risk_count, delta=f"{low_risk_count/total*100:.1f}%")

            st.markdown("---")
            st.subheader("Screening Cohort Results")
            st.dataframe(results_df, use_container_width=True)

            # Export Batch Results
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Analyzed Cohort CSV",
                data=csv_buffer.getvalue(),
                file_name="screened_patients_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# -----------------------------------------------------------------------------
# MODULE 3: MODEL COMPARISON & ANALYTICS
# -----------------------------------------------------------------------------
elif page == "📊 Model Comparison & Analytics":
    st.subheader("Model Performance Comparison (CodeAlpha Task)")
    st.write("We trained and compared all 4 algorithms requested by CodeAlpha on 20% unseen test data (61 patients).")

    if metadata and "all_results" in metadata:
        results_df = pd.DataFrame(metadata["all_results"]).T
        results_df = results_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
        if hasattr(results_df, "map"):
            results_df = results_df.map(lambda x: f"{x * 100:.2f}%")
        else:
            results_df = results_df.applymap(lambda x: f"{x * 100:.2f}%")
        st.dataframe(results_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Visual Analytics")
    tab1, tab2, tab3 = st.tabs(["Algorithm Accuracy Comparison", "Confusion Matrix", "Correlation Heatmap"])

    with tab1:
        comp_img = os.path.join(VISUALS_DIR, "model_comparison.png")
        if os.path.exists(comp_img):
            st.image(comp_img, caption="Comparison of 4 CodeAlpha Algorithms", use_container_width=True)

    with tab2:
        cm_img = os.path.join(VISUALS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix of Random Forest on Test Patients", use_container_width=True)

    with tab3:
        corr_img = os.path.join(VISUALS_DIR, "correlation_heatmap.png")
        if os.path.exists(corr_img):
            st.image(corr_img, caption="Correlation Matrix of Clinical Biomarkers", use_container_width=True)

# -----------------------------------------------------------------------------
# MODULE 4: CLINICAL GUIDELINES (AHA)
# -----------------------------------------------------------------------------
else:
    st.subheader("American Heart Association (AHA) Clinical Reference")
    st.write("Reference standards used to calibrate our biomarker assessment rules:")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🩺 Blood Pressure Stages (AHA Guidelines)")
        st.markdown("""
        | BP Category | Systolic (mm Hg) | & / or | Diastolic (mm Hg) |
        | :--- | :--- | :---: | :--- |
        | **Normal** | Less than 120 | and | Less than 80 |
        | **Elevated** | 120 – 129 | and | Less than 80 |
        | **Stage 1 Hypertension** | 130 – 139 | or | 80 – 89 |
        | **Stage 2 Hypertension** | 140 or higher | or | 90 or higher |
        | **Hypertensive Crisis** | Higher than 180 | and/or | Higher than 120 |
        """)

    with col_b:
        st.markdown("#### 🧪 Serum Cholesterol Classifications")
        st.markdown("""
        | Total Cholesterol | Category | Clinical Implication |
        | :--- | :--- | :--- |
        | **Less than 200 mg/dl** | Desirable | Lower risk of coronary artery disease |
        | **200 – 239 mg/dl** | Borderline High | Lifestyle modifications recommended |
        | **240 mg/dl and above** | High Risk | Double the risk of coronary disease |
        """)

    st.markdown("---")
    st.markdown("#### 📖 Feature Dictionary")
    st.markdown("""
    * **Chest Pain Type (`cp`):** 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal pain, 3: Asymptomatic.
    * **Exercise Angina (`exang`):** Pain in chest induced by physical exercise.
    * **ST Depression (`oldpeak`):** Electrocardiographic marker indicating myocardial ischemia during stress.
    * **Thalassemia (`thal`):** Inherited blood hemoglobin disorder affecting oxygen delivery.
    """)
