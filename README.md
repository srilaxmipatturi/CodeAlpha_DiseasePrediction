# ❤️ Disease Prediction from Medical Data using Machine Learning
**CodeAlpha Data Science & Machine Learning Internship Task**

An end-to-end Machine Learning healthcare solution to predict the risk of cardiovascular (heart) disease based on clinical patient records, symptoms, and diagnostic test results.

---

## 📌 Project Overview
* **Objective:** Predict the possibility of heart disease from structured medical records.
* **Dataset:** UCI Machine Learning Repository (Cleveland Heart Disease Dataset - 303 patient records, 14 clinical features).
* **Techniques:** Classification, Feature Scaling (`StandardScaler`), Train-Test Split (80/20 Stratified).
* **Algorithms Compared:**
  1. **Logistic Regression**
  2. **Support Vector Machine (SVM)**
  3. **Random Forest Classifier** (Selected Best Model - **83.61% Accuracy**, **96.97% Recall**)
  4. **XGBoost Classifier**
* **Deployment:** Interactive Web Dashboard powered by **Streamlit** with:
  * ⚡ **1-Click Clinical Presets:** Demo Healthy, High-Risk, and Borderline profiles instantly.
  * 🩺 **Live AHA Vitals Indicators:** Real-time American Heart Association blood pressure & lipid classifications.
  * 🧠 **Explainable AI (XAI):** Feature importance chart displaying top predictive biomarkers.
  * 📁 **Batch Patient Screening:** Upload a multi-patient CSV for cohort diagnostic analytics.
  * 📥 **Downloadable Diagnostic Health Reports:** One-click patient report export.

---

## 📊 Dataset & Clinical Features
The dataset contains 13 patient attributes and 1 target diagnosis:

| Feature Name | Medical Description | Unit / Encoding |
| :--- | :--- | :--- |
| `age` | Patient Age | Years |
| `sex` | Biological Sex | 1 = Male, 0 = Female |
| `cp` | Chest Pain Type | 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic |
| `trestbps` | Resting Blood Pressure | mm Hg |
| `chol` | Serum Cholesterol | mg/dl |
| `fbs` | Fasting Blood Sugar > 120 mg/dl | 1 = True, 0 = False |
| `restecg` | Resting Electrocardiogram | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
| `thalach` | Maximum Heart Rate Achieved | bpm |
| `exang` | Exercise Induced Angina | 1 = Yes, 0 = No |
| `oldpeak` | ST Depression Induced by Exercise | Numerical (0.0 to 6.2) |
| `slope` | Slope of Peak Exercise ST Segment | 0: Upsloping, 1: Flat, 2: Downsloping |
| `ca` | Number of Major Vessels Colored | 0 to 3 |
| `thal` | Thalassemia Status | 1: Normal, 2: Fixed defect, 3: Reversible defect |
| `target` | **Diagnosis (Ground Truth)** | **0: Healthy / Low Risk, 1: Heart Disease** |

---

## 🏆 Model Performance & Comparison
All 4 algorithms required by CodeAlpha were evaluated on **20% unseen test data** (61 patients) using stratified sampling:

| Algorithm | Test Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.33% | 76.92% | 90.91% | 83.33% | Baseline |
| **Support Vector Machine (SVM)** | 81.97% | 77.50% | 93.94% | 84.93% | Linear/RBF Margin |
| **Random Forest Classifier** | **83.61%** | **78.05%** | **96.97%** | **86.49%** | 🏆 **Best Model** |
| **XGBoost Classifier** | 80.33% | 75.61% | 93.94% | 83.78% | Gradient Boosting |

### Why Random Forest is the Best Model:
1. **High Accuracy:** Achieved **83.61%** test accuracy on unseen patient profiles.
2. **Superior Recall (96.97%):** In clinical medicine, **Recall** is the most critical metric. A 96.97% recall means the model correctly identifies 97 out of 100 actual heart disease patients, drastically minimizing deadly **False Negatives**.

---

## 📁 Project Architecture
```text
CodeAlpha_DiseasePrediction/
│
├── data/
│   └── heart.csv                 # Raw UCI Cleveland dataset (303 records)
│
├── models/
│   ├── best_model.pkl            # Serialized Random Forest model
│   ├── scaler.pkl                # Fitted StandardScaler object
│   └── metadata.json             # Model metrics and feature list
│
├── visuals/
│   ├── target_distribution.png   # Class balance visual
│   ├── correlation_heatmap.png   # Correlation between features
│   ├── model_comparison.png      # 4-model accuracy comparison bar chart
│   └── confusion_matrix.png      # Best model confusion matrix
│
├── src/
│   ├── eda.py                    # Exploratory Data Analysis script
│   ├── train.py                  # Model training and comparison pipeline
│   └── predict.py                # Standalone command-line inference tool
│
├── app/
│   └── app.py                    # Interactive Streamlit Web Application
│
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation & review guide
```

---

## 🚀 How to Run the Project

### 1. Clone or Open the Repository
```bash
git clone https://github.com/<your-username>/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run Exploratory Data Analysis (EDA)
Inspect dataset distribution and generate charts:
```bash
python src/eda.py
```

### 4. Train and Compare Models
Train all 4 models and save the top performer:
```bash
python src/train.py
```

### 5. Run Command-Line Inference Test
Test predictions for sample patient profiles:
```bash
python src/predict.py
```

### 6. Launch the Interactive Web Dashboard
Run the Streamlit web app in your browser:
```bash
python -m streamlit run app/app.py
```

---

## 🎓 Internship Review Q&A (Be Ready to Explain!)

#### Q1: Why did you use StandardScaler?
> **Answer:** Medical features have vastly different scales. For example, cholesterol is in the 200s, while fasting blood sugar is 0 or 1. Scaling standardizes all numerical features to have a mean of 0 and variance of 1, preventing high-magnitude features from dominating distance-based models like SVM and Logistic Regression.

#### Q2: What is the difference between Precision and Recall in this medical context?
> **Answer:** 
> * **Precision:** Out of all patients predicted to have heart disease, how many actually had it?
> * **Recall (Sensitivity):** Out of all patients who genuinely had heart disease, how many did the model catch? In medicine, **Recall** is paramount because failing to identify a sick patient (False Negative) can be life-threatening.

#### Q3: Why did Random Forest outperform the other algorithms?
> **Answer:** Random Forest is an ensemble of multiple uncorrelated decision trees. By averaging their votes, it captures non-linear clinical interactions (e.g., combinations of age, angina, and depression) while effectively reducing overfitting and variance compared to a single decision tree or simpler linear models.

#### Q4: Was the dataset balanced?
> **Answer:** Yes, the dataset contained 138 healthy cases (45.5%) and 165 disease cases (54.5%), which is well-balanced. We also used `stratify=y` during train-test splitting to maintain this exact ratio in both training and test sets.

---

## 👨‍💻 Author
* **Project:** Disease Prediction from Medical Data
* **Internship:** CodeAlpha Data Science & Machine Learning Internship
* **Tools:** Python 3.11, Scikit-Learn, XGBoost, Pandas, Streamlit, Matplotlib, Seaborn
