# 🏥 HealthGuard AI  
### AI-Powered Maternal High-Risk Pregnancy Prediction System

HealthGuard AI is a machine learning-powered web application designed to predict high-risk pregnancy cases using maternal health indicators.  

The system leverages a Random Forest classifier trained on structured maternal health data and is deployed as an interactive web app using Streamlit.

🔗 **Live App:**  
https://healthguardai-1.onrender.com

---

## 🚀 Project Overview

Maternal health risk detection is critical for early intervention, especially in low-resource settings.

HealthGuard AI provides:
- Real-time risk prediction
- Probability-based risk scoring
- Clear high-risk vs low-risk classification
- Simple, accessible interface for health workers

The system predicts whether a pregnancy is **High Risk** or **Low Risk** based on clinical indicators.

---

## 🧠 Machine Learning Model

- Algorithm: Random Forest Classifier
- Framework: Scikit-learn
- Model Accuracy: 92%
- High-Risk Recall: 97%
- Model File: `healthguard_model.pkl`

### Features Used:
- Age
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Blood Glucose Level
- Body Temperature
- Heart Rate

The model was trained and validated with proper class balancing techniques to improve high-risk detection performance.

---

## 🖥️ Tech Stack

- Python 3.10
- Streamlit (Web App Framework)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Render (Cloud Deployment)

---

## 📂 Project Structure

health_guard_ai/
│
├── app.py # Streamlit Web App
├── train.py # Model Training Script
├── healthguard_model.pkl # Trained ML Model
├── requirements.txt # Dependencies
├── render.yaml # Deployment Configuration
└── README.md


---

## 🧪 Running Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/health_guard_ai.git
cd health_guard_ai
