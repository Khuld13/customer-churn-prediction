# 📉 Telco Customer Churn Prediction

> Predict which telecom customers will cancel their subscription — before they do.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-189FDD?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## Overview

An end-to-end machine learning pipeline that predicts customer churn for a telecom company using the IBM Telco Customer Churn dataset. The project covers the full data science workflow — from raw data cleaning and exploratory analysis to model training, hyperparameter tuning, and an interactive Streamlit dashboard for live predictions.

**Business problem:** A telecom company loses ~27% of its customers each year. This model identifies high-risk customers in advance so the retention team can intervene with targeted offers — before the customer cancels.

---

## 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/🔴%20Live%20App-Streamlit-FF4B4B?style=for-the-badge)](https://customer-churn-predict-model.streamlit.app/)


## 📁 Project Structure

```
telco-churn-prediction/
│
├── churn.ipynb               ← Full ML pipeline notebook (EDA → model → evaluation)
├── churn_dashboard.py        ← Streamlit app for live churn prediction
├── churn_model_final.pkl     ← Saved tuned Random Forest model
│
├── X_train.csv               ← Processed training features
├── X_test.csv                ← Processed test features
├── y_train.csv               ← Training labels
├── y_test.csv                ← Test labels
│
├── requirements.txt          ← Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | IBM Telco Customer Churn |
| Rows | 7,043 customers |
| Features | 33 (demographics, account info, services) |
| Target | `Churn Value` — `1` = churned, `0` = retained |
| Class balance | ~73% retained / ~27% churned (imbalanced) |

**Feature categories:**
- **Demographics** — gender, senior citizen status, partner, dependents
- **Account** — tenure, contract type, payment method, monthly charges, total charges
- **Services** — phone, internet, streaming TV/movies, online security, tech support

---

## 🔬 Notebook Walkthrough

| Phase | Section | What Happens |
|---|---|---|
| **Phase 1** | Data Loading | Load raw Excel, inspect shape and types |
| | Null Check | Fix `Total Charges` object → float (11 blank rows for 0-tenure customers) |
| | Duplicate Check | Verify no duplicate rows or CustomerIDs |
| | EDA | Histograms, churn distribution, contract type analysis, correlation heatmap, scatter, boxplots |
| | Feature Engineering | Drop leakage columns, binary encode, one-hot encode, frequency encode City |
| | Train/Test Split | 80/20 stratified split → save CSVs |
| **Phase 2** | Model Training | Logistic Regression · Random Forest · XGBoost (all with class imbalance handling) |
| | Model Selection | Compare F1 and AUC-ROC across all three models |
| | Hyperparameter Tuning | `RandomizedSearchCV` (5-fold CV, optimising F1) on best model |
| | Final Evaluation | Confusion matrix · ROC curve · classification report |
| | Feature Importance | Top 10 features driving churn |
| | Save Model | Export final model as `.pkl` |

---

## 🤖 Models Trained

| Model | Imbalance Handling |
|---|---|
| Logistic Regression | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` |

All models are compared on **F1 score** and **AUC-ROC** — both chosen because accuracy alone is misleading on imbalanced data.

---

## 🔢 Encoding Reference

Binary columns are label-encoded alphabetically:

| Column | 0 | 1 |
|---|---|---|
| `Gender` | Female | Male |
| `Senior Citizen` | No | Yes |
| `Partner` | No | Yes |
| `Dependents` | No | Yes |
| `Phone Service` | No | Yes |
| `Paperless Billing` | No | Yes |
| `Churn Value` *(target)* | **Retained** | **Churned** |

---

## ⚙️ Setup & Installation

**1. Clone the repo**
```bash
git clone https://github.com/Khuld13/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the notebook**
```bash
jupyter notebook churn.ipynb
```

**4. Launch the Streamlit app**
```bash
streamlit run Churn_prediction.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
joblib
streamlit
openpyxl
```

---

## 💡 Key Findings

- **Month-to-month contracts** have the highest churn rate by far — long-term contracts act as a natural retention mechanism
- **High monthly charges + low tenure** is the strongest churn signal — price-sensitive new customers are the most at-risk group
- `Churn Score`, `Churn Reason`, and `CLTV` were identified and removed as **data leakage** features before modelling
- Class imbalance (~27% churn) was handled explicitly in all three models — not ignored

---

## 👩‍💻 Author

**Sayyeda Khuld**  
ML Engineering Intern @ FlyRank AI · BSc Information Technology, University of Education Lahore

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sayyedakhuld--analyst-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/sayyedakhuld)
[![GitHub](https://img.shields.io/badge/GitHub-Khuld13-181717?style=flat&logo=github)](https://github.com/Khuld13)

---

## 📄 License

This project is licensed under the MIT License.
