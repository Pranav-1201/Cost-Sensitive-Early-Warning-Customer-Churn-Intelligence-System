# 📉 Customer Churn Prediction using Cost-Sensitive Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-End%20to%20End-green)
![Status](https://img.shields.io/badge/Project-Production%20Ready-success)

---

## 📌 Overview

This project is a **complete end-to-end Customer Churn Prediction system** designed to optimize **real-world business decisions**, not just model accuracy.

Unlike traditional ML projects, this system focuses on:

* 💰 Minimizing financial loss (₹)
* 🎯 Optimizing decision thresholds
* 🧠 Selecting models based on business impact

---

## 🚀 Why This Project Stands Out

Most ML projects optimize for accuracy.

👉 This project optimizes for **business cost**, making it **production-relevant and decision-focused**.

✔ Uses cost-sensitive learning
✔ Selects threshold based on financial impact
✔ Chooses model based on real-world loss
✔ Includes explainability + deployment pipeline

---

## 💰 Cost-Sensitive Learning (Core Innovation)

* False Negative (missed churner) → ₹10,000 loss
* False Positive (unnecessary offer) → ₹500 cost

👉 Model is optimized to **minimize total cost**, not maximize accuracy.

---

## 🎯 Threshold Optimization

Instead of default `0.5`, the project:

* Tests multiple thresholds
* Selects **cost-optimal threshold**

📉 Result:

* Default cost: ₹944,000
* Optimized cost: ₹382,500
* 💸 Savings: **₹561,500**

---

## 📊 Business Impact Visualization

![alt text](image.png)
![alt text](image-1.png)
```
images/cost_vs_threshold.png
images/shap_summary.png
```

---

## 🧠 Business-Driven Model Selection

Even after training advanced models:

* XGBoost
* LightGBM
* CatBoost
* Stacking
* ANN

👉 Final selected model: **Logistic Regression**

✔ Reason: **Lowest business cost**, not highest accuracy

---

## ⚙️ Complete ML Pipeline

```
![alt text](image-2.png)
Data Loading → EDA → Cleaning → Feature Engineering →
Encoding → Training → Evaluation →
Threshold Optimization → Business Analysis →
Model Saving → Inference
```

---

## 📊 Models Used

* Logistic Regression (Baseline + Tuned)
* Decision Tree
* Random Forest
* XGBoost (Calibrated with Isotonic Regression)
* LightGBM
* CatBoost
* Stacking Ensemble
* Artificial Neural Network (PyTorch)

---

## 📈 Evaluation Metrics

* Accuracy
* Precision / Recall / F1-score
* ROC-AUC & PR-AUC
* Confusion Matrix
* Cross-validation
* 💰 **Business Cost (Primary Metric)**

---

## 🔍 Explainability

* SHAP Global Feature Importance
* SHAP Individual Predictions

👉 Helps identify **key churn drivers**

---

## 🤖 Deep Learning (ANN)

* PyTorch-based neural network
* Dropout + BatchNorm
* Early stopping + checkpointing
* Class imbalance handled using weighted loss

---

## 📊 Final Results

| Model               | ROC-AUC | Cost (₹)            |
| ------------------- | ------- | ------------------- |
| Logistic Regression | ~0.845  | **₹392,500 (Best)** |
| ANN                 | ~0.845  | ₹383,500            |
| Stacking            | ~0.846  | ₹400,500            |

👉 Final Model: **Logistic Regression (Cost-Optimal)**

---

## 📂 Project Structure

```
├── data/
│   └── raw/
│       └── telco_churn.csv
├── models/
│   └── churn_model.pkl
├── notebook/
│   └── churn_pipeline.ipynb
├── images/
│   └── (plots here)
├── README.md
```

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Inference (Production-Ready)

```python
import pickle
import pandas as pd

with open("models/churn_model.pkl", "rb") as f:
    saved_obj = pickle.load(f)

model = saved_obj["model"]
threshold = saved_obj["threshold"]
features = saved_obj["feature_names"]

# Example input
input_df = pd.DataFrame([...])

# Use pipeline-safe prediction
preds = model.predict_proba(input_df)[:, 1]
predictions = (preds >= threshold).astype(int)
```

---

## 🎯 Real-World Applications

* Telecom churn prediction
* SaaS & subscription retention
* Banking & insurance analytics
* E-commerce personalization

---

## 📌 Key Takeaways

✔ Accuracy alone is misleading
✔ Threshold tuning is critical
✔ Business cost should drive ML decisions
✔ Simpler models can outperform complex ones

---

## 👨‍💻 Author

**Pranav**

---

## ⭐ Final Thought

> Machine Learning is not about predicting correctly.
> It is about making the **right decision at the right cost**.
