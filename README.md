
# 📉 Customer Churn Prediction using Cost-Sensitive Machine Learning

## 📌 Overview

This project is a **complete end-to-end Customer Churn Prediction system** designed to optimize **real-world business decisions**, not just model accuracy.

Unlike traditional ML projects, this system focuses on:

- 💰 Minimizing financial loss (₹)
- 🎯 Optimizing decision thresholds
- 🧠 Selecting models based on business impact

---

## 🚀 Key Highlights

### 💰 Cost-Sensitive Learning (Core Innovation)

- False Negative (missed churner) → ₹10,000 loss  
- False Positive (unnecessary offer) → ₹500 cost  

👉 Model is optimized to **minimize total cost**, not maximize accuracy.

---

### 🎯 Threshold Optimization

Instead of default `0.5`, the project:

- Tests multiple thresholds
- Selects **cost-optimal threshold**

📉 Result:
- Default cost: ₹944,000  
- Optimized cost: ₹382,500  
- 💸 Savings: **₹561,500**

---

### 🧠 Business-Driven Model Selection

Even after training advanced models:

- XGBoost
- LightGBM
- CatBoost
- Stacking
- ANN

👉 Final selected model: **Logistic Regression**

✔ Reason: **Lowest business cost**, not highest accuracy

---

### ⚙️ Complete ML Pipeline

```

Data Loading → EDA → Cleaning → Feature Engineering →
Encoding → Training → Evaluation →
Threshold Optimization → Business Analysis →
Model Saving → Inference

```

---

### 📊 Models Used

- Logistic Regression (Baseline + Tuned)
- Decision Tree
- Random Forest
- XGBoost (Calibrated)
- LightGBM
- CatBoost
- Stacking Ensemble
- Artificial Neural Network (PyTorch)

---

### 📈 Evaluation Metrics

- Accuracy
- Precision / Recall / F1-score
- ROC-AUC & PR-AUC
- Confusion Matrix
- Cross-validation
- 💰 **Business Cost (Primary Metric)**

---

### 🔍 Explainability

- SHAP Global Feature Importance
- SHAP Individual Predictions
- Helps identify **key churn drivers**

---

### 🤖 Deep Learning (ANN)

- PyTorch-based neural network
- Dropout + BatchNorm
- Early stopping
- Class imbalance handled using weighted loss

---

## 📊 Final Results

| Model | ROC-AUC | Cost (₹) |
|------|--------|---------|
| Logistic Regression | ~0.845 | **₹392,500 (Best)** |
| ANN | ~0.845 | ₹383,500 |
| Stacking | ~0.846 | ₹400,500 |

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
├── README.md

````

---

## ⚙️ Installation & Setup (Run Locally)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
````

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost shap optuna torch
```

---

### 4️⃣ Add Dataset

Place the dataset here:

```
data/raw/telco_churn.csv
```

---

### 5️⃣ Run the Project

#### Option 1: Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
churn_pipeline.ipynb
```

Run all cells.

---

#### Option 2: Run as Python Script (if converted)

```bash
python main.py
```

---

## 💾 Model Output

After running:

```
models/churn_model.pkl
```

Contains:

* Trained model
* Optimal threshold

---

## 🚀 Inference (Using Saved Model)

```python
import pickle

with open("models/churn_model.pkl", "rb") as f:
    saved_obj = pickle.load(f)

model = saved_obj["model"]
threshold = saved_obj["threshold"]

preds = model.predict_proba(X_new)[:, 1]
predictions = (preds >= threshold).astype(int)
```

---

## 🎯 Real-World Applications

* Telecom churn prediction
* Subscription services (Netflix, SaaS)
* Banking & insurance retention
* E-commerce customer analytics

---

## 📌 Key Takeaways

✔ Accuracy alone is misleading
✔ Threshold tuning is critical
✔ Business cost should drive ML decisions
✔ Simpler models can outperform complex ones in real-world

---

## 👨‍💻 Author

**Pranav**

---

## ⭐ Final Thought

> Machine Learning is not about predicting correctly.
> It is about making the **right decision at the right cost**.

```


