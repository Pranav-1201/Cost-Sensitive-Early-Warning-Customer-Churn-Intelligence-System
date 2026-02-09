# 📉 Customer Churn Prediction using Cost-Sensitive Machine Learning

## 📌 Overview

This project implements an **end-to-end Customer Churn Prediction system** that focuses on **real-world business impact rather than just model accuracy**. Unlike standard churn classifiers, this system uses **cost-sensitive learning** to explicitly penalize expensive mistakes—especially **false negatives**, where a churner is incorrectly predicted as a loyal customer.

The goal is not just to predict churn, but to **support smarter retention decisions**.

---

## 🚀 Key Features

* **Cost-Sensitive Modeling**
  Incorporates misclassification costs to reflect real financial losses caused by churn.

* **Class Imbalance Handling**
  Addresses skewed churn distributions using weighting strategies instead of naïve oversampling.

* **Multiple Model Comparisons**
  Trains and evaluates baseline models and advanced classifiers under the same cost-aware framework.

* **Business-Oriented Evaluation**
  Goes beyond accuracy using:

  * Precision, Recall, F1-score
  * ROC–AUC
  * Confusion Matrices
  * Cost-based performance metrics

* **Model Explainability**
  Uses feature importance and explainable AI techniques to interpret predictions and identify churn drivers.

* **Modular & Scalable Pipeline**
  Clean separation of preprocessing, training, evaluation, robustness testing, and inference.

---

## 🧠 Why Cost-Sensitive Learning?

In real businesses:

* Losing a customer (false negative) is far more expensive than contacting a loyal one (false positive).
* Traditional ML models treat all errors equally, leading to misleading “high accuracy.”

This project explicitly **optimizes for decision quality**, ensuring the model focuses on customers that matter most.

---

## 🗂️ Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
│   └── feature_engineering.py
├── models/
│   ├── baseline/
│   └── cost_sensitive/
├── training/
│   └── train_models.py
├── evaluation/
│   ├── evaluate_models.py
│   └── robustness_test.py
├── explainability/
│   └── model_explanations.py
├── utils/
│   └── metrics.py
└── README.md
```

---

## ⚙️ Workflow

1. Data preprocessing & feature engineering
2. Cost-sensitive model training
3. Model comparison and evaluation
4. Robustness and generalization testing
5. Explainability and insight extraction

---

## 📊 Results

The cost-sensitive models consistently achieve **higher recall for churners** and **lower overall business cost** compared to accuracy-optimized baselines, making them more suitable for real-world deployment.

---

## 🧪 Tech Stack

* Python
* Scikit-learn
* Pandas, NumPy
* Matplotlib / Seaborn
* Explainable AI tools

---

## 🎯 Use Cases

* Telecom churn prediction
* Subscription-based services
* Banking and insurance retention
* Any imbalanced, cost-critical classification problem

---

## 📌 Key Takeaway

This project demonstrates how **machine learning decisions should align with business objectives**, not just leaderboard metrics.

