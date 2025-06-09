# 💳 Credit Card Fraud Detection using Neural Networks

This project focuses on detecting fraudulent transactions using a Multi-Layer Perceptron (MLP) neural network. It tackles the challenge of highly imbalanced datasets with oversampling techniques and evaluates model performance using key classification metrics.

---

## 📌 Project Overview

- Developed a neural network model (MLP) to detect fraud in credit card transactions.
- Addressed severe class imbalance using **ADASYN (Adaptive Synthetic Sampling)**.
- Trained and compared two MLP models:
  - A **basic MLP** architecture.
  - An **MLP with Dropout** to reduce overfitting.
- Performed hyperparameter tuning using **GridSearchCV**.
- Implemented **Early Stopping** to prevent overtraining.
- Evaluated model performance using:
  - **Accuracy**
  - **F1-Score**
  - **ROC-AUC Score**

---

## 🔧 Technologies & Libraries

- Python
- NumPy, Pandas
- Scikit-learn
- Keras / TensorFlow
- ADASYN (from `imblearn`)
- Matplotlib, Seaborn (for visualization)

---

## 🚀 Results

- The **dropout-enhanced MLP** showed improved generalization and better fraud detection performance.
- Achieved higher **F1-score and ROC-AUC** compared to the basic model.
- Model training was more efficient due to early stopping and optimized hyperparameters.

---

## 🧠 Key Concepts

- **Imbalanced Learning**  
- **Synthetic Oversampling (ADASYN)**  
- **Dropout Regularization**  
- **GridSearchCV for Tuning**  
- **Early Stopping in Training**  
- **Binary Classification Metrics**

---
