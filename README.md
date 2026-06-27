# 💳 Credit Card Fraud Detection using Unsupervised Machine Learning

An end-to-end Machine Learning project that detects fraudulent credit card transactions using **Unsupervised Learning** techniques. The application is built with **Streamlit** and allows users to upload transaction data, train anomaly detection models, visualize fraud patterns, and compare model performance.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://YOUR-APP-LINK.streamlit.app

🔗 **GitHub Repository:** https://github.com/YOUR_USERNAME/YOUR_REPOSITORY

---

## 📌 Project Overview

Credit card fraud is one of the biggest challenges in digital payments because fraudulent transactions represent only a tiny fraction of all transactions.

This project uses **unsupervised anomaly detection** to identify suspicious transactions without relying on balanced labeled datasets.

The application provides an interactive dashboard for:

- Uploading transaction datasets
- Exploratory Data Analysis (EDA)
- Detecting anomalies
- Comparing machine learning models
- Visualizing fraud distribution
- Downloading prediction results

---

## 🎯 Features

✅ Interactive Streamlit Dashboard

✅ Upload your own dataset

✅ Exploratory Data Analysis

✅ Isolation Forest

✅ Deep Learning Autoencoder (PyTorch)

✅ PCA Visualization

✅ ROC-AUC Evaluation

✅ Threshold Optimization

✅ Business Cost Analysis

✅ Fraud Detection Report

---

## 🧠 Machine Learning Pipeline

```text
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Feature Scaling
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Isolation Forest
        +
Autoencoder
   │
   ▼
Fraud Score Generation
   │
   ▼
Visualization
   │
   ▼
Fraud Prediction Dashboard
```

---

## 🛠 Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Plotly

---

## 📊 Models Used

### Isolation Forest

- Tree-based anomaly detection
- Suitable for highly imbalanced datasets
- Detects unusual transaction patterns

---

### Autoencoder

- Deep Neural Network
- Learns normal transaction behavior
- Flags transactions with high reconstruction error

---

## 📁 Project Structure

```text
Credit-Card-Fraud-Detection/

│── app.py
│── requirements.txt
│── README.md
│── notebooks/
│── src/
│── models/
│── assets/
│── screenshots/
```

---

## 📷 Dashboard Preview

(Add screenshots here)

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📈 Results

The project successfully detects anomalous transactions using unsupervised learning techniques.

Performance comparison includes:

- ROC-AUC
- Precision
- Recall
- F1 Score
- Reconstruction Error
- Anomaly Score Distribution

---

## 💡 Future Improvements

- Real-time fraud detection
- REST API deployment
- Docker support
- Cloud deployment
- Explainable AI (SHAP)
- Model monitoring

---

## 👨‍💻 Author

**Your Name**

LinkedIn: https://linkedin.com/in/YOUR_PROFILE

GitHub: https://github.com/YOUR_USERNAME

---

⭐ If you found this project useful, consider giving it a star.
