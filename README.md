# 🩸 Blood Disease Prediction using Machine Learning

## 📌 Overview

This project builds a simple machine learning model to predict blood-related conditions (such as anemia) using patient data.
It uses basic data analysis, visualization, and a Random Forest model to make predictions.

---

## 🚀 Features

* 📊 Data analysis (EDA)
* 📈 Visualizations (countplot & heatmap)
* 🤖 Machine Learning model (Random Forest)
* 📉 Evaluation metrics (Accuracy, Precision, Recall, F1-score)
* 🔍 Confusion Matrix
* ⭐ Feature Importance
* 🔮 Sample prediction

---

## 🛠️ Technologies Used

* Python
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Dataset

The dataset contains blood sample data with multiple features and a target column representing the disease.

👉 Place your dataset file here:

```bash
D:/blood.csv
```

---

## ▶️ How to Run

1. Install required libraries:

```bash
pip install pandas matplotlib seaborn scikit-learn
```

2. Run the Python file:

```bash
python dppp.py
```

---

## 📊 Output

The model will:

* Show dataset details
* Display visualizations:

  * Target Distribution
  * Correlation Heatmap
* Train a Random Forest model
* Print:

  * Accuracy
  * Classification Report (Precision, Recall, F1-score)
* Show Confusion Matrix
* Show Feature Importance
* Predict sample output

---

## 🧠 Model Used

* Random Forest Classifier

Why Random Forest?

* High accuracy
* Handles complex data
* Reduces overfitting

---

## 🎯 Results

The model achieved high accuracy on test data and successfully predicted disease labels.

---

## 💡 Future Improvements

* Add more models (SVM, XGBoost)
* Improve dataset size and quality
* Build a web app using Streamlit
* Add multi-disease prediction system

---
