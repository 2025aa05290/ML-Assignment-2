
# Machine Learning Assignment 2  
## Heart Disease Classification using Multiple ML Models  


**Name:** Deepak

**ID:** 2025AA052290

**Email:** 2025aa05290@wilp.bits-pilani.ac.in

**Course:** Machine Learning  

---

# a. Problem Statement

The objective of this project is to build and compare multiple machine learning 
classification models to predict the presence of heart disease based on medical attributes.

The goals are:
- Implement six different classification models  
- Evaluate them using multiple performance metrics  
- Compare their performance  
- Deploy the models using a Streamlit web application  

This project demonstrates the complete end-to-end ML workflow including model training, 
evaluation, UI development, and deployment.

---

# b. Dataset Description

Dataset: Heart Disease Dataset (UCI Repository / Kaggle Version)

### Dataset Details:

- Total Instances: 1025  
- Number of Features: 13  
- Target Variable: `target`  
  - 0 → No Heart Disease  
  - 1 → Heart Disease  

### Feature Description:

- age – Age of patient  
- sex – Gender (1 = male, 0 = female)  
- cp – Chest pain type  
- trestbps – Resting blood pressure  
- chol – Serum cholesterol  
- fbs – Fasting blood sugar  
- restecg – Resting ECG results  
- thalach – Maximum heart rate achieved  
- exang – Exercise induced angina  
- oldpeak – ST depression  
- slope – Slope of peak exercise ST segment  
- ca – Number of major vessels  
- thal – Thalassemia  

The dataset contains no missing values and is suitable for binary classification.

---

# c. Models Used

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

---

# Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

---

# Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|------------|--------|----------|------|
| Logistic Regression | XX | XX | XX | XX | XX | XX |
| Decision Tree | XX | XX | XX | XX | XX | XX |
| KNN | XX | XX | XX | XX | XX | XX |
| Naive Bayes | XX | XX | XX | XX | XX | XX |
| Random Forest (Ensemble) | XX | XX | XX | XX | XX | XX |
| XGBoost (Ensemble) | XX | XX | XX | XX | XX | XX |

(Replace XX with actual results from notebook.)

---

# Observations on Model Performance

- Logistic Regression performs well for linearly separable data and provides stable results.
- Decision Tree captures non-linear patterns but may slightly overfit.
- KNN performance depends on proper scaling and choice of K.
- Naive Bayes is fast and performs reasonably despite independence assumption.
- Random Forest improves performance by reducing overfitting using bagging.
- XGBoost generally provides strong performance due to boosting and optimized learning.

---

# Streamlit Application Features

The deployed Streamlit web application includes:

- Dataset upload option (CSV file)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Classification report  

---

# Project Structure

ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── model/
│     ├── Logistic_Regression.pkl
│     ├── Decision_Tree.pkl
│     ├── KNN.pkl
│     ├── Naive_Bayes.pkl
│     ├── Random_Forest.pkl
│     ├── XGBoost.pkl
│     └── scaler.pkl

---

# Running Locally

1. Install dependencies:

   pip install -r requirements.txt

2. Run Streamlit:

   streamlit run app.py

---

# Deployment

Live Streamlit App Link: (Add your deployed link here)  
GitHub Repository Link: (Add your repository link here)

---

# Conclusion

This project successfully demonstrates:

- Implementation of multiple classification algorithms  
- Performance comparison using advanced evaluation metrics  
- Use of ensemble learning techniques  
- End-to-end ML deployment using Streamlit  

Ensemble models, particularly XGBoost, provide strong predictive performance 
on the Heart Disease dataset.
