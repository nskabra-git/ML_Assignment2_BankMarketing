
# ML Assignment 2 – Bank Marketing Classification

## 🔍 Problem Statement

The objective of this project is to build and compare multiple machine learning models to predict whether a client will subscribe to a term deposit based on marketing campaign data.

The target variable:
- `y = 1` → Client subscribed to term deposit
- `y = 0` → Client did not subscribe

---

## 📊 Dataset

Dataset: Bank Marketing Dataset (UCI)

- Total records: 45,211
- Features: 16 input features
- Target: Binary classification (`yes` / `no`)
- Class imbalance:
  - No ≈ 88%
  - Yes ≈ 12%

Due to class imbalance, multiple evaluation metrics were used instead of relying only on accuracy.

---

## 🏗 Project Structure
ML_Assignment2_BankMarketing/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
│
├── data/
│ └── bank-full.csv
│
├── notebooks/
│ └── exploration.ipynb
│
├── model/
│ ├── preprocessing.py
│ ├── evaluate.py
│ ├── train_models.py
│ ├── *.pkl models
│ ├── sample_test_data.csv
│ └── sample_test_with_target.csv


---

## ⚙️ Models Implemented

Six classification models were implemented and compared:

1. Logistic Regression  
2. Decision Tree  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes  
5. Random Forest  
6. XGBoost  

All models were built using a shared preprocessing pipeline:
- One-hot encoding for categorical variables
- Standard scaling for numerical variables
- Stratified train-test split (80/20)

---

## 📈 Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC (ROC)
- Matthews Correlation Coefficient (MCC)

Accuracy alone was not sufficient due to class imbalance.

---

## 🏆 Model Comparison (Test Set)

| Model | Accuracy | AUC | F1 Score | MCC |
|-------|----------|------|----------|------|
| Logistic Regression | 0.846 | 0.908 | 0.553 | 0.509 |
| Decision Tree | 0.878 | 0.713 | 0.488 | 0.419 |
| KNN | 0.896 | 0.837 | 0.444 | 0.407 |
| Naive Bayes | 0.864 | 0.809 | 0.456 | 0.380 |
| Random Forest | 0.904 | 0.927 | 0.486 | 0.456 |
| XGBoost | **0.908** | **0.929** | **0.561** | **0.515** |

---

## 📌 Key Observations

- **XGBoost** achieved the best overall performance across Accuracy, AUC, F1 Score, and MCC.
- **Logistic Regression** achieved the highest Recall, making it suitable when minimizing false negatives is important.
- Tree-based ensemble methods (Random Forest, XGBoost) handled class imbalance better than simple models.
- Accuracy alone would have been misleading due to the 88/12 class distribution.

---

## 💻 Streamlit Application

The deployed application allows:

- Model selection via dropdown
- Downloading a sample test dataset
- Uploading a custom test dataset (CSV)
- Viewing:
  - Evaluation metrics
  - Confusion matrix (heatmap)
  - Tabulated classification report
  - Downloadable predictions

### 🔗 Live App:
https://2025aa05719-nitin-shriram-kabra-ml-assignment2.streamlit.app/

---

## 📦 Installation (Local Run)

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py


|        ML Model Name      | Accuracy |   AUC     | Precision | Recall   |  F1 Score |   MCC      |
|---------------------------|---------:|----------:|----------:|---------:|----------:|-----------:|
| Logistic Regression       | 0.845958 |  0.907837 | 0.418571  | 0.813800 |  0.552809 |  0.509131  |
| Decision Tree             | 0.877695 |  0.713460 | 0.478261  | 0.499055 |  0.488437 |  0.419140  |
| KNN                       | 0.896052 |  0.837260 | 0.593060  | 0.355388 |  0.444444 |  0.406696  |
| Naive Bayes               | 0.863873 |  0.808785 | 0.428216  | 0.487713 |  0.456032 |  0.379655  |
| Random Forest             | 0.904456 |  0.927197 | 0.655449  | 0.386578 |  0.486326 |  0.456080  |
| XGBoost                   | 0.907995 |  0.929058 | 0.634845  | 0.502836 |  0.561181 |  0.514894  |
