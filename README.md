```
# Thiranex Data Science Internship - Week 2: Predictive Modeling Using Machine Learning

## 📌 Project Overview
This repository contains the completed work for **Task 2 (Predictive Modeling Using Machine Learning)** of the Thiranex Data Science Internship.

The goal of this project is to build, evaluate, and compare supervised machine learning models to predict classification outcomes based on demographic and financial variables (`Age`, `Income`, `Credit_Score`, `Years_Employed`).

---

## 🛠️ Machine Learning Pipeline Workflow

1. **Dataset Preparation &amp; Preprocessing**:
   - Generated/loaded target binary classification features (`Age`, `Income`, `Credit_Score`, `Years_Employed`, `Target`).
   - Stratified dataset into an **80% training** and **20% testing** split (`train_test_split`).
   - Applied **`StandardScaler`** to normalize numerical features prior to model fitting.

2. **Multi-Model Training**:
   - Trained three distinct classification algorithms:
     * **Logistic Regression**
     * **Decision Tree Classifier**
     * **Random Forest Classifier**

3. **Evaluation &amp; Performance Comparison**:
   - Calculated key classification metrics: **Accuracy**, **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**.
   - Plotted **Confusion Matrices** and **ROC Curves** to evaluate true vs. false predictions and trade-offs across classification thresholds.

---

## 📊 Model Comparison &amp; Results

Below is the comparative evaluation table across the three models on the test dataset [9]:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8333 | 0.8472 | 0.8714 | 0.8592 | **0.9331** 🏆 |
| **Decision Tree** | 0.8667 | 0.8857 | 0.8857 | 0.8857 | 0.8871 |
| **Random Forest** | 0.8167 | 0.8158 | 0.8857 | 0.8493 | 0.9234 |

### 🏆 Best Model Selection
* **Selected Model**: **Logistic Regression**
* **Justification**: Although the Decision Tree achieved slightly higher accuracy, **Logistic Regression** demonstrated superior overall class separation capability with the highest **ROC-AUC score of 0.9331** [9].

---

## 📈 Visualizations Saved

* **`model_comparison_confusion_matrices.png`**: Heatmaps displaying True Positives, True Negatives, False Positives, and False Negatives for each classifier.
* **`model_comparison_roc.png`**: Combined ROC Curve chart illustrating performance against the random classifier baseline (AUC = 0.50).

---

## 📁 Repository Structure

```text
Thiranex_Week_2/
│
├── Predictive_Modelling_using_ML.ipynb  # Main Jupyter Notebook
├── sample_ml_dataset.csv                # Project dataset
├── model_comparison_roc.png             # Combined ROC Curve plot
├── model_comparison_confusion_matrices.png  # Subplotted Confusion Matrices
└── README.md                            # Documentation for Task 2

```

---

## 💻 Tech Stack &amp; Libraries Used

* **Language**: Python 3
* **Environment**: Google Colab / JupyterLab
* **Libraries**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
