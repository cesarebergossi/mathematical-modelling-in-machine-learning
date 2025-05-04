# Multiclass Classification with SVM vs. Logistic Regression

This project explores and compares two classical machine learning models — **Linear Support Vector Machine** and **Multinomial Logistic Regression** — for multiclass classification on a structured dataset. The project was developed for the *Mathematical Modelling in Machine Learning* course during my BSc in Mathematical and Computing Sciences for Artificial Intelligence.


## Problem Overview

Given a labeled dataset with 1100 samples and 20 numerical features, the objective was to build a model capable of predicting one of **five possible classes**. Two models were tested and compared:

- `LinearSVC` from `sklearn.svm`  
- `LogisticRegression` from `sklearn.linear_model`

The main goal was to optimize each model’s parameters and select the best-performing method based on validation performance.

## Dataset Description

- **Rows**: 1100 samples (plus a separate unlabeled test set)
- **Columns**:
  - Column 0: Sample ID  
  - Column 1: Class label (0 to 4)  
  - Columns 2–21: Numerical features

A separate **test dataset** (`Test_Dataset.csv`) without labels was provided for final predictions.


## Workflow Summary

1. **Exploratory Data Analysis (EDA)**  
   - Feature distribution inspection  
   - Class balance and correlation analysis

2. **Preprocessing**  
   - Feature scaling using `StandardScaler`  
   - Train-validation split

3. **Model Training**  
   - Grid search over regularization parameters `C` for both models  
   - Cross-validation to assess robustness  
   - Hyperparameter tuning and selection based on validation accuracy

4. **Evaluation**  
   - Final model trained on full training set  
   - Predictions made on the test set  
   - Model performance analyzed via accuracy and confusion matrices (on validation set)


## Final Choice

After experimentation, **Multinomial Logistic Regression** with optimal regularization was selected due to its superior performance in terms of cross-validation accuracy and stability.

## Repository Contents
```
Project.ipynb           # Jupyter notebook with full analysis
Dataset.csv             # Training dataset with labels
Test_Dataset.csv        # Unlabeled test dataset
Predictions.txt         # Final test predictions (one label per line)
Description.txt         # Task description provided by course instructor
```

## How to Run

1. Open `Project.ipynb` in Jupyter Notebook  
2. Install required libraries (if needed): `scikit-learn`, `matplotlib`, `pandas`, `numpy`  
3. Run cells from top to bottom to reproduce results and generate predictions
