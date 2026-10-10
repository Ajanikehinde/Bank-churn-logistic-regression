# Bank Customer Churn: Logistic Regression and Random Forest

## Overview

This project develops and evaluates machine learning models for predicting customer churn using a bank customer dataset.

The analysis begins with Logistic Regression as an interpretable baseline and then evaluates the effect of class balancing before comparing the results with Random Forest models.

A key focus is **class imbalance**. Only around 20% of customers in the dataset are churners, so accuracy alone can give a misleading impression of model performance. The project therefore evaluates Accuracy, Precision, Recall, F1-score, and ROC-AUC.

The final model selected in the notebook is a **class-weighted Random Forest**.

---

## Objectives

* Predict whether a bank customer will churn (`Exited`).
* Establish a Logistic Regression baseline.
* Evaluate the impact of class weighting on churn detection.
* Compare Logistic Regression with Random Forest.
* Evaluate models using metrics beyond accuracy.
* Verify and correct the categorical encoding approach.
* Identify important features associated with model predictions.

---

## Dataset

The project uses the Bank Customer Churn Modelling dataset.

The target variable is:

* `Exited` — whether the customer exited/churned.

The original dataset contains customer, demographic, account, and activity information.

### Features removed

The following columns were excluded because they are identifiers or names rather than useful predictive features:

* `CustomerId`
* `RowNumber`
* `Surname`

### Selected features

The modelling pipeline uses:

**Numerical features**

* `CreditScore`
* `Age`
* `Balance`
* `NumOfProducts`

**Categorical features**

* `Geography`
* `Gender`

**Additional retained features**

* `IsActiveMember`

`HasCrCard`, `Tenure` and `EstimatedSalary` are excluded from the final feature set. This choice was made for the logistic regression models and has **not** been shown to be harmless for the Random Forest: the companion project [`bank-churn-classification`](https://github.com/Ajanikehinde/bank-churn-classification) keeps all features and reports a higher Random Forest ROC-AUC (0.8542 vs. 0.8400 here for the default forest). See [Relationship to the Companion Project](#relationship-to-the-companion-project).

---

## Exploratory Data Analysis

The analysis checks:

* Missing values
* Duplicate records
* Class distribution
* Numerical feature distributions
* Categorical variables
* Relationships between customer characteristics and churn

The churn target is imbalanced, with approximately 20% of customers classified as churners.

An always-predict-no-churn baseline achieves:

**Accuracy: 0.7965**

This demonstrates why accuracy alone is insufficient for evaluating the churn models.

---

## Data Preprocessing

Categorical variables are encoded using `OneHotEncoder` and numerical variables are standardised using `StandardScaler`.

The preprocessing pipeline uses `ColumnTransformer` so that transformations are applied explicitly to the relevant columns.

The categorical encoding was also corrected to use:

```python
OneHotEncoder(drop='first')
```

This avoids redundant dummy-variable representations for the categorical features.

The train/test split uses:

```python
train_test_split(
    new_data,
    y,
    train_size=0.8,
    random_state=42,
    stratify=y
)
```

This preserves the class distribution between the training and test sets.

---

## Logistic Regression Baseline

The initial Logistic Regression model uses the default class weighting.

### Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.8095 |
| Precision | 0.5985 |
| Recall    | 0.1941 |
| F1-score  | 0.2931 |
| ROC-AUC   | 0.7753 |

Although the accuracy is above the trivial 0.7965 baseline, the model identifies only **19.41% of actual churners**.

This highlights the importance of evaluating recall and F1-score for an imbalanced churn problem.

---

## Handling Class Imbalance

A second Logistic Regression model uses:

```python
class_weight='balanced'
```

### Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.7145 |
| Precision | 0.3892 |
| Recall    | 0.7076 |
| F1-score  | 0.5022 |
| ROC-AUC   | 0.7779 |

Compared with the default Logistic Regression model, class weighting substantially increases recall from **0.1941 to 0.7076**.

However, this comes with lower precision and overall accuracy.

This illustrates the trade-off between identifying more potential churners and reducing false positives.

---

## Random Forest Comparison

Two Random Forest configurations were evaluated:

* Random Forest
* Random Forest with `class_weight='balanced'`

Both models use:

```python
n_estimators=300
random_state=42
```

### Model comparison

| Model                          |   Accuracy |  Precision |     Recall |         F1 |    ROC-AUC |
| ------------------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression (default)  |     0.8095 |     0.5985 |     0.1941 |     0.2931 |     0.7753 |
| Logistic Regression (balanced) |     0.7145 |     0.3892 | **0.7076** |     0.5022 |     0.7779 |
| Random Forest (default)        |     0.8530 |     0.7055 |     0.4767 |     0.5689 | **0.8400** |
| Random Forest (balanced)       | **0.8540** | **0.7106** |     0.4767 | **0.5706** |     0.8393 |

The Random Forest models provide substantially higher ROC-AUC than either Logistic Regression configuration on this test split. Balanced Logistic Regression has the highest recall (0.7076), at the cost of precision (0.3892); the balanced Random Forest has the highest accuracy, precision and F1.

The balanced Random Forest records the highest accuracy, precision, and F1-score among the tested models, while the default Random Forest has the highest ROC-AUC.

---

## Final Model

The notebook selects **Random Forest (balanced)** as the final model:

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42
)
```

Its test-set performance is:

* **Accuracy:** 85.40%
* **Precision:** 71.06%
* **Recall:** 47.67%
* **F1-score:** 57.06%
* **ROC-AUC:** 83.93%

The final model is subsequently evaluated using:

* Confusion matrix
* ROC curve
* Feature importance analysis

---

## Encoding Verification

The notebook includes an explicit verification of the categorical encoding correction.

The original encoding without dropping one category produced:

* `Gender_Female`: coefficient = -0.014
* `Gender_Male`: coefficient = -0.539

After using `drop='first'`, the `Gender_Male` coefficient was:

* `Gender_Male`: coefficient = -0.526

The resulting Logistic Regression accuracy remained identical:

* As-written encoding: **0.8095**
* Fixed encoding: **0.8095**

This confirms that the encoding correction did not change the measured test accuracy for the baseline Logistic Regression model.

---

## Key Findings

1. **Accuracy alone is insufficient for churn prediction.**
   The trivial always-no-churn baseline achieves 79.65% accuracy despite identifying no churners.

2. **Default Logistic Regression has limited churn recall.**
   Its recall is 19.41%, meaning many actual churners are missed.

3. **Class weighting changes the Logistic Regression trade-off substantially.**
   Recall increases to 70.76%, while precision and accuracy decrease.

4. **Random Forest provides stronger discrimination on this test split.**
   Both Random Forest configurations achieve ROC-AUC around 0.84 compared with approximately 0.78 for Logistic Regression.

5. **The balanced Random Forest achieves the highest F1-score among the tested models.**
   Its F1-score is 0.5706.

6. **Model selection should depend on the business cost of false positives and false negatives.**
   The current notebook evaluates standard classification metrics but does not perform business-cost-based threshold optimisation.

---

## Relationship to the Companion Project

[`bank-churn-classification`](https://github.com/Ajanikehinde/bank-churn-classification) uses the same dataset and the same stratified 80/20 split (`random_state=42`). The two projects differ in the following ways, so their scores are not directly comparable:

| | This project | `bank-churn-classification` |
| --- | --- | --- |
| Features | 7 (drops `Tenure`, `HasCrCard`, `EstimatedSalary`) | All 10 |
| Class weighting | Tested (`class_weight='balanced'`) | Not used |
| Random Forest ROC-AUC | 0.8400 (default), 0.8393 (balanced) | 0.8542 |

The default Random Forests in both projects use 300 trees and no class weighting, so the 0.8400 vs. 0.8542 gap is most likely due to the dropped features (`EstimatedSalary` and `Tenure` carry real importance in the full-feature forest). This has not been tested directly in this notebook.

---

## Limitations and Future Improvements

The current analysis has several limitations:

* The models are evaluated using a single stratified 80/20 train/test split.
* Cross-validation is not included.
* The Random Forest uses 300 trees but does not undergo systematic hyperparameter tuning.
* Gradient boosting models were not evaluated.
* `class_weight='balanced'` is a general approach to class imbalance and may not reflect the actual business cost of churn errors.
* Classification threshold optimisation was not performed.
* Probability calibration was not assessed.
* The decision to drop `HasCrCard`, `Tenure` and `EstimatedSalary` was made for logistic regression and is not shown to hold for the Random Forest; the companion project's all-feature forest scores higher.
* The results should be validated on unseen or future customer data before operational deployment.

Potential future improvements include:

* Re-run the Random Forest on all 10 features to test whether dropping `Tenure`, `HasCrCard` and `EstimatedSalary` costs ROC-AUC
* Stratified cross-validation
* Hyperparameter optimisation
* Gradient Boosting / XGBoost-style models
* Threshold optimisation based on business costs
* Probability calibration
* SHAP or other explainability techniques
* Customer-level churn risk segmentation
* Cost-sensitive evaluation

---

## Technologies and Skills

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Exploratory Data Analysis
* Feature preprocessing
* One-hot encoding
* Feature scaling
* Logistic Regression
* Random Forest
* Classification
* Imbalanced classification
* Model evaluation
* ROC-AUC
* Confusion matrices
* Feature importance
* Data-driven decision making

---

## Repository Structure

```text
bank-churn-logistic-regression/
│
├── Churn_Logistic_Regression_corrected.ipynb
└── README.md
```

The repository intentionally contains only the final notebook and project documentation. Dataset files, intermediate notebooks, checkpoints, and temporary inspection scripts are excluded.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Ajanikehinde/bank-churn-logistic-regression.git
cd bank-churn-logistic-regression
```

### 2. Open the notebook

Launch Jupyter Notebook or JupyterLab and open:

```text
Churn_Logistic_Regression_corrected.ipynb
```

### 3. Dataset

The dataset is intentionally not committed to the repository. Place the required dataset in the expected local `data/` directory before executing the notebook.

---

## Reproducibility

The analysis uses fixed random states for the train/test split and machine learning models.

The main train/test split uses:

```python
random_state=42
```

The Random Forest models use:

```python
n_estimators=300
random_state=42
```

The reported metrics are from the executed notebook's 80/20 stratified test split.
