# Heart Disease Prediction

Predicts whether a patient has heart disease from clinical measurements, using classic machine learning classifiers.

## Problem Statement
Given patient attributes such as age, blood pressure, cholesterol and ECG results, classify whether the person has heart disease (`HeartDisease` = 1) or not (0).

## Dataset
- 918 rows, 12 columns (11 features + target)
- Features: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
- Target: `HeartDisease`, which is almost balanced, so no resampling was needed
- File used: `heart.csv` (not included in this repo; add your own copy)

## Approach
1. **EDA**: summary statistics, correlation heatmap, pairplot, boxplots for outliers, countplots for categorical features
2. **Cleaning**: removed one row with `RestingBP = 0` (invalid value); kept other outliers since they were plausible
3. **Encoding**: `LabelEncoder` on categorical columns
4. **Split**: 70/30 train-test split, stratified on the target (`random_state=123`)
5. **Models**: Logistic Regression, SVM (kernel chosen with `GridSearchCV`), Decision Tree, plus KNN as an extra

## Results (test set, 276 samples)
| Model | Accuracy |
|---|---|
| **SVM (linear kernel)** | **85%** |
| Decision Tree | 77% |
| KNN (k=7) | 70% |

The SVM also had balanced precision and recall (about 0.85 to 0.87) for both classes.

## Tech Stack
Python, pandas, seaborn, matplotlib, scikit-learn

## How to Run
1. Place `heart.csv` in your working directory and update the path in the notebook
2. Open `Heart.ipynb` in Jupyter or Colab and run all cells

## Possible Improvements
- Feature scaling before SVM/KNN
- One-hot encoding for nominal features instead of label encoding
- Cross-validated comparison and ROC-AUC reporting
