# Titanic Survival Prediction — ML Classification

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview

End-to-end machine learning pipeline to predict passenger survival on the Titanic
using advanced feature engineering, hyperparameter tuning via RandomizedSearchCV
and GridSearchCV, and two classification algorithms — Random Forest and AdaBoost.

Built and run in **VS Code** using Python scripts (not Jupyter Notebook).

---

## Key Highlights

- Custom **title extraction** from passenger names (Mr, Mrs, Ms, Military, Nobility)
- **Age imputation by passenger class** using class-wise medians (not global median)
- **Fare-per-ticket** feature — splits shared ticket fares for accurate per-person cost
- **Age binning and Fare binning** to reduce noise in continuous variables
- **Family size** feature engineered from SibSp + Parch
- Two-stage hyperparameter tuning: RandomizedSearchCV (broad) → GridSearchCV (fine)
- Both **Random Forest** and **AdaBoost** classifiers trained and compared

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic) |
| Training rows | 891 |
| Test rows | 418 |
| Features (raw) | 12 |
| Target | Survived (0 = No, 1 = Yes) |

---

## Project Structure

```
titanic-survival-prediction/
│
├── titanic.py          # Main ML pipeline script
├── train.csv           # Training dataset (from Kaggle)
├── test.csv            # Test dataset (from Kaggle)
└── README.md
```

---

## Pipeline — Step by Step

### 1. Data Cleaning
- Combined train and test sets for consistent preprocessing
- **Age** — filled missing values with **median age per passenger class**
  (1st class ~37, 2nd class ~29, 3rd class ~24)
- **Fare** — filled 1 missing value with median fare of similar passengers
  (Pclass=3, Embarked=S)
- **Embarked** — filled 2 missing values with 'C' based on fare/class analysis
- **Cabin** — extracted deck letter (first character); filled missing with 'M';
  reassigned rare 'T' cabin to 'M'

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `Title` | Extracted from name — grouped into Mr, Mrs, Ms, Dr, Military, Nobility |
| `Age_Bins` | Age bucketed into 10-year intervals [0–10, 10–20, ..., 70–80] |
| `Fare_per_ticket` | Fare divided by number of passengers sharing the same ticket |
| `Fare_bins` | Fare-per-ticket bucketed into 5 ranges [0–20, 20–40, ..., 80–150] |
| `Num_family` | SibSp + Parch + 1 (self) — total family size on board |

### 3. Encoding & Scaling
- **Label Encoding** — applied to all categorical columns
- **One-Hot Encoding** — applied to Sex, Embarked, Title
- **MinMaxScaler** — scaled all features to [0, 1] range

### 4. Models & Tuning

#### Random Forest Classifier
```
Stage 1 — RandomizedSearchCV (cv=5)
  Parameters searched: criterion, n_estimators, max_depth,
                       min_samples_split, max_features, max_samples

Stage 2 — GridSearchCV (cv=5)
  Fine-tuned around best RandomizedSearch parameters
```

#### AdaBoost Classifier
```
Stage 1 — RandomizedSearchCV (cv=5)
  Parameters searched: n_estimators [50–1000], learning_rate [0.001–1.0]

Stage 2 — GridSearchCV (cv=5)
  Fine-tuned n_estimators and learning_rate around best params
```

---

## Results

| Model | Tuning | Train Accuracy | Test Accuracy |
|---|---|---|---|
| Random Forest | RandomizedSearchCV | ~% | ~% |
| Random Forest | GridSearchCV | ~% | ~% |
| AdaBoost | RandomizedSearchCV | ~% | ~% |
| AdaBoost | GridSearchCV | ~% | ~% |

> Replace `~%` with your actual output values from the terminal

---

## Tech Stack

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
  - RandomForestClassifier
  - AdaBoostClassifier
  - RandomizedSearchCV / GridSearchCV
  - MinMaxScaler / LabelEncoder
- Seaborn
- Matplotlib
- VS Code

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/rutwikbelsare/titanic-survival-prediction
cd titanic-survival-prediction

# Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib

# Download dataset from Kaggle and place in root folder
# train.csv and test.csv must be in the same directory as titanic.py

# Run the script
python titanic.py
```

---

## What I Would Do Next

- Add XGBoost / LightGBM for potentially higher accuracy
- Build a Streamlit UI to predict survival for custom passenger inputs
- Submit final predictions to Kaggle and track leaderboard score
- Add SHAP values for model explainability

---

## Author

**Rutwik Belsare**
MCA 2026 | Zeal Institute of Business Administration, Pune
[LinkedIn](https://linkedin.com/in/rutwikbelsare) · [GitHub](https://github.com/rutwikbelsare)
