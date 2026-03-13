# Titanic - Machine Learning from Disaster

**Kaggle Username:** salmankhamis

**Best Leaderboard Score:** 79.186% ✅

---

## Project Overview

Predicting passenger survival on the Titanic using machine learning. The dataset includes passenger information such as age, sex, class, and fare. The goal is to build a model that accurately predicts whether a given passenger survived.

---

## Feature Engineering

The following features were engineered to improve model performance:

### Name → Title
Extracted passenger titles from the `Name` column and grouped them into meaningful categories:
- `Mr`, `Mrs`, `Ms` — standard titles
- `Military` — Capt, Col, Major
- `Nobility` — Don, Dona, Sir, Lady, Jonkheer, Master, the Countess

### Age Imputation
Missing ages were filled using the **median age per passenger class (Pclass)**, preserving class-based age distributions rather than using a global median.

### Age Binning
Ages were binned into 10-year intervals (`0–10`, `10–20`, ..., `70–80`) to reduce noise from treating age as a continuous variable.

### Fare per Ticket
Some tickets were shared between multiple passengers, so the fare was divided by the number of passengers sharing the same ticket number (`Fare_per_Ticket`). This gives a more accurate individual fare cost.

### Fare Binning
`Fare_per_Ticket` was binned into ranges: `[0–20, 20–40, 40–60, 60–80, 80–150]`.

### Cabin
Missing cabin values were filled with `"M"` (missing). Only the first letter of the cabin was kept to represent the cabin deck. The rare `"T"` cabin was mapped to `"M"`.

### Family Size & Alone Flag
- `Num_Family = SibSp + Parch + 1` — total number of family members including the passenger
- `Is_Alone = 1` if traveling alone, `0` otherwise

### Woman in 3rd Class
`Woman_3rd` flag — a binary feature indicating female passengers in 3rd class, who faced significantly worse survival odds despite being female.

### One-Hot Encoding
Categorical features (`Sex`, `Embarked`, `Title`, `Cabin`, `Age_bin`, `Fare_Bin`, `Pclass`) were one-hot encoded for model input.

---

## Model

**Random Forest Classifier** with the following hyperparameters (tuned via `GridSearchCV` with 5-fold cross-validation):

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `max_depth` | 6 |
| `max_features` | log2 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |

The model was first validated on a 80/20 train/test split, then retrained on the full training set before generating Kaggle predictions.

---

## Results

| Split | Accuracy |
|---|---|
| Validation (20% holdout) | ~79% |
| Kaggle Leaderboard | **79.186%** ✅ |

---

## Repository Structure

```
project/
│   README.md
│   requirements.txt
│