# Heart Disease Predictor — Project Plan

## Goal

Binary classification: predict presence or absence of heart disease using the
UC Irvine Heart Disease dataset (Cleveland subset as primary, with Hungarian,
Switzerland, and VA Long Beach available for extended experiments).

---

## Repository Layout

```
heart-disease-predictor/
├── heart+disease/              # Raw UCI data files
│   ├── cleveland.data          # Full Cleveland data (76 attributes)
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   ├── processed.va.data
│   └── heart-disease.names     # Official attribute documentation
├── PROJECT.md                  # This file
└── (notebooks / scripts to be created)
```

The `processed.*.data` files already contain only the 14 selected attributes
(see Feature Reference below).

---

## Features Used (14 attributes)

| # | Name       | Type              | Description |
|---|-----------|-------------------|-------------|
| 1 | `age`      | Continuous        | Age in years |
| 2 | `sex`      | Binary (0/1)      | 1 = male, 0 = female |
| 3 | `cp`       | Nominal (1–4)     | Chest pain type: 1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic |
| 4 | `trestbps` | Continuous        | Resting blood pressure (mm Hg) |
| 5 | `chol`     | Continuous        | Serum cholesterol (mg/dl) |
| 6 | `fbs`      | Binary (0/1)      | Fasting blood sugar > 120 mg/dl |
| 7 | `restecg`  | Nominal (0–2)     | Resting ECG results: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy |
| 8 | `thalach`  | Continuous        | Maximum heart rate achieved |
| 9 | `exang`    | Binary (0/1)      | Exercise-induced angina |
|10 | `oldpeak`  | Continuous        | ST depression (exercise vs rest) |
|11 | `slope`    | Ordinal (1–3)     | Peak exercise ST slope: 1=upsloping, 2=flat, 3=downsloping |
|12 | `ca`       | Ordinal (0–3)     | Number of major vessels colored by fluoroscopy |
|13 | `thal`     | Nominal (3/6/7)   | 3=normal, 6=fixed defect, 7=reversible defect |
|14 | `num`      | **Target** (0–4)  | Angiographic disease status (binarized: 0 = no disease, 1 = disease) |

---

## Data Handling Steps

### 1. Load Data

- Primary: `processed.cleveland.data` (~303 rows, no header)
- Column names must be assigned manually in the listed order above
- Missing values are encoded as `?` — parse as `na_values="?"`

### 2. Binarize Target

`num` takes values 0–4. Collapse to binary:
- `0` → 0 (no disease: < 50% diameter narrowing)
- `1, 2, 3, 4` → 1 (disease present: > 50% narrowing in at least one vessel)

### 3. Missing Value Imputation

`ca` and `thal` contain the most missing values in this dataset.

| Feature    | Strategy |
|-----------|----------|
| `ca`       | Median imputation (ordinal, small range) |
| `thal`     | Mode imputation (nominal) |
| `trestbps` | Mean or median imputation |
| `chol`     | Mean or median imputation |

With the small Cleveland set (~6 rows missing), dropping missing rows is also
acceptable. For multi-dataset experiments, imputation is preferred.

### 4. Encode Categorical Features

| Feature   | Encoding |
|----------|----------|
| `cp`      | One-hot encode — nominal, 4 values |
| `restecg` | One-hot encode — nominal, 3 values |
| `thal`    | One-hot encode — nominal, non-contiguous values (3/6/7) |
| `slope`   | Treat as ordinal numeric or one-hot — has natural order |
| `ca`      | Treat as numeric ordinal (0–3 vessel count) |
| `sex`, `fbs`, `exang` | Already binary — no change needed |

### 5. Feature Scaling

Apply **StandardScaler** to continuous features: `age`, `trestbps`, `chol`,
`thalach`, `oldpeak`.

**Important:** Fit the scaler on training data only. Transform both train and
test sets using the training fit to avoid data leakage.

### 6. Train/Test Split

- 80/20 stratified split on the target variable
- Use **stratified k-fold cross-validation** (k=5 or k=10) for hyperparameter
  tuning — critical given the small dataset size

---

## Exploratory Data Analysis (EDA)

Before modeling, examine:

- Class balance (target distribution)
- Correlation heatmap — `oldpeak`, `ca`, `thal`, `cp`, `thalach` typically
  correlate most with the target
- Distribution plots (continuous features) by class — look for separability in
  `thalach` and `oldpeak`
- Count plots (categorical features) by class — `cp`, `thal`, `ca` often show
  strong within-category class imbalance

---

## Models to Explore

### Baseline
**Logistic Regression**
- Interpretable; strong baseline for tabular clinical data
- Hyperparameters: `C` (try `[0.01, 0.1, 1, 10]`), `penalty` (`l1`/`l2`/`elasticnet`), `solver` (`liblinear` for l1, `lbfgs` for l2)

### Tree-Based
**Decision Tree**
- Useful for understanding decision logic; prone to overfitting
- Hyperparameters: `max_depth` (`[3–10]`), `min_samples_split`, `min_samples_leaf`, `criterion` (`gini`/`entropy`)

**Random Forest**
- Strong on small tabular datasets; provides feature importance
- Hyperparameters: `n_estimators` (`[100–500]`), `max_depth`, `max_features` (`sqrt`/`log2`), `min_samples_leaf`

**Gradient Boosting (XGBoost or sklearn GBM)**
- Typically the top performer on this dataset
- Hyperparameters: `n_estimators`, `learning_rate` (`[0.01–0.3]`), `max_depth` (`[3–6]`), `subsample` (`[0.6–1.0]`)
- XGBoost extras: `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`

### Kernel / Distance-Based
**SVM (SVC)**
- Effective on small datasets; requires scaled features
- Hyperparameters: `C`, `kernel` (`rbf`/`linear`), `gamma` (`scale`/`auto` or numeric)

**K-Nearest Neighbors**
- Simple comparison baseline
- Hyperparameters: `n_neighbors` (`[3–15]`), `metric` (`euclidean`/`manhattan`), `weights` (`uniform`/`distance`)

### Probabilistic
**Gaussian Naive Bayes**
- Fast and surprisingly competitive on medical data
- Hyperparameters: `var_smoothing`

---

## Evaluation Metrics

Accuracy alone is insufficient for clinical classification.

| Metric | Notes |
|--------|-------|
| **ROC-AUC** | Primary metric — threshold-independent |
| **Recall (Sensitivity)** | Prioritized clinically — false negatives (missed disease) are costly |
| **Precision** | Track alongside recall |
| **F1-Score** | Harmonic mean for overall summary |
| **Confusion Matrix** | Explicit FP/FN counts for each model |

---

## Recommended Workflow

```
1. Load processed.cleveland.data, assign column names
2. Parse missing values (?), binarize target
3. EDA (distributions, correlations, class balance)
4. Impute missing values
5. Encode categoricals, scale continuous features
6. Stratified 80/20 train/test split
7. For each model: stratified cross-validated GridSearchCV / RandomizedSearchCV
8. Evaluate best models on held-out test set
9. Feature importance analysis on the best model
10. (Optional) Repeat with combined Cleveland + Hungarian datasets
```

---

## Notes for Extended Experiments

- **Multi-dataset training:** `processed.hungarian.data`, `processed.switzerland.data`,
  and `processed.va.data` use the same 14-feature schema and can be combined.
  Class distributions differ significantly across sites — consider stratified
  sampling or site as an additional feature.
- **Multi-class option:** Keep `num` as 0–4 to model severity. Switch from
  binary cross-entropy to softmax; ROC-AUC becomes macro-averaged.
- **Feature interactions:** Consider adding `age × thalach` or `oldpeak × slope`
  as engineered features — both have known clinical interaction effects.
