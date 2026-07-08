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
├── scripts/
│   ├── parse_data.py           # Raw site files → data/combined.csv
│   └── clean_data.py           # combined.csv → reduced + train/val/test sets
├── data/                       # Generated CSVs (see Scripts section)
├── notebooks/                  # EDA and model exploration
├── models/                     # Saved model artifacts
└── PROJECT.md                  # This file
```

The `processed.*.data` files already contain only the 14 selected attributes
(see Feature Reference below).

---

## Scripts

The preprocessing pipeline is split into two scripts. Run them in order from
the project root (`py scripts/parse_data.py` then `py scripts/clean_data.py`).

### `scripts/parse_data.py` — parse and combine

Reads the four `processed.{cleveland,hungarian,switzerland,va}.data` files,
assigns the 14 column names, and concatenates them into a single table with a
leading `source` column identifying the collection site.

- **Missing values** (`?`) are parsed as NA and written as empty fields.
- **`reprocessed.hungarian.data` is skipped** — it duplicates
  `processed.hungarian.data` in a different encoding.
- **`chol == 0` is treated as missing** — a zero cholesterol encodes "not
  measured" (all of Switzerland, a few VA rows), not a real value.
- Nullable integer dtypes keep integer columns from being written as floats.

**Output:** `data/combined.csv` — 920 rows × 15 columns (`source` + 14 features).

### `scripts/clean_data.py` — reduce, split, and impute

Turns `combined.csv` into model-ready, leak-free train/test sets.

0. **Drops `source`** — the collection-site label is kept in `combined.csv`
   for reference but is not used as a model feature.
1. **Reports missingness** per feature on the full data.
2. **Drops configured columns** (`DROP_COLUMNS`, default `ca`, `thal`, `slope`
   — the three sparsest features, ~34–66% missing) and writes
   `data/combined_reduced.csv`. Missingness is reported again after the cut.
3. **Splits into train/validation/test first** (60/20/20, stratified on the
   binarized target) — *before* any imputation, so fill values are never
   learned from the held-out rows.
4. **Imputes with statistics fit on the training set only:**
   - `chol` → **iterative imputation** (`IterativeImputer`), regressed on the
     other features; the `num` target is excluded to avoid leakage. Results
     are rounded back to whole mg/dl.
   - `trestbps`, `thalach`, `oldpeak` → **median** imputation.
   - `fbs`, `restecg`, `exang` (and `slope`/`thal` if kept) → **mode**
     imputation.
5. **One-hot encodes the nominal features** (`cp`, `restecg`) before the
   iterative imputer runs, so categories are never treated as ordered numbers.
   Test columns are reindexed to the train columns so both sets align.

**Output:** `data/combined_train.csv` (552 rows), `data/combined_val.csv`
(184 rows), and `data/combined_test.csv` (184 rows), each 16 columns with zero
missing values.

**Note on iterative imputation:** predicting the conditional mean shrinks the
variance of the imputed `chol` values (they cluster near the center), so ~22%
of that column is low-variance synthetic data. Keep this in mind when reading
`chol` feature importance.

---

## Model Exploration (planned)

> Status: designed, not yet implemented. Scripts below are built one at a time.

The modeling stage compares several classifier families on the cleaned splits,
prioritizing **ROC-AUC** and **recall** (a missed diagnosis is the costly error).

### Target framing

Binarize the target: `y = (num > 0)` — disease present vs. absent. The raw 0–4
`num` column stays in the CSVs; binarization happens at load time.

### Split usage (60 / 20 / 20)

- **train (552)** — hyperparameter tuning via `GridSearchCV` with
  `StratifiedKFold` (k=5), scoring `roc_auc`.
- **val (184)** — model selection: compare each tuned model, pick the best by
  ROC-AUC (recall as tie-breaker).
- **test (184)** — touched once at the end, on the selected model refit on
  train+val. Final reported numbers only.

### Scaling policy (per-model, per-column)

Standardizing the whole matrix would distort the one-hot/binary columns. Instead
a `ColumnTransformer` scales **only the 5 continuous features** (`age`,
`trestbps`, `chol`, `thalach`, `oldpeak`) with `StandardScaler` and passes the
binary/dummy columns through unchanged, so neither group dominates. Scaling is
applied only for the distance/kernel/linear models (**LogReg, SVM, KNN**); tree
models and Naive Bayes use the raw features. The scaler lives inside each
`Pipeline`, so it is refit within every CV fold and never sees held-out rows.

### Planned scripts

**`scripts/dataset.py`** — shared helper: feature-group constants, `load_splits()`,
`get_xy(df)` (with binarization), and `build_scaler()` (continuous-only
`ColumnTransformer`). Reused by the training script and the notebook.

**`scripts/train_models.py`** — a `MODELS` registry of the seven families below,
each with an estimator, hyperparameter grid, and a `needs_scaling` flag. For each:
build a `Pipeline`, tune with `GridSearchCV` on train, evaluate on val, then select
the best and report final test metrics. Outputs `models/comparison.csv` (per-model
val metrics) and `models/best_model.joblib`.

| Model | Scaling | Key hyperparameters |
|---|---|---|
| LogisticRegression | yes | `C`, `penalty`/`solver` |
| SVC (`probability=True`) | yes | `C`, `kernel`, `gamma` |
| KNeighborsClassifier | yes | `n_neighbors`, `weights`, `metric` |
| DecisionTreeClassifier | no | `max_depth`, `min_samples_leaf`, `criterion` |
| RandomForestClassifier | no | `n_estimators`, `max_depth`, `max_features` |
| GradientBoostingClassifier | no | `n_estimators`, `learning_rate`, `max_depth` |
| HistGradientBoostingClassifier | no | `learning_rate`, `max_iter`, `max_depth` |
| GaussianNB | no | `var_smoothing` |

(HistGradientBoosting is the fast sklearn boosting stand-in for XGBoost, which is
intentionally not a dependency.)

**`notebooks/model_exploration.ipynb`** — matplotlib EDA (class balance, continuous
distributions by class, correlation heatmap, categorical counts) plus modeling
results: comparison table, overlaid ROC curves, confusion matrices, metric bar chart.

### Metrics reported per model

ROC-AUC (primary), recall (sensitivity), precision, F1, accuracy, and the confusion
matrix.

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
