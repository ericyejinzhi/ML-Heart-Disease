# Heart Disease Predictor

Binary classification of heart disease presence from the
[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(Cleveland, Hungarian, Switzerland, and VA Long Beach sites combined - 920
patients). Eight scikit-learn model families are tuned and compared under a
strict train/validation/test protocol; the selected model is evaluated on the
test set exactly once.

**Result:** random forest, chosen on validation ROC-AUC 0.868 / recall 0.902,
scores **ROC-AUC 0.907** and **recall 0.863** on the held-out test set.

See [PROJECT.md](PROJECT.md) for the full plan, feature reference, and design
decisions.

## Repository layout

```
heart-disease-predictor/
├── heart+disease.zip           # Raw UCI archive (extract to heart+disease/)
├── scripts/
│   ├── parse_data.py           # Raw site files → data/combined.csv
│   ├── clean_data.py           # combined.csv → reduced + train/val/test CSVs
│   ├── dataset.py              # Shared loaders + continuous-only scaler
│   └── train_models.py         # Tune/compare all 8 families, batch version
├── notebooks/
│   ├── <model>.ipynb           # One notebook per family (8 total)
│   └── comparison.ipynb        # Cross-model comparison + single test eval
├── data/                       # Generated CSVs (git-ignored)
├── models/                     # comparison.csv output (git-ignored)
└── PROJECT.md                  # Project plan and methodology
```

## Setup

Requires Python 3.12+ with `pandas`, `scikit-learn`, and `matplotlib`
(plus `jupyter` for the notebooks):

```
pip install pandas scikit-learn matplotlib jupyter
```

Then, from the project root:

```
# 1. Extract the raw UCI files (once)
#    heart+disease.zip → heart+disease/

# 2. Parse the four site files into one table
py scripts/parse_data.py        # → data/combined.csv

# 3. Reduce, split 60/20/20, impute (train-fit only)
py scripts/clean_data.py        # → data/combined_{train,val,test}.csv

# 4. Tune and compare all eight families
py scripts/train_models.py      # → models/comparison.csv + test metrics
```

The notebooks import their estimators, grids, and loaders from
`scripts/train_models.py` and `scripts/dataset.py`, so they reproduce the same
numbers as the batch script (CV splits are seeded).

## Method

- **Target** - `num` (0-4 angiographic status) binarized to disease present /
  absent at load time.
- **Features** - 10 of the 14 standard attributes; `ca`, `thal`, and `slope`
  are dropped (34-66% missing across sites). `cp` and `restecg` are one-hot
  encoded. `chol == 0` is treated as missing and imputed iteratively; other
  gaps get train-set median/mode fills. Splitting happens *before* imputation,
  so fill statistics never see held-out rows.
- **Splits** - 60/20/20 stratified: tune on train (5-fold `GridSearchCV`,
  scoring ROC-AUC), select on validation (ROC-AUC primary, recall as
  tie-breaker), then refit the winner on train+val and evaluate once on test.
- **Scaling** - `StandardScaler` on the five continuous features only, applied
  inside the pipeline and only for the models that need it (logreg, SVM, KNN).
- **Metrics** - ROC-AUC primary; recall emphasized because a missed diagnosis
  is the costly error in this setting.

## Results

Validation comparison (sorted by ROC-AUC; full table in
`models/comparison.csv` and `notebooks/comparison.ipynb`):

| Model | Val ROC-AUC | Val recall |
|---|---|---|
| **random_forest** | **0.868** | **0.902** |
| svm | 0.865 | 0.873 |
| logreg | 0.865 | 0.853 |
| knn | 0.859 | 0.863 |
| naive_bayes | 0.859 | 0.833 |
| gradient_boosting | 0.856 | 0.873 |
| hist_gradient_boosting | 0.852 | 0.873 |
| decision_tree | 0.807 | 0.794 |

Final test evaluation of the selected random forest (refit on train+val, 184
held-out rows):

| ROC-AUC | Recall | Precision | F1 | Accuracy |
|---|---|---|---|---|
| 0.907 | 0.863 | 0.793 | 0.826 | 0.799 |

Each per-model notebook adds a family-specific diagnostic - e.g. the k-curve
for KNN, staged ROC-AUC for the boosting models, permutation importance for
hist gradient boosting, and probability calibration for Naive Bayes. The test
set is touched nowhere except the final cells of `comparison.ipynb`.

## Data

Creators of the UCI Heart Disease dataset: Andras Janosi (Hungarian Institute
of Cardiology), William Steinbrunn (University Hospital, Zurich), Matthias
Pfisterer (University Hospital, Basel), and Robert Detrano (V.A. Medical
Center, Long Beach & Cleveland Clinic Foundation).
