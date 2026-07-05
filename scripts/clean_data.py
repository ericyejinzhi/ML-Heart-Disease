"""Clean the combined heart disease CSV into model-ready train/test sets.

Reads data/combined.csv (produced by parse_data.py) and:
  1. reports missingness per feature;
  2. drops the configured columns and writes data/combined_reduced.csv;
  3. splits into stratified train/test sets *before* any imputation;
  4. fits all imputation statistics on the training set only and applies
     them to both sets, so the test set never informs the fill values;
  5. one-hot encodes the nominal categoricals so the iterative imputer
     (and later models) never treat them as ordered numbers;
  6. writes data/combined_train.csv and data/combined_test.csv.

Imputation strategy:
  - `chol`  : iterative (regressed on the other features)
  - median  : the other continuous features (trestbps, thalach, oldpeak)
  - mode    : the categorical features (fbs, restecg, exang, ...)
"""

from pathlib import Path

import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "combined.csv"
REDUCED_PATH = ROOT / "data" / "combined_reduced.csv"
TRAIN_PATH = ROOT / "data" / "combined_train.csv"
VAL_PATH = ROOT / "data" / "combined_val.csv"
TEST_PATH = ROOT / "data" / "combined_test.csv"

# Features to drop. Edit this list to choose which columns to remove;
# leave it empty to keep every feature.
DROP_COLUMNS = ["ca", "thal", "slope"]

# Column typing that drives cleaning. `chol` is imputed iteratively; the
# other CONTINUOUS features get the median; BINARY and NOMINAL features get
# the mode; NOMINAL features are additionally one-hot encoded.
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY = ["sex", "fbs", "exang"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
ITERATIVE_COLUMN = "chol"
TARGET = "num"

# 60 / 20 / 20 train / validation / test split.
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42


def report_missing(df: pd.DataFrame) -> None:
    """Print the count and percentage of missing entries per feature."""
    missing = df.isna().sum()
    total = len(df)
    print(f"\nMissing entries per feature (out of {total} rows):")
    print(f"{'feature':<12} {'missing':>8} {'percent':>9}")
    for feature, count in missing.items():
        pct = count / total * 100 if total else 0
        print(f"{feature:<12} {count:>8} {pct:>8.1f}%")
    print(f"{'TOTAL':<12} {missing.sum():>8}")


def drop_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with the named columns removed.

    Columns not present in df are ignored (with a warning) so the list
    can be edited freely without breaking the run.
    """
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"\nWarning: columns not found, skipped: {', '.join(missing)}")
    if present:
        print(f"\nDropping features: {', '.join(present)}")
    return df.drop(columns=present)


def clean(train: pd.DataFrame,
          holdouts: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Impute and one-hot encode, fitting every statistic on train only.

    Returns the cleaned train frame and the cleaned holdout frames (e.g.
    validation and test) with no missing values and nominal features
    expanded into one-hot columns. Median, mode, and the iterative
    imputer are all fit on the training rows so the holdouts stay
    genuine hold-outs.
    """
    train = train.copy()
    holdouts = [df.copy() for df in holdouts]
    everything = [train, *holdouts]

    # 1. Median imputation for continuous features (except chol).
    median_cols = [c for c in CONTINUOUS if c != ITERATIVE_COLUMN and c in train]
    for col in median_cols:
        fill = train[col].median()
        for df in everything:
            df[col] = df[col].fillna(fill)

    # 2. Mode imputation for categorical features, then cast to int so the
    #    one-hot columns come out as clean category names (cp_1, not cp_1.0).
    categorical = [c for c in BINARY + NOMINAL if c in train]
    for col in categorical:
        fill = train[col].mode().iloc[0]
        for df in everything:
            df[col] = df[col].fillna(fill).astype(int)

    # 3. One-hot encode the nominal features. Encoding is row-wise and does
    #    not leak, but we still fit the category set on train and reindex the
    #    holdouts to match, so every set ends up with identical columns.
    nominal = [c for c in NOMINAL if c in train]
    if nominal:
        print(f"\nOne-hot encoding: {', '.join(nominal)}")
        train = pd.get_dummies(train, columns=nominal, prefix=nominal, dtype=int)
        holdouts = [
            pd.get_dummies(df, columns=nominal, prefix=nominal, dtype=int)
            .reindex(columns=train.columns, fill_value=0)
            for df in holdouts
        ]
        everything = [train, *holdouts]

    # 4. Iterative imputation for chol, fit on train and applied to all sets.
    #    The target is excluded from the predictor matrix.
    if ITERATIVE_COLUMN in train and train[ITERATIVE_COLUMN].isna().any():
        features = [c for c in train.columns if c != TARGET]
        print(f"\nIteratively imputing '{ITERATIVE_COLUMN}' from "
              f"{len(features) - 1} features (fit on train)")
        imputer = IterativeImputer(random_state=RANDOM_STATE)
        imputer.fit(train[features].astype(float))
        idx = features.index(ITERATIVE_COLUMN)
        for df in everything:
            filled = imputer.transform(df[features].astype(float))
            # Cholesterol is measured in whole mg/dl; round back to integers.
            df[ITERATIVE_COLUMN] = filled[:, idx].round().astype(int)

    return train, holdouts


def main() -> None:
    combined = pd.read_csv(IN_PATH)

    print(f"Read {len(combined)} rows from {IN_PATH}")

    # The collection site is not used as a feature; drop it from cleaning.
    combined = combined.drop(columns="source")

    report_missing(combined)

    reduced = combined
    if DROP_COLUMNS:
        reduced = drop_features(combined, DROP_COLUMNS)
        reduced.to_csv(REDUCED_PATH, index=False)
        print(f"\nWrote {len(reduced)} rows x {reduced.shape[1]} columns "
              f"to {REDUCED_PATH}")
        report_missing(reduced)

    # Split before imputing so fill values are learned from train only.
    # Stratify on the binarized target to keep disease prevalence balanced.
    # Two steps: carve off the test set, then split the rest into
    # train/validation so the final ratio is 60 / 20 / 20.
    strat = (reduced[TARGET] > 0).astype(int)
    trainval, test = train_test_split(
        reduced, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat
    )
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        trainval, test_size=val_fraction, random_state=RANDOM_STATE,
        stratify=(trainval[TARGET] > 0).astype(int),
    )
    print(f"\nSplit into {len(train)} train / {len(val)} val / {len(test)} "
          f"test rows (stratified)")

    train, (val, test) = clean(train, [val, test])

    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)
    print(f"\nWrote {len(train)} rows x {train.shape[1]} columns to {TRAIN_PATH}")
    print(f"Wrote {len(val)} rows x {val.shape[1]} columns to {VAL_PATH}")
    print(f"Wrote {len(test)} rows x {test.shape[1]} columns to {TEST_PATH}")
    report_missing(train)
    report_missing(val)
    report_missing(test)


if __name__ == "__main__":
    main()
