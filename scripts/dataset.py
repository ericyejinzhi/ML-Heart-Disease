"""Shared dataset helpers for model exploration.

Loads the cleaned train/validation/test splits, exposes the feature/target
split with the target binarized, and builds the continuous-only scaler used
by the models that need it. Imported by train_models.py and the notebook.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGET = "num"

# Only the continuous features are scaled; the binary and one-hot columns are
# left as 0/1 so neither group dominates. Dummy columns (cp_*, restecg_*) are
# discovered at run time from the actual frame, so this list stays stable even
# if the one-hot set changes.
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the (train, val, test) DataFrames from data/."""
    train = pd.read_csv(DATA_DIR / "combined_train.csv")
    val = pd.read_csv(DATA_DIR / "combined_val.csv")
    test = pd.read_csv(DATA_DIR / "combined_test.csv")
    return train, val, test


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a frame into features X and binarized target y (num > 0)."""
    y = (df[TARGET] > 0).astype(int)
    X = df.drop(columns=TARGET)
    return X, y


def build_scaler(columns: list[str] | None = None) -> ColumnTransformer:
    """Build a ColumnTransformer that scales only the continuous features.

    All other columns pass through unchanged. If `columns` is given (the
    feature names of the frame being modeled), only the continuous names
    actually present are scaled; otherwise the full CONTINUOUS list is used.
    """
    to_scale = CONTINUOUS if columns is None else [c for c in CONTINUOUS if c in columns]
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), to_scale)],
        remainder="passthrough",
    )


if __name__ == "__main__":
    # Quick self-check when run directly.
    train, val, test = load_splits()
    X, y = get_xy(train)
    print(f"train {X.shape}  val {val.shape}  test {test.shape}")
    print(f"features: {list(X.columns)}")
    print(f"scaled columns: {[c for c in CONTINUOUS if c in X.columns]}")
    print(f"train disease rate: {y.mean():.3f}")
