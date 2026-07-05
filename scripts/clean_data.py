"""Clean the combined heart disease CSV: report missingness and drop features.

Reads data/combined.csv (produced by parse_data.py), prints the missing
entries per feature, drops the configured columns, and writes the reduced
table to data/combined_reduced.csv, reporting missingness again after the cut.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "combined.csv"
REDUCED_PATH = ROOT / "data" / "combined_reduced.csv"

# Features to drop. Edit this list to choose which columns to remove;
# leave it empty to keep every feature.
DROP_COLUMNS = ["ca", "thal", "slope"]


def report_missing(df: pd.DataFrame) -> None:
    """Print the count and percentage of missing entries per feature."""
    missing = df.isna().sum()
    total = len(df)
    print(f"\nMissing entries per feature (out of {total} rows):")
    print(f"{'feature':<10} {'missing':>8} {'percent':>9}")
    for feature, count in missing.items():
        pct = count / total * 100 if total else 0
        print(f"{feature:<10} {count:>8} {pct:>8.1f}%")
    print(f"{'TOTAL':<10} {missing.sum():>8}")


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


def main() -> None:
    combined = pd.read_csv(IN_PATH)

    print(f"Read {len(combined)} rows from {IN_PATH}")
    report_missing(combined)

    if DROP_COLUMNS:
        reduced = drop_features(combined, DROP_COLUMNS)
        reduced.to_csv(REDUCED_PATH, index=False)
        print(f"\nWrote {len(reduced)} rows x {reduced.shape[1]} columns "
              f"to {REDUCED_PATH}")
        report_missing(reduced)


if __name__ == "__main__":
    main()
