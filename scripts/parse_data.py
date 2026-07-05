"""Parse and combine the UCI heart disease site files into one CSV.

Reads the four processed.*.data files (one patient per line, 14
comma-separated attributes, '?' for missing). reprocessed.hungarian.data
is skipped: it duplicates processed.hungarian.data in a different
encoding. Missing values are written as empty fields in the output.

Output: data/combined.csv with a leading `source` column identifying
the collection site. Cleaning and feature reduction live in clean_data.py.
"""

from pathlib import Path

import pandas as pd

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

SOURCES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "heart+disease"
OUT_PATH = ROOT / "data" / "combined.csv"


def main() -> None:
    frames = []
    for site, filename in SOURCES.items():
        df = pd.read_csv(
            DATA_DIR / filename, header=None, names=COLUMNS, na_values="?"
        )
        df.insert(0, "source", site)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    # A cholesterol of 0 encodes "not measured" (all Switzerland rows,
    # a few VA rows) — treat it as missing rather than a real value.
    combined.loc[combined["chol"] == 0, "chol"] = pd.NA

    # Nullable integer dtypes keep integer-valued columns from being
    # written as floats (63 instead of 63.0) despite the NaNs.
    combined = combined.convert_dtypes()

    combined.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(combined)} rows to {OUT_PATH}")
    print("\nRows per site:")
    print(combined["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
