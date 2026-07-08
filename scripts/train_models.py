"""Train and compare classifier families on the heart disease splits.

For each model: tune hyperparameters with GridSearchCV (stratified 5-fold) on
the training set, evaluate the best estimator on the validation set, then select
the strongest model by validation ROC-AUC and report final metrics on the test
set (refit on train+val). Scaling is applied only to the continuous features and
only for the models that need it (see dataset.build_scaler).

Output:
  models/comparison.csv   - per-model validation metrics

The selected model is refit on train+val and evaluated on test in-process;
no model artifact is persisted.
"""

from pathlib import Path

import pandas as pd 
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataset import ROOT, build_scaler, get_xy, load_splits

MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Each entry: estimator, whether it needs the continuous-only scaler, and a
# hyperparameter grid. Grid keys are prefixed "model__" because the estimator
# is the "model" step of the pipeline. Grids follow PROJECT.md.
MODELS = {
    "logreg": {
        # sklearn 1.8 replaced `penalty` with `l1_ratio` (0 = L2, 1 = L1,
        # in between = elastic net); saga supports the full range.
        "estimator": LogisticRegression(
            solver="saga", max_iter=10000, random_state=RANDOM_STATE
        ),
        "needs_scaling": True,
        "grid": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__l1_ratio": [0.0, 0.5, 1.0],
        },
    },
    "svm": {
        "estimator": SVC(probability=True, random_state=RANDOM_STATE),
        "needs_scaling": True,
        "grid": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    },
    "knn": {
        "estimator": KNeighborsClassifier(),
        "needs_scaling": True,
        "grid": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "needs_scaling": False,
        "grid": {
            "model__max_depth": [3, 5, 7, 10, None],
            "model__min_samples_leaf": [1, 3, 5],
            "model__criterion": ["gini", "entropy"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
        "needs_scaling": False,
        "grid": {
            "model__n_estimators": [100, 300, 500],
            "model__max_depth": [None, 5, 10],
            "model__max_features": ["sqrt", "log2"],
        },
    },
    "gradient_boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "needs_scaling": False,
        "grid": {
            "model__n_estimators": [100, 300],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [2, 3],
        },
    },
    "hist_gradient_boosting": {
        "estimator": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "needs_scaling": False,
        "grid": {
            "model__learning_rate": [0.01, 0.1],
            "model__max_iter": [100, 300],
            "model__max_depth": [None, 3, 5],
        },
    },
    "naive_bayes": {
        "estimator": GaussianNB(),
        "needs_scaling": False,
        "grid": {
            "model__var_smoothing": [1e-11, 1e-9, 1e-7, 1e-5],
        },
    },
}


def build_pipeline(estimator, needs_scaling: bool, columns: list[str]) -> Pipeline:
    """Wrap an estimator in a pipeline, adding the scaler only when needed."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", build_scaler(columns)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def evaluate(model, X, y) -> dict:
    """Return the standard metric set for a fitted model on (X, y)."""
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "recall": recall_score(y, pred),
        "precision": precision_score(y, pred),
        "f1": f1_score(y, pred),
        "accuracy": accuracy_score(y, pred),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    train, val, test = load_splits()
    X_train, y_train = get_xy(train)
    X_val, y_val = get_xy(val)
    X_test, y_test = get_xy(test)
    columns = list(X_train.columns)

    results = []
    tuned = {}
    for name, cfg in MODELS.items():
        pipe = build_pipeline(cfg["estimator"], cfg["needs_scaling"], columns)
        search = GridSearchCV(pipe, cfg["grid"], cv=CV, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)

        best = search.best_estimator_
        tuned[name] = best
        metrics = evaluate(best, X_val, y_val)
        metrics.update(model=name, cv_roc_auc=search.best_score_)
        results.append(metrics)
        print(f"{name:24s} cv_auc={search.best_score_:.3f}  "
              f"val_auc={metrics['roc_auc']:.3f}  val_recall={metrics['recall']:.3f}")

    comparison = pd.DataFrame(results).set_index("model").sort_values(
        "roc_auc", ascending=False
    )
    comparison = comparison[
        ["cv_roc_auc", "roc_auc", "recall", "precision", "f1", "accuracy"]
    ]
    comparison.to_csv(MODELS_DIR / "comparison.csv")
    print("\nValidation comparison (sorted by ROC-AUC):")
    print(comparison.round(3).to_string())

    # Select best by validation ROC-AUC, recall as tie-breaker.
    best_name = comparison.sort_values(
        ["roc_auc", "recall"], ascending=False
    ).index[0]
    print(f"\nSelected best model: {best_name}")

    # Refit the selected pipeline on train + val, evaluate once on test.
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)
    best_model = tuned[best_name].fit(X_trainval, y_trainval)

    test_metrics = evaluate(best_model, X_test, y_test)
    print("\nFinal test metrics (selected model, refit on train+val):")
    for k, v in test_metrics.items():
        print(f"  {k:10s} {v:.3f}")
    print("  confusion matrix [[TN FP][FN TP]]:")
    print(confusion_matrix(y_test, best_model.predict(X_test)))

    print(f"\nWrote {MODELS_DIR / 'comparison.csv'}")


if __name__ == "__main__":
    main()
