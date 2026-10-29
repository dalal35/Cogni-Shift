"""Compare fixed classifiers using provider-held-out cross-validation."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

N_FOLDS = 5
SEED = 42


def build_models() -> dict[str, Pipeline | DummyClassifier]:
    return {
        "Logistic Regression": Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=5,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Majority baseline": DummyClassifier(strategy="prior"),
    }


def evaluate(examples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if examples["provider_id"].nunique() < N_FOLDS:
        raise ValueError(
            f"Need at least {N_FOLDS} distinct providers for grouped cross-validation."
        )

    X = examples[FEATURES]
    rates = examples["future_modification_rate"]
    groups = examples["provider_id"]

    metrics = []
    predictions = []

    cv = GroupKFold(n_splits=N_FOLDS)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, groups=groups),
        start=1,
    ):
        # Calculate cutoff using training outcomes only
        cutoff = float(rates.iloc[train_idx].quantile(0.75))

        y_train = (rates.iloc[train_idx] > cutoff).astype(int)

        y_val = (rates.iloc[val_idx] > cutoff).astype(int)

        if y_train.nunique() < 2:
            raise ValueError(
                f"Fold {fold} has only one training class. "
                "Inspect modification-rate distribution."
            )

        if y_val.nunique() < 2:
            print(
                f"WARNING: fold {fold} has one validation "
                "class; ROC-AUC will be undefined."
            )

        for model_name, model in build_models().items():
            model.fit(
                X.iloc[train_idx],
                y_train,
            )

            probability = model.predict_proba(X.iloc[val_idx])[:, 1]

            predicted = (probability >= 0.5).astype(int)

            auc = roc_auc_score(y_val, probability) if y_val.nunique() == 2 else np.nan

            metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_idx),
                    "validation_rows": len(val_idx),
                    "train_75pct_cutoff": cutoff,
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_val.mean()),
                    "roc_auc": auc,
                    "f1": f1_score(y_val, predicted, zero_division=0),
                    "precision": precision_score(y_val, predicted, zero_division=0),
                    "recall": recall_score(y_val, predicted, zero_division=0),
                }
            )

            for ix, actual, prob, pred in zip(
                val_idx,
                y_val,
                probability,
                predicted,
            ):
                predictions.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "shift_id": examples.iloc[ix]["shift_id"],
                        "provider_id": examples.iloc[ix]["provider_id"],
                        "future_modification_rate": float(rates.iloc[ix]),
                        "train_75pct_cutoff": cutoff,
                        "true_high_activity": int(actual),
                        "predicted_probability": float(prob),
                        "predicted_high_activity": int(pred),
                    }
                )

        print(
            f"Fold {fold}: threshold>{cutoff:.2f}, "
            f"training positives={y_train.mean():.1%}, "
            f"validation positives={y_val.mean():.1%}"
        )

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(predictions),
    )
