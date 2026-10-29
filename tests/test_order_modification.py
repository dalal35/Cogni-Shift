"""Behavior checks with fabricated orders; no clinical extracts required."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from cognishift import evaluation
from cognishift.data import load_orders
from cognishift.features import FEATURES, build_shift_examples
from cognishift.synthetic import make_synthetic_orders


def shift_rows(provider="provider", count=15, day=3, rate=0.0):
    return [
        {
            "order_provider_id": provider,
            "ordertime": f"2026-01-{day:02d} 08:{minute:02d}:00",
            "transaction_type": "Change"
            if 5 <= minute < 5 + round(rate * 10)
            else "New",
            "order_type": "Medication" if minute % 2 else "Lab",
        }
        for minute in range(count)
    ]


def load_rows(rows):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "orders.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return load_orders(path)


def evaluation_examples():
    rows = [
        row
        for provider in range(10)
        for day, rate in enumerate(
            [0.0, 0.1, 0.3, 0.4 if provider < 2 else 0.3, 0.8], start=3
        )
        for row in shift_rows(f"p{provider:02d}", day=day, rate=rate)
    ]
    examples = build_shift_examples(load_rows(rows))
    # Provider-specific distributions expose preprocessing fitted before splitting.
    examples["early_orders_per_hour"] = np.arange(len(examples), dtype=float) ** 2
    examples.loc[::7, "early_orders_per_hour"] = np.nan
    return examples


class OrderTests(unittest.TestCase):
    def test_synthetic_generator_is_reproducible_and_has_no_clinical_fields(self):
        raw = make_synthetic_orders()
        pd.testing.assert_frame_equal(raw, make_synthetic_orders())
        self.assertEqual(
            set(raw.columns),
            {"order_provider_id", "ordertime", "transaction_type", "order_type"},
        )
        self.assertGreaterEqual(raw["order_provider_id"].nunique(), 5)

    def test_shift_boundaries_and_input_normalization(self):
        rows = [
            {
                "order_provider_id": "provider",
                "ordertime": f"2026-01-03 {time}",
                "transaction_type": tx,
                "order_type": None,
            }
            for time, tx in [
                ("19:00:00", "New"),
                ("06:59:00", " change "),
                ("18:59:00", " d/c "),
                ("07:00:00", None),
            ]
        ]
        rows += [
            dict(rows[0], ordertime="invalid"),
            dict(rows[0], order_provider_id=None),
            dict(rows[0], order_provider_id="   "),
        ]
        orders = load_rows(rows)
        self.assertEqual(
            orders["shift_id"].tolist(),
            [
                "provider_2026-01-02_night",
                "provider_2026-01-03_day",
                "provider_2026-01-03_day",
                "provider_2026-01-03_night",
            ],
        )
        self.assertEqual(orders["is_change"].tolist(), [1, 0, 0, 0])
        self.assertEqual(orders["is_dc"].tolist(), [0, 0, 1, 0])
        self.assertEqual(orders["order_type"].tolist(), ["Unknown"] * 4)

    def test_mixed_timestamp_formats_are_not_silently_dropped(self):
        rows = shift_rows(count=2)
        rows[1]["ordertime"] = "2026-01-03T08:01:00"
        self.assertEqual(len(load_rows(rows)), 2)

    def test_numeric_looking_provider_ids_remain_distinct(self):
        rows = shift_rows("001") + shift_rows("1")
        orders = load_rows(rows)
        self.assertEqual(set(orders["order_provider_id"]), {"001", "1"})
        self.assertEqual(len(build_shift_examples(orders)), 2)

    def test_missing_schema_has_useful_error(self):
        with self.assertRaisesRegex(ValueError, "order_provider_id|columns"):
            load_rows([{"unrelated": 1}])

    def test_empty_cleaned_input_is_rejected(self):
        rows = shift_rows(count=1)
        rows[0]["ordertime"] = "not-a-timestamp"
        with self.assertRaisesRegex(ValueError, "valid|usable|empty|No orders"):
            load_rows(rows)

    def test_future_orders_do_not_change_features_or_extend_outcome_window(self):
        rows = shift_rows(count=16)
        baseline = build_shift_examples(load_rows(rows))
        changed = [dict(row) for row in rows]
        for row in changed[5:15]:
            row["transaction_type"] = "Change"
            row["order_type"] = "Other"
        future = build_shift_examples(load_rows(changed))
        pd.testing.assert_frame_equal(baseline[FEATURES], future[FEATURES])
        self.assertEqual(baseline["future_modification_rate"].iloc[0], 0.0)
        self.assertEqual(future["future_modification_rate"].iloc[0], 1.0)
        changed[15]["transaction_type"] = "D/C"
        pd.testing.assert_frame_equal(future, build_shift_examples(load_rows(changed)))

    def test_fewer_than_fifteen_orders_are_excluded(self):
        rows = shift_rows("short", count=14)
        with self.assertRaisesRegex(ValueError, "15 or more orders"):
            build_shift_examples(load_rows(rows))
        rows += shift_rows("eligible")
        self.assertEqual(
            build_shift_examples(load_rows(rows))["provider_id"].tolist(), ["eligible"]
        )

    def test_ties_preserve_fifth_sixth_order_boundary_and_missing_gaps(self):
        rows = shift_rows()
        for row in rows:
            row["ordertime"] = rows[0]["ordertime"]
        rows[4]["transaction_type"] = "D/C"
        rows[5]["transaction_type"] = "Change"
        example = build_shift_examples(load_rows(rows)).iloc[0]
        self.assertEqual(example["early_discontinuations"], 1)
        self.assertEqual(example["early_changes"], 0)
        self.assertEqual(example["future_modification_rate"], 0.1)
        self.assertTrue(np.isnan(example["early_median_gap_minutes"]))
        self.assertEqual(example["early_orders_per_hour"], 20.0)
        self.assertEqual(example["start_weekend"], 1)

    def test_overnight_shift_date_and_actual_start_weekday(self):
        rows = shift_rows()
        for minute, row in enumerate(rows):
            row["ordertime"] = f"2026-01-05 00:{minute:02d}:00"
        example = build_shift_examples(load_rows(rows)).iloc[0]
        self.assertEqual(example["shift_id"], "provider_2026-01-04_night")
        self.assertEqual(example["start_weekend"], 0)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = evaluation_examples()
        cls.fitted = []
        original = evaluation.build_models

        def record_models():
            models = original()
            cls.fitted.append(models)
            return models

        with patch.object(evaluation, "build_models", side_effect=record_models):
            with contextlib.redirect_stdout(io.StringIO()):
                cls.metrics, cls.predictions = evaluation.evaluate(cls.examples)

    def test_each_provider_is_held_out_once_and_threshold_is_training_only(self):
        predictions = self.predictions
        self.assertTrue(
            (predictions.groupby("provider_id")["fold"].nunique() == 1).all()
        )
        self.assertTrue((predictions.groupby(["model", "shift_id"]).size() == 1).all())
        self.assertGreater(self.metrics["train_75pct_cutoff"].nunique(), 1)
        for fold in range(1, 6):
            held_out = set(predictions.loc[predictions["fold"] == fold, "provider_id"])
            train = self.examples.loc[~self.examples["provider_id"].isin(held_out)]
            cutoff = train["future_modification_rate"].quantile(0.75)
            val = predictions.loc[predictions["fold"] == fold]
            self.assertTrue((val["train_75pct_cutoff"] == cutoff).all())
            np.testing.assert_array_equal(
                val["true_high_activity"],
                (val["future_modification_rate"] > cutoff).astype(int),
            )
            self.assertEqual(len(train) + len(held_out) * 5, len(self.examples))

    def test_imputation_and_scaling_are_fitted_only_on_training_providers(self):
        for fold, models in enumerate(self.fitted, start=1):
            held_out = self.predictions.loc[
                self.predictions["fold"] == fold, "provider_id"
            ]
            train = self.examples.loc[
                ~self.examples["provider_id"].isin(held_out), FEATURES
            ]
            for name in ["Logistic Regression", "Random Forest"]:
                imputer = models[name].named_steps["impute"]
                np.testing.assert_allclose(imputer.statistics_, train.median())
            pipeline = models["Logistic Regression"].named_steps
            imputed = pipeline["impute"].transform(train)
            np.testing.assert_allclose(pipeline["scale"].mean_, imputed.mean(axis=0))

    def test_exported_metrics_match_held_out_predictions(self):
        for metric in self.metrics.itertuples():
            val = self.predictions.loc[
                (self.predictions["model"] == metric.model)
                & (self.predictions["fold"] == metric.fold)
            ]
            truth = val["true_high_activity"]
            self.assertAlmostEqual(
                metric.roc_auc, roc_auc_score(truth, val["predicted_probability"])
            )
            self.assertAlmostEqual(
                metric.f1,
                f1_score(truth, val["predicted_high_activity"], zero_division=0),
            )
        baseline = self.metrics.loc[self.metrics["model"] == "Majority baseline"]
        self.assertTrue((baseline["roc_auc"] == 0.5).all())
        self.assertTrue((baseline["f1"] == 0.0).all())

    def test_all_missing_training_feature_is_retained_and_predictions_are_finite(self):
        features = self.examples[FEATURES].copy()
        features["early_median_gap_minutes"] = np.nan
        labels = (self.examples["future_modification_rate"] > 0.3).astype(int)
        for name, model in evaluation.build_models().items():
            model.fit(features, labels)
            self.assertTrue(np.isfinite(model.predict_proba(features)).all(), name)
            if name != "Majority baseline":
                transformed = model.named_steps["impute"].transform(features)
                self.assertEqual(transformed.shape[1], len(FEATURES))

    def test_repeated_evaluation_is_deterministic(self):
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, predictions = evaluation.evaluate(self.examples)
        pd.testing.assert_frame_equal(self.metrics, metrics)
        pd.testing.assert_frame_equal(
            self.predictions, predictions, check_exact=False, atol=1e-14, rtol=1e-12
        )

    def test_too_few_providers_and_single_training_class_have_useful_errors(self):
        with self.assertRaisesRegex(ValueError, "5 distinct providers"):
            evaluation.evaluate(self.examples.iloc[:4])
        examples = self.examples.copy()
        examples["future_modification_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "one training class"):
            evaluation.evaluate(examples)

    def test_single_validation_class_returns_missing_auc(self):
        examples = self.examples.copy()
        _, val = next(
            GroupKFold(5).split(examples[FEATURES], groups=examples["provider_id"])
        )
        examples.loc[val, "future_modification_rate"] = 0.0
        with contextlib.redirect_stdout(io.StringIO()):
            metrics, _ = evaluation.evaluate(examples)
        self.assertTrue(metrics.loc[metrics["fold"] == 1, "roc_auc"].isna().all())


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "cognishift", *map(str, args)],
            text=True,
            capture_output=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
            check=False,
        )

    def test_synthetic_cli_writes_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "results"
            result = self.run_cli("--synthetic", "--output-dir", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("synthetic", result.stdout.lower())
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "grouped_cv_metrics.csv",
                    "grouped_cv_predictions.csv",
                    "grouped_cv_scores.png",
                    "run_summary.json",
                },
            )
            self.assertEqual(len(pd.read_csv(output / "grouped_cv_metrics.csv")), 15)
            self.assertGreater((output / "grouped_cv_scores.png").stat().st_size, 1000)
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertEqual(summary["mode"], "synthetic")
            self.assertIsNone(summary["input_sha256"])

    def test_help_and_bad_inputs_have_useful_errors_without_tracebacks(self):
        self.assertEqual(self.run_cli("--help").returncode, 0)
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "bad.csv"
            bad.write_text("wrong\nvalue\n")
            for args in [
                ["--input", Path(directory) / "missing.csv"],
                ["--input", bad],
                ["--synthetic", "--input", bad],
            ]:
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertIn("error:", result.stderr.lower())
                self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
