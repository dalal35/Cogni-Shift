"""One command for the real evaluation and a data-free synthetic demonstration."""

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .data import load_orders, prepare_orders
from .evaluation import N_FOLDS, SEED, evaluate
from .features import FIRST_N, NEXT_N, build_shift_examples
from .reporting import plot_cv_scores
from .synthetic import make_synthetic_orders


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Predict next-ten order-modification activity from first-five orders "
            "using five-fold provider-grouped evaluation."
        )
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input", type=Path, default=Path("data/poe.csv"), help="Authorized order CSV"
    )
    source.add_argument(
        "--synthetic", action="store_true", help="Run fictional demonstration data"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    try:
        if args.synthetic:
            print("SYNTHETIC DEMONSTRATION — metrics are not research evidence.")
            orders = prepare_orders(make_synthetic_orders())
            input_sha256 = None
        else:
            if not args.input.is_file():
                parser.error(
                    "Input CSV not found. Supply --input PATH to authorized data "
                    "or use --synthetic."
                )
            input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
            orders = load_orders(args.input)
        examples = build_shift_examples(orders)
        print(
            f"Loaded {len(orders):,} orders; {len(examples):,} eligible constructed "
            f"windows from {examples['provider_id'].nunique():,} providers."
        )
        metrics, predictions = evaluate(examples)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(args.output_dir / "grouped_cv_metrics.csv", index=False)
        predictions.to_csv(args.output_dir / "grouped_cv_predictions.csv", index=False)
        plot_cv_scores(metrics, args.output_dir / "grouped_cv_scores.png")
        summary = {
            "mode": "synthetic" if args.synthetic else "local",
            "input_sha256": input_sha256,
            "cleaned_orders": len(orders),
            "eligible_shifts": len(examples),
            "providers": int(examples["provider_id"].nunique()),
            "seed": SEED,
            "n_folds": N_FOLDS,
            "first_n": FIRST_N,
            "next_n": NEXT_N,
            "versions": {
                "python": platform.python_version(),
                **{
                    name: version(name)
                    for name in (
                        "numpy",
                        "pandas",
                        "scikit-learn",
                        "scipy",
                        "matplotlib",
                    )
                },
            },
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    except pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError:
        parser.error(
            "Input must be a nonempty, valid UTF-8 CSV with a consistent schema."
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print("\nProvider-grouped CV: mean ± sample SD across five folds")
    for name, group in metrics.groupby("model", sort=False):
        print(
            f"{name:20s} | ROC-AUC {group['roc_auc'].mean():.3f} "
            f"± {group['roc_auc'].std():.3f} | F1 {group['f1'].mean():.3f} "
            f"± {group['f1'].std():.3f}"
        )
    print(
        f"\nSaved metrics, local predictions, figure, and run summary to {args.output_dir}"
    )
    print(
        "Outcome: order-modification activity; not a fatigue or clinical-risk diagnosis."
    )
    return 0
