"""Render aggregate model comparisons without provider-level information."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cv_scores(metrics: pd.DataFrame, output_file: Path) -> None:
    """Show fold means, sample SD, and individual folds on a shared 0–1 scale."""
    names = ["Logistic Regression", "Random Forest", "Majority baseline"]
    colors = ["#0072B2", "#D55E00", "#71717A"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#202733",
            "text.color": "#202733",
            "xtick.color": "#202733",
            "ytick.color": "#202733",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        for ax, metric, label in zip(
            axes, ["roc_auc", "f1"], ["ROC-AUC", "Positive-class F1"]
        ):
            for i, (name, color) in enumerate(zip(names, colors)):
                scores = metrics.loc[metrics["model"] == name, metric].dropna()
                if scores.empty:
                    ax.text(i, 0.05, "Undefined", ha="center", fontsize=9)
                    continue
                mean, std = scores.mean(), scores.std()
                ax.bar(
                    i,
                    mean,
                    yerr=std if pd.notna(std) else None,
                    color=color,
                    alpha=0.85,
                    width=0.58,
                    capsize=5,
                    error_kw={"elinewidth": 1.5, "capthick": 1.5},
                    zorder=2,
                )
                ax.scatter(
                    i + np.linspace(-0.12, 0.12, len(scores)),
                    scores,
                    s=24,
                    facecolors="white",
                    edgecolors="#202733",
                    linewidths=0.8,
                    zorder=3,
                    clip_on=False,
                )
                ax.text(
                    i,
                    min(mean + (std if pd.notna(std) else 0) + 0.055, 0.975),
                    f"{mean:.3f} ± {std:.3f}" if pd.notna(std) else f"{mean:.3f}",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )
            ax.set_xticks(
                range(3),
                ["Logistic\nregression", "Random\nforest", "Majority\nbaseline"],
            )
            ax.set_ylim(0, 1)
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.set_ylabel(label)
            ax.set_title(label, loc="left", fontweight="bold", pad=15)
            ax.grid(axis="y", color="#E3E7EB", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
        fig.suptitle(
            "Predicting subsequent order-modification activity",
            fontsize=16,
            fontweight="bold",
            x=0.07,
            ha="left",
            y=0.98,
        )
        fig.text(
            0.07,
            0.90,
            "First 5 orders → next 10 orders · provider-grouped cross-validation",
            fontsize=11,
        )
        fig.text(
            0.07,
            0.035,
            "Bars: fold means · error bars: ±1 sample SD · dots: individual folds (not confidence intervals)",
            fontsize=9,
            color="#455060",
        )
        fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.17, wspace=0.23)
        try:
            fig.savefig(output_file, dpi=200, bbox_inches="tight")
        finally:
            plt.close(fig)
