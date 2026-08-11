"""Render the published confusion matrix beside its corrected labelling.

    python tools/make_relabelling_figure.py

Cell counts are transcribed from Fig. 3 of the 2025 manuscript. The two panels
hold identical numbers; only the axis labels differ. The left panel uses the
hand-written order the plotting call passed positionally, the right the
alphabetical order LabelEncoder actually assigned.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT = Path(__file__).resolve().parent.parent / "results" / "confusion_relabelling.png"

CM = np.array([
    [5,  0,  0, 0,  0,   2],
    [0, 10,  0, 0,  0,   0],
    [0,  0, 28, 1,  6,   2],
    [1,  0,  0, 2,  0,   0],
    [1,  2,  1, 0, 17,   2],
    [0,  2,  0, 0,  5, 101],
])

AS_PUBLISHED = ["Parkinson's", "Healthy", "Other Movement Disorders",
                "Essential Tremor", "Atypical Parkinsonism", "Multiple Sclerosis"]

CORRECTED = ["Atypical Parkinsonism", "Essential Tremor", "Healthy",
             "Multiple Sclerosis", "Other Movement Disorders", "Parkinson's"]

# Participants per class in the PADS release.
COHORT = {
    "Atypical Parkinsonism": 15,
    "Essential Tremor": 28,
    "Healthy": 79,
    "Multiple Sclerosis": 11,
    "Other Movement Disorders": 60,
    "Parkinson's": 276,
}


def panel(ax, names, title, highlight):
    sns.heatmap(CM, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=names, yticklabels=names, ax=ax,
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    for label in ax.get_xticklabels():
        label.set_ha("right")

    row = names.index(highlight)
    ax.add_patch(plt.Rectangle((0, row), len(names), 1, fill=False,
                               edgecolor="crimson", lw=2.5, clip_on=False))


def main():
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.6))

    panel(axes[0], AS_PUBLISHED, "As published (Fig. 3)", "Multiple Sclerosis")
    panel(axes[1], CORRECTED, "Correctly labelled (LabelEncoder order)",
          "Parkinson's")

    fig.suptitle(
        "Class names were passed positionally while LabelEncoder assigns them "
        "alphabetically:\nidentical counts, every label wrong",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))

    for ax, caption in zip(axes, [
        'Highlighted row: 108 samples, 101 correct, labelled "Multiple\n'
        'Sclerosis" — a class with 11 participants in all of PADS',
        "Same row, correctly labelled Parkinson's —\n"
        "276 participants, 59% of the cohort",
    ]):
        x = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(x, 0.04, caption, ha="center", va="bottom", fontsize=9.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor="white")
    print(f"wrote {OUT}")

    total = CM.sum()
    print(f"\ntest samples: {total}")
    print(f"accuracy: {np.trace(CM) / total:.4f}")
    recalls = CM.diagonal() / CM.sum(axis=1)
    print(f"balanced accuracy: {recalls.mean():.4f}")
    print("\nrow totals vs cohort share:")
    for i, name in enumerate(CORRECTED):
        expected = COHORT[name] * 2 * 0.2
        print(f"  {name:26} observed {CM[i].sum():3d}   expected {expected:5.1f}")


if __name__ == "__main__":
    main()
