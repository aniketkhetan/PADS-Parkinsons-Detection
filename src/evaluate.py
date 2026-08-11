"""Metrics and figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, average="weighted"):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def report(y_true, y_pred, encoder):
    return classification_report(
        y_true, y_pred, target_names=list(encoder.classes_), zero_division=0
    )


def plot_loss(train_losses, test_losses, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, marker="o", label="Train Loss")
    plt.plot(test_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    _save(out_path)


def plot_accuracy(train_accuracies, test_accuracies, out_path):
    plt.figure(figsize=(7, 4))
    plt.plot(train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(test_accuracies, marker="o", label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    _save(out_path)


def plot_confusion_matrix(y_true, y_pred, encoder, out_path, normalize=False):
    """Class names are taken from the encoder so they cannot desync from the codes."""
    labels = np.arange(len(encoder.classes_))
    names = list(encoder.classes_)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fmt = "d"
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = ".2f"

    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=names, yticklabels=names, square=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix" + (" (row-normalised)" if normalize else ""))
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _save(out_path)


def _save(out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
