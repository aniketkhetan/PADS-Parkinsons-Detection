"""Run one PADS classification experiment.

    python src/run_experiment.py --protocol clean --task six_class

Protocols:

  leaky   Augment the whole dataset, fit MiniRocket on the whole dataset, then
          split at random. Augmenting first leaves most test rows with a
          near-identical synthetic twin in the training set.

  clean   Split by subject first, then augment and fit on the training split
          only.

The two do not evaluate on comparable test sets: leaky tests on ~188 rows of
real and synthetic data, clean on ~94 real recordings.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from augment import augment_dataset
from evaluate import (
    compute_metrics,
    plot_accuracy,
    plot_confusion_matrix,
    plot_loss,
    report,
)
from features import fit_minirocket, transform
from model import ROCKETMLP

HEALTHY = "Healthy"
PD = "Parkinson's"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def filter_task(X, y, task):
    """Map the six PADS classes onto the label set for `task`.

    Task names describe what the code does. The 2025 manuscript's names for
    them are given alongside; see the README for how each was identified.
    """
    if task == "six_class":            # manuscript: "PD vs All Classes"
        return X, y
    if task == "pd_vs_rest":           # PD against every other participant
        return X, np.where(y == PD, PD, "Not PD")
    if task == "pd_hc_other":          # manuscript: "PD vs DD"
        return X, np.where(np.isin(y, [PD, HEALTHY]), y, "Other Disease")
    if task == "subset_pd_hc":         # manuscript: "PD vs Healthy"
        keep = np.isin(y, [PD, HEALTHY])
        return X[keep], y[keep]
    if task == "subset_pd_dd":         # PD against the differential diagnoses
        keep = y != HEALTHY
        return X[keep], np.where(y[keep] == PD, PD, "Differential Diagnosis")
    raise ValueError(f"unknown task {task!r}")


def build_features(X, y, args):
    """Return X_train, X_test, y_train, y_test as MiniRocket features."""
    if args.protocol == "leaky":
        X_aug, y_aug, _ = augment_dataset(X, y, seed=args.seed)
        print(f"augmented (all): {X.shape[0]} -> {X_aug.shape[0]}")

        rocket = fit_minirocket(X_aug, args.num_kernels, args.seed)
        F = transform(rocket, X_aug)
        if args.scale:
            F = StandardScaler().fit_transform(F).astype(np.float32)
        print(f"features: {F.shape}")

        return train_test_split(
            F, y_aug, test_size=args.test_size, random_state=args.seed
        )

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, y))
    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    print(f"subject split: {len(train_idx)} train / {len(test_idx)} test")

    X_train_raw, y_train, _ = augment_dataset(X_train_raw, y_train, seed=args.seed)
    print(f"augmented (train only): {len(train_idx)} -> {X_train_raw.shape[0]}")

    rocket = fit_minirocket(X_train_raw, args.num_kernels, args.seed)
    F_train = transform(rocket, X_train_raw)
    F_test = transform(rocket, X_test_raw)
    if args.scale:
        scaler = StandardScaler().fit(F_train)
        F_train = scaler.transform(F_train).astype(np.float32)
        F_test = scaler.transform(F_test).astype(np.float32)
    print(f"features: train {F_train.shape}, test {F_test.shape}")

    return F_train, F_test, y_train, y_test


def train(F_train, F_test, y_train, y_test, num_classes, args, device):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(F_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(F_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = ROCKETMLP(F_train.shape[1], num_classes, args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_loss, best_state, since_best = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        correct = total = 0
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            running += loss.item()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        history["train_loss"].append(running / len(train_loader))
        history["train_acc"].append(100 * correct / total)

        model.eval()
        correct = total = 0
        running = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                running += criterion(out, yb).item()
                pred = out.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
                preds.extend(pred.cpu().numpy())
                labels.extend(yb.cpu().numpy())

        val_loss = running / len(test_loader)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(100 * correct / total)

        print(f"Epoch [{epoch + 1}/{args.epochs}], "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_acc'][-1]:.2f}%, "
              f"Test Loss: {val_loss:.4f}, "
              f"Test Acc: {history['test_acc'][-1]:.2f}%")

        # Off by default so that `leaky` reproduces the original run.
        if args.patience > 0:
            if val_loss < best_loss - 1e-4:
                best_loss, since_best = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= args.patience:
                    print(f"early stopping at epoch {epoch + 1}")
                    break

    if args.patience > 0 and best_state is not None:
        model.load_state_dict(best_state)
        preds, labels = [], []
        model.eval()
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
                labels.extend(yb.numpy())

    return model, history, np.array(labels), np.array(preds)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="data/processed/movement.npz")
    parser.add_argument("--protocol", choices=["leaky", "clean"], default="clean")
    parser.add_argument("--task", default="six_class",
                        choices=["six_class", "pd_vs_rest", "pd_hc_other",
                                 "subset_pd_hc", "subset_pd_dd"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--no-scale", dest="scale", action="store_false",
                        help="skip feature standardisation; without it the MLP "
                             "collapses to the majority class (every MiniRocket "
                             "PPV feature has mean ~0.5, so a 9,996-dim input is "
                             "dominated by common-mode offset)")
    parser.add_argument("--patience", type=int, default=0,
                        help="early-stopping patience; 0 disables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    blob = np.load(args.data, allow_pickle=True)
    X, y = blob["X"], blob["y"].astype(str)
    print(f"loaded {X.shape[0]} recordings, {X.shape[1]} channels")

    X, y = filter_task(X, y, args.task)
    classes, counts = np.unique(y, return_counts=True)
    print(f"task {args.task}: " + ", ".join(f"{c}={n}" for c, n in zip(classes, counts)))

    started = time.time()
    F_train, F_test, y_train_s, y_test_s = build_features(X, y, args)

    encoder = LabelEncoder().fit(np.concatenate([y_train_s, y_test_s]))
    y_train = encoder.transform(y_train_s)
    y_test = encoder.transform(y_test_s)

    model, history, y_true, y_pred = train(
        F_train, F_test, y_train, y_test, len(encoder.classes_), args, device
    )

    metrics = compute_metrics(y_true, y_pred)
    print("\n" + report(y_true, y_pred, encoder))
    for key, value in metrics.items():
        print(f"{key:>18}: {value:.4f}")

    tag = f"{args.task}_{args.protocol}"
    out = Path(args.out)
    plot_loss(history["train_loss"], history["test_loss"], out / f"{tag}_loss.png")
    plot_accuracy(history["train_acc"], history["test_acc"], out / f"{tag}_accuracy.png")
    plot_confusion_matrix(y_true, y_pred, encoder, out / f"{tag}_confusion.png")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{tag}.json", "w") as fh:
        json.dump(
            {
                "config": vars(args),
                "classes": list(encoder.classes_),
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "metrics": metrics,
                "history": history,
                "elapsed_sec": round(time.time() - started, 1),
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out / f'{tag}.json'}")


if __name__ == "__main__":
    main()
