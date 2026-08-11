"""Which task definitions did the 2025 pipeline use?

    python tools/task_definition_probe.py

The manuscript names three settings but never defines them. This runs every
plausible definition under the leaky protocol and scores each against the
published table.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from augment import augment_dataset  # noqa: E402
from features import fit_minirocket, transform  # noqa: E402
from model import ROCKETMLP  # noqa: E402

PD = "Parkinson's"
HEALTHY = "Healthy"
OMD = "Other Movement Disorders"
SEED = 42

# Manuscript Table I: (accuracy, balanced accuracy).
PUBLISHED = {
    "PD vs All Classes": (0.9096, 0.8775),
    "PD vs Healthy": (0.9648, 0.9289),
    "PD vs DD": (0.8989, 0.8694),
}


def six_class(y):
    return y


def three_class(y):
    """The first commented line: PD / Healthy / Other Disease, all 469."""
    return np.where(np.isin(y, [PD, HEALTHY]), y, "Other Disease")


def binary_pd_vs_rest(y):
    """The second commented line: PD against everyone, all 469."""
    return np.where(y == PD, PD, "Not PD")


# (name, subset mask fn or None, label fn)
VARIANTS = [
    ("6-class, all 469", None, six_class),
    ("3-class PD/HC/Other, all 469", None, three_class),
    ("binary PD vs rest, all 469", None, binary_pd_vs_rest),
    ("subset PD+HC only", lambda y: np.isin(y, [PD, HEALTHY]), six_class),
    ("subset PD+DD (drop HC)", lambda y: y != HEALTHY,
     lambda y: np.where(y == PD, PD, "DD")),
    ("subset PD+OMD class only", lambda y: np.isin(y, [PD, OMD]), six_class),
]


def evaluate(F_train, F_test, y_train_s, y_test_s):
    enc = LabelEncoder().fit(np.concatenate([y_train_s, y_test_s]))
    y_train, y_test = enc.transform(y_train_s), enc.transform(y_test_s)

    torch.manual_seed(SEED)
    model = ROCKETMLP(F_train.shape[1], len(enc.classes_))
    loader = DataLoader(
        TensorDataset(torch.tensor(F_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True,
    )
    opt = optim.Adam(model.parameters(), lr=0.002)
    crit = nn.CrossEntropyLoss()
    for _ in range(20):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(F_test)).argmax(1).numpy()
    return accuracy_score(y_test, pred), balanced_accuracy_score(y_test, pred)


def leaky_features(X, y):
    """Original ordering: augment all, fit on all, scale on all, then split."""
    X_aug, y_aug, _ = augment_dataset(X, y, seed=SEED)
    rocket = fit_minirocket(X_aug, 10000, SEED)
    F = transform(rocket, X_aug)
    F = StandardScaler().fit_transform(F).astype(np.float32)
    return F, y_aug


def main():
    blob = np.load(ROOT / "data/processed/movement.npz", allow_pickle=True)
    X, y = blob["X"], blob["y"].astype(str)

    # Full-cohort variants share one feature matrix and one split.
    print("computing features for the full cohort (469)...", flush=True)
    F_full, y_full = leaky_features(X, y)

    rows = []
    for name, mask_fn, label_fn in VARIANTS:
        if mask_fn is None:
            F, y_rows = F_full, y_full
        else:
            keep = mask_fn(y)
            print(f"computing features for '{name}' ({keep.sum()} subjects)...",
                  flush=True)
            F, y_rows = leaky_features(X[keep], y[keep])

        labels = label_fn(y_rows)
        F_tr, F_te, y_tr, y_te = train_test_split(
            F, labels, test_size=0.2, random_state=SEED
        )
        acc, bal = evaluate(F_tr, F_te, y_tr, y_te)
        rows.append((name, len(np.unique(labels)), len(y_te), acc, bal))
        print(f"  -> {name}: acc {acc:.4f}, balanced {bal:.4f}", flush=True)

    print("\n| Variant | classes | n test | Accuracy | Balanced Acc. | closest published | |delta| bal |")
    print("|---|---:|---:|---:|---:|---|---:|")
    for name, n_cls, n_test, acc, bal in rows:
        best, gap = None, 1e9
        for pub_name, (_, pub_bal) in PUBLISHED.items():
            if abs(bal - pub_bal) < gap:
                best, gap = pub_name, abs(bal - pub_bal)
        print(f"| {name} | {n_cls} | {n_test} | {acc:.4f} | {bal:.4f} | {best} | {gap:.4f} |")

    print("\nPublished targets (accuracy / balanced accuracy):")
    for pub_name, (pub_acc, pub_bal) in PUBLISHED.items():
        print(f"  {pub_name:20} {pub_acc:.4f} / {pub_bal:.4f}")


if __name__ == "__main__":
    main()
