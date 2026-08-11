"""Repeat every task x protocol across seeds and report mean +- std.

    python tools/seed_sweep.py --seeds 42 43 44 45 46

Single-run results here vary by up to 14 points of balanced accuracy with RNG
state, so nothing should be quoted from one run.

Relabelings of the full cohort share one MiniRocket fit per seed. The clean
split is stratified on the six-class labels whatever the task, so the same
subjects are held out across tasks and the tasks stay comparable.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from augment import augment_dataset  # noqa: E402
from features import fit_minirocket, transform  # noqa: E402
from model import ROCKETMLP  # noqa: E402
from run_experiment import filter_task  # noqa: E402

TASKS = ["six_class", "pd_vs_rest", "pd_hc_other"]
LABELS = {
    "six_class": "PD vs All Classes (6-class)",
    "pd_vs_rest": "PD vs Healthy (really PD vs rest)",
    "pd_hc_other": "PD vs DD (really PD/HC/Other)",
    "subset_pd_hc": "PD vs HC (subset, comparable to PADS)",
    "subset_pd_dd": "PD vs DD (subset, comparable to PADS)",
}


def train_eval(F_train, F_test, y_train_s, y_test_s, seed, epochs=20, lr=0.002):
    enc = LabelEncoder().fit(np.concatenate([y_train_s, y_test_s]))
    y_train, y_test = enc.transform(y_train_s), enc.transform(y_test_s)

    torch.manual_seed(seed)
    model = ROCKETMLP(F_train.shape[1], len(enc.classes_))
    loader = DataLoader(
        TensorDataset(torch.tensor(F_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True,
    )
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(F_test)).argmax(1).numpy()
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }


def leaky_split(X, y6, seed):
    """Augment all -> fit on all -> scale on all -> random split."""
    X_aug, y6_aug, _ = augment_dataset(X, y6, seed=seed)
    rocket = fit_minirocket(X_aug, 10000, seed)
    F = transform(rocket, X_aug)
    F = StandardScaler().fit_transform(F).astype(np.float32)
    idx = np.arange(len(F))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    return F, y6_aug, tr, te


def clean_split(X, y6, seed):
    """Split subjects -> augment train only -> fit on train only."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(splitter.split(X, y6))

    X_tr, y6_tr, _ = augment_dataset(X[tr_idx], y6[tr_idx], seed=seed)
    rocket = fit_minirocket(X_tr, 10000, seed)
    F_tr = transform(rocket, X_tr)
    F_te = transform(rocket, X[te_idx])
    scaler = StandardScaler().fit(F_tr)
    return (scaler.transform(F_tr).astype(np.float32),
            scaler.transform(F_te).astype(np.float32),
            y6_tr, y6[te_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--out", default="results/seed_sweep.json")
    args = parser.parse_args()

    blob = np.load(ROOT / "data/processed/movement.npz", allow_pickle=True)
    X, y6 = blob["X"], blob["y"].astype(str)

    # Relabelings share one feature matrix per seed; subsets change X and need
    # their own.
    relabel = [t for t in args.tasks if not t.startswith("subset_")]
    subsets = [t for t in args.tasks if t.startswith("subset_")]

    records = []
    for seed in args.seeds:
        np.random.seed(seed)

        if relabel:
            print(f"[seed {seed}] leaky features (full cohort)...", flush=True)
            F, y6_aug, tr, te = leaky_split(X, y6, seed)
            for task in relabel:
                _, labels = filter_task(X, y6_aug, task)
                m = train_eval(F[tr], F[te], labels[tr], labels[te], seed)
                records.append(dict(seed=seed, task=task, protocol="leaky", **m))
                print(f"  leaky {task:14} bal {m['balanced_accuracy']:.4f}", flush=True)

            print(f"[seed {seed}] clean features (full cohort)...", flush=True)
            F_tr, F_te, y6_tr, y6_te = clean_split(X, y6, seed)
            for task in relabel:
                _, l_tr = filter_task(X, y6_tr, task)
                _, l_te = filter_task(X, y6_te, task)
                m = train_eval(F_tr, F_te, l_tr, l_te, seed)
                records.append(dict(seed=seed, task=task, protocol="clean", **m))
                print(f"  clean {task:14} bal {m['balanced_accuracy']:.4f}", flush=True)

        for task in subsets:
            X_s, y_s = filter_task(X, y6, task)
            print(f"[seed {seed}] {task} ({len(y_s)} subjects)...", flush=True)

            F, y_aug, tr, te = leaky_split(X_s, y_s, seed)
            m = train_eval(F[tr], F[te], y_aug[tr], y_aug[te], seed)
            records.append(dict(seed=seed, task=task, protocol="leaky", **m))
            print(f"  leaky {task:14} bal {m['balanced_accuracy']:.4f}", flush=True)

            F_tr, F_te, l_tr, l_te = clean_split(X_s, y_s, seed)
            m = train_eval(F_tr, F_te, l_tr, l_te, seed)
            records.append(dict(seed=seed, task=task, protocol="clean", **m))
            print(f"  clean {task:14} bal {m['balanced_accuracy']:.4f}", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"seeds": args.seeds, "records": records}, fh, indent=2)

    print(f"\n### Balanced accuracy, mean +- std over {len(args.seeds)} seeds\n")
    print("| Task | leaky | clean | delta |")
    print("|---|---:|---:|---:|")
    for task in args.tasks:
        vals = {}
        for protocol in ("leaky", "clean"):
            v = np.array([r["balanced_accuracy"] for r in records
                          if r["task"] == task and r["protocol"] == protocol])
            vals[protocol] = v
        delta = vals["clean"].mean() - vals["leaky"].mean()
        print(f"| {LABELS[task]} | "
              f"{vals['leaky'].mean():.3f} ± {vals['leaky'].std():.3f} | "
              f"{vals['clean'].mean():.3f} ± {vals['clean'].std():.3f} | "
              f"{delta:+.3f} |")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
