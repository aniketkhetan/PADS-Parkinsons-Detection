"""Which experiment produced each row of the manuscript's Table I?

    python tools/row_matching.py --seeds 42 43 44 45 46

Runs every candidate variant under the leaky protocol, records the full
five-metric vector, and reports which (variant, seed) best reproduces each
published row by mean absolute difference. Note that weighted recall equals
accuracy by construction, so the rows carry four independent numbers, not five.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from augment import augment_dataset  # noqa: E402
from features import fit_minirocket, transform  # noqa: E402
from model import ROCKETMLP  # noqa: E402
from run_experiment import filter_task  # noqa: E402

METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]

PUBLISHED = {
    "PD vs All Classes": [0.9096, 0.8775, 0.9109, 0.9096, 0.9076],
    "PD vs Healthy": [0.9648, 0.9289, 0.9647, 0.9648, 0.9641],
    "PD vs DD": [0.8989, 0.8694, 0.9022, 0.8989, 0.8981],
}

VARIANTS = ["six_class", "pd_vs_rest", "pd_hc_other", "subset_pd_hc", "subset_pd_dd"]


def metrics_of(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return [
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(p), float(r), float(f),
    ]


def train_eval(F_tr, F_te, y_tr_s, y_te_s, seed):
    enc = LabelEncoder().fit(np.concatenate([y_tr_s, y_te_s]))
    y_tr, y_te = enc.transform(y_tr_s), enc.transform(y_te_s)

    torch.manual_seed(seed)
    model = ROCKETMLP(F_tr.shape[1], len(enc.classes_))
    loader = DataLoader(
        TensorDataset(torch.tensor(F_tr), torch.tensor(y_tr)),
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
        pred = model(torch.tensor(F_te)).argmax(1).numpy()
    return metrics_of(y_te, pred)


def leaky(X, y, seed):
    X_aug, y_aug, _ = augment_dataset(X, y, seed=seed)
    rocket = fit_minirocket(X_aug, 10000, seed)
    F = transform(rocket, X_aug)
    F = StandardScaler().fit_transform(F).astype(np.float32)
    tr, te = train_test_split(np.arange(len(F)), test_size=0.2, random_state=seed)
    return F, y_aug, tr, te


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--out", default="results/row_matching.json")
    args = parser.parse_args()

    blob = np.load(ROOT / "data/processed/movement.npz", allow_pickle=True)
    X, y6 = blob["X"], blob["y"].astype(str)

    relabel = [v for v in VARIANTS if not v.startswith("subset_")]
    subsets = [v for v in VARIANTS if v.startswith("subset_")]

    records = []
    for seed in args.seeds:
        np.random.seed(seed)
        print(f"[seed {seed}] full cohort...", flush=True)
        F, y_aug, tr, te = leaky(X, y6, seed)
        for v in relabel:
            _, lab = filter_task(X, y_aug, v)
            m = train_eval(F[tr], F[te], lab[tr], lab[te], seed)
            records.append({"variant": v, "seed": seed, "metrics": m})
            print(f"  {v:14} " + " ".join(f"{x:.4f}" for x in m), flush=True)

        for v in subsets:
            X_s, y_s = filter_task(X, y6, v)
            print(f"[seed {seed}] {v} ({len(y_s)})...", flush=True)
            Fs, ys, trs, tes = leaky(X_s, y_s, seed)
            m = train_eval(Fs[trs], Fs[tes], ys[trs], ys[tes], seed)
            records.append({"variant": v, "seed": seed, "metrics": m})
            print(f"  {v:14} " + " ".join(f"{x:.4f}" for x in m), flush=True)

    with open(ROOT / args.out, "w") as fh:
        json.dump({"published": PUBLISHED, "records": records}, fh, indent=2)

    print("\n### Best match for each published row\n")
    print("| Published row | Best-matching variant | seed | mean abs diff | "
          + " | ".join(METRICS) + " |")
    print("|---|---|---:|---:|" + "---:|" * len(METRICS))
    for name, target in PUBLISHED.items():
        best = min(records,
                   key=lambda r: np.mean(np.abs(np.array(r["metrics"]) - target)))
        diff = float(np.mean(np.abs(np.array(best["metrics"]) - target)))
        print(f"| {name} | {best['variant']} | {best['seed']} | {diff:.4f} | "
              + " | ".join(f"{x:.4f}" for x in best["metrics"]) + " |")
        print(f"| _(target)_ | | | | " + " | ".join(f"{x:.4f}" for x in target) + " |")


if __name__ == "__main__":
    main()
