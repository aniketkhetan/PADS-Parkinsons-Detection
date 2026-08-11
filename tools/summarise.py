"""Assemble results/*.json into the markdown tables used in the README.

    python tools/summarise.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Internal name -> the label the manuscript used for it.
TASKS = [
    ("six_class", "PD vs All Classes (6-class)"),
    ("subset_pd_hc", "PD vs Healthy (355-subject subset)"),
    ("pd_hc_other", "PD vs DD (3-class)"),
]
METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced Acc."),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
]

# As reported in the 2025 manuscript, Table I.
PUBLISHED = {
    "six_class": dict(accuracy=.9096, balanced_accuracy=.8775, precision=.9109,
                      recall=.9096, f1=.9076),
    "subset_pd_hc": dict(accuracy=.9648, balanced_accuracy=.9289, precision=.9647,
                         recall=.9648, f1=.9641),
    "pd_hc_other": dict(accuracy=.8989, balanced_accuracy=.8694, precision=.9022,
                        recall=.8989, f1=.8981),
}

# Test-set size implied by each published accuracy, since accuracy can only
# take values k/n: 171/188, 137/142, 169/188.
PUBLISHED_N = {"six_class": 188, "subset_pd_hc": 142, "pd_hc_other": 188}


def load(task, protocol):
    path = RESULTS / f"{task}_{protocol}.json"
    if not path.exists():
        return None
    with open(path) as fh:
        return json.load(fh)


def main():
    print("### Reproduction vs. published\n")
    header = "| Task | Protocol | " + " | ".join(n for _, n in METRICS) + " | n test |"
    print(header)
    print("|---|---|" + "---:|" * (len(METRICS) + 1))

    for task, task_name in TASKS:
        pub = PUBLISHED[task]
        print(f"| {task_name} | published (2025) | "
              + " | ".join(f"{pub[k]:.4f}" for k, _ in METRICS)
              + f" | {PUBLISHED_N[task]} |")
        for protocol in ("leaky", "clean"):
            result = load(task, protocol)
            if result is None:
                print(f"| {task_name} | {protocol} | " + " | ".join("--" for _ in METRICS) + " | -- |")
                continue
            m = result["metrics"]
            print(f"| {task_name} | {protocol} | "
                  + " | ".join(f"{m[k]:.4f}" for k, _ in METRICS)
                  + f" | {result['n_test']} |")

    print("\n### Cost of the leak (balanced accuracy)\n")
    print("| Task | leaky | clean | delta |")
    print("|---|---:|---:|---:|")
    for task, task_name in TASKS:
        a, b = load(task, "leaky"), load(task, "clean")
        if not (a and b):
            print(f"| {task_name} | -- | -- | -- |")
            continue
        la = a["metrics"]["balanced_accuracy"]
        lc = b["metrics"]["balanced_accuracy"]
        print(f"| {task_name} | {la:.4f} | {lc:.4f} | {lc - la:+.4f} |")

    missing = [f"{t}_{p}" for t, _ in TASKS for p in ("leaky", "clean")
               if load(t, p) is None]
    if missing:
        print(f"\n_missing runs: {', '.join(missing)}_")


if __name__ == "__main__":
    main()
