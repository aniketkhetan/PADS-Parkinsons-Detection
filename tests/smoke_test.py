"""End-to-end smoke test on synthetic data. No PADS download required.

    python tests/smoke_test.py

Runs both protocols at tiny scale and asserts what separates them: under leaky
the test set is drawn from the augmented pool and comes out twice the size it
should be, while under clean it holds real recordings only.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

N_PATIENTS = 60
N_CHANNELS = 12
N_TIMESTEPS = 200
CLASSES = ["Parkinson's", "Healthy", "Essential Tremor"]


def make_synthetic(path, seed=0):
    """Three separable classes: each gets a distinct sinusoidal signature."""
    rng = np.random.default_rng(seed)
    X = np.empty((N_PATIENTS, N_CHANNELS, N_TIMESTEPS), dtype=np.float32)
    y = []
    t = np.linspace(0, 4 * np.pi, N_TIMESTEPS)

    for i in range(N_PATIENTS):
        cls = i % len(CLASSES)
        freq = 1.0 + cls
        signal = np.sin(freq * t) + 0.1 * rng.standard_normal((N_CHANNELS, N_TIMESTEPS))
        X[i] = signal
        y.append(CLASSES[cls])

    np.savez_compressed(
        path, X=X, y=np.array(y),
        patient_ids=np.arange(N_PATIENTS),
        channels=np.array([f"ch{j}" for j in range(N_CHANNELS)]),
    )


def run(protocol, data, out):
    cmd = [
        sys.executable, str(SRC / "run_experiment.py"),
        "--data", str(data), "--out", str(out),
        "--protocol", protocol, "--task", "six_class",
        "--epochs", "3", "--num-kernels", "84", "--batch-size", "8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"{protocol} run failed")
    with open(Path(out) / f"six_class_{protocol}.json") as fh:
        return json.load(fh)


def test_augment_tracks_sources():
    from augment import augment_dataset

    X = np.random.randn(10, 3, 50).astype(np.float32)
    y = np.array(["a"] * 5 + ["b"] * 5)
    X_aug, y_aug, source = augment_dataset(X, y, seed=0)

    assert X_aug.shape[0] == 20, "augmentation must double the dataset"
    assert (y_aug[:10] == y_aug[10:]).all(), "labels must be preserved"
    assert (source[:10] == source[10:]).all(), "each copy must point at its source"
    assert not np.allclose(X_aug[:10], X_aug[10:]), "copies must actually differ"
    print("ok  augment_dataset tracks source indices")


def main():
    test_augment_tracks_sources()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        data = tmp / "movement.npz"
        make_synthetic(data)

        leaky = run("leaky", data, tmp / "leaky")
        clean = run("clean", data, tmp / "clean")

        expected_leaky = int(N_PATIENTS * 2 * 0.2)
        expected_clean = int(N_PATIENTS * 0.2)

        assert leaky["n_test"] == expected_leaky, (
            f"leaky test set should hold {expected_leaky} rows "
            f"(augmented pool), got {leaky['n_test']}"
        )
        assert clean["n_test"] == expected_clean, (
            f"clean test set should hold {expected_clean} real recordings, "
            f"got {clean['n_test']}"
        )
        assert clean["n_train"] == (N_PATIENTS - expected_clean) * 2, (
            "clean protocol should augment the training split only"
        )
        print(f"ok  leaky:  {leaky['n_train']} train / {leaky['n_test']} test "
              f"(test drawn from augmented pool)")
        print(f"ok  clean:  {clean['n_train']} train / {clean['n_test']} test "
              f"(test is real recordings only)")

        for name, result in (("leaky", leaky), ("clean", clean)):
            for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
                value = result["metrics"][key]
                assert 0.0 <= value <= 1.0, f"{name}/{key} out of range: {value}"
            print(f"ok  {name}: accuracy {result['metrics']['accuracy']:.3f}, "
                  f"balanced {result['metrics']['balanced_accuracy']:.3f}")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
