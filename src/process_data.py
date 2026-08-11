"""Decode the PADS preprocessed .bin movement files into a single array.

One recording per participant, 132 channels of 976 samples
(11 tasks x 2 wrists x 2 sensors x 3 axes).

Writes data/processed/movement.npz with X, y, patient_ids and channels.
"""

import argparse
import re
from pathlib import Path

import numpy as np

from extract_labels import load_patient_dict

num_channels = 132  # tasks x wrists x sensors x axes
time_steps = 976    # samples per channel

tasks = ["Relaxed1", "Relaxed2", "RelaxedTask1", "RelaxedTask2", "StretchHold",
         "HoldWeight", "DrinkGlas", "CrossArms", "TouchNose", "Entrainment1",
         "Entrainment2"]
wrists = ["Left", "Right"]
sensors = ["Acceleration", "Rotation"]  # accelerometer and gyroscope
axes = ["X", "Y", "Z"]

channel_names = [f"{task}_{wrist}_{sensor}_{axis}"
                 for task in tasks for wrist in wrists
                 for sensor in sensors for axis in axes]

assert len(channel_names) == num_channels


def _infer_dtype(path, num_channels, time_steps):
    """Derive the sample width from the file size instead of assuming it."""
    size = Path(path).stat().st_size
    expected = num_channels * time_steps
    for dtype in (np.float32, np.float64):
        if size == expected * np.dtype(dtype).itemsize:
            return dtype
    raise ValueError(
        f"{path}: {size} bytes does not match {num_channels}x{time_steps} "
        f"at 4 or 8 bytes per sample"
    )


def load_bin_file(file_path, num_channels, time_steps, dtype=None):
    """Load a .bin file and reshape it into (num_channels, time_steps)."""
    if dtype is None:
        dtype = _infer_dtype(file_path, num_channels, time_steps)
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size != num_channels * time_steps:
        raise ValueError(
            f"{file_path}: got {raw.size} values, expected "
            f"{num_channels * time_steps}"
        )
    return raw.reshape(num_channels, time_steps).astype(np.float32)


def build(root, out_path):
    root = Path(root)
    movement_dir = root / "preprocessed" / "movement"
    patients_dir = root / "patients"

    if not movement_dir.is_dir():
        raise FileNotFoundError(f"expected {movement_dir}")

    patient_dict = load_patient_dict(patients_dir)
    print(f"labels for {len(patient_dict)} patients")

    records, labels, ids = [], [], []
    skipped = []

    for path in sorted(movement_dir.glob("*.bin")):
        match = re.search(r"(\d+)", path.stem)
        if match is None:
            skipped.append((path.name, "no id in filename"))
            continue
        pid = int(match.group(1))

        if pid not in patient_dict:
            skipped.append((path.name, "no label"))
            continue

        records.append(load_bin_file(path, num_channels, time_steps))
        labels.append(patient_dict[pid])
        ids.append(pid)

    if not records:
        raise RuntimeError(f"no usable .bin files found in {movement_dir}")

    X = np.stack(records).astype(np.float32)
    y = np.array(labels)
    patient_ids = np.array(ids, dtype=int)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path, X=X, y=y, patient_ids=patient_ids,
        channels=np.array(channel_names),
    )

    print(f"\nwrote {out_path}  X={X.shape}")
    classes, counts = np.unique(y, return_counts=True)
    for cls, count in sorted(zip(classes, counts), key=lambda t: -t[1]):
        print(f"  {count:4d}  {cls}")
    if skipped:
        print(f"\nskipped {len(skipped)} file(s); first few:")
        for name, why in skipped[:5]:
            print(f"  {name}: {why}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="data/pads-parkinsons-disease-smartwatch-dataset-1.0.0",
        help="root of the extracted PADS release",
    )
    parser.add_argument("--out", default="data/processed/movement.npz")
    args = parser.parse_args()
    build(args.root, args.out)
