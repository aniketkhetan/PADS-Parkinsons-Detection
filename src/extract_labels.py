"""Build {patient_id: condition} from the PADS patient JSON files."""

import json
import re
from pathlib import Path

ID_KEYS = ("id", "patient_id", "subject_id", "subject")
CONDITION_KEYS = ("condition", "diagnosis", "label", "group", "disorder")

CANONICAL = {
    "parkinson": "Parkinson's",
    "parkinsons": "Parkinson's",
    "parkinson's disease": "Parkinson's",
    "healthy": "Healthy",
    "healthy control": "Healthy",
    "control": "Healthy",
    "atypical parkinsonism": "Atypical Parkinsonism",
    "essential tremor": "Essential Tremor",
    "multiple sclerosis": "Multiple Sclerosis",
    "other movement disorders": "Other Movement Disorders",
    "other movement disorder": "Other Movement Disorders",
}


def _first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def normalise(condition):
    if condition is None:
        return None
    key = str(condition).strip().lower()
    return CANONICAL.get(key, str(condition).strip())


def load_patient_dict(patients_dir):
    """Return {int patient_id: str condition}."""
    patients_dir = Path(patients_dir)
    files = sorted(patients_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no patient JSON files under {patients_dir}")

    patient_dict = {}
    for path in files:
        with open(path) as fh:
            record = json.load(fh)

        pid = _first_key(record, ID_KEYS)
        if pid is None:
            match = re.search(r"(\d+)", path.stem)
            if match is None:
                continue
            pid = match.group(1)

        condition = normalise(_first_key(record, CONDITION_KEYS))
        if condition is None:
            continue

        patient_dict[int(pid)] = condition

    return patient_dict


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser()
    parser.add_argument("patients_dir")
    args = parser.parse_args()

    pd_ = load_patient_dict(args.patients_dir)
    print(f"{len(pd_)} patients")
    for condition, count in Counter(pd_.values()).most_common():
        print(f"  {count:4d}  {condition}")
