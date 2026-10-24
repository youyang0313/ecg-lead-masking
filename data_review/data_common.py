"""Shared, model-free PTB-XL development-data utilities."""
from pathlib import Path
from hashlib import sha256
import ast
import csv
import json
import platform
import re
import sys

REVIEW = Path(__file__).resolve().parent
PROJECT = REVIEW.parent
DATA = PROJECT / "data" / "ptbxl_1_0_3"
ORIGINALS = DATA / "originals"
PREPARED = DATA / "prepared"
PRIVATE_PACKAGES = PROJECT / ".runtime" / "data-site-packages"
sys.path.insert(0, str(PRIVATE_PACKAGES))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROTOCOL_PATH = PROJECT / "report" / "protocol_v1.json"
CONFIG_PATH = PROJECT / "report" / "lead_configurations.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CLASSES = PROTOCOL["dataset"]["classes"]
LEADS = CONFIG["canonical_names_for_this_document"]
NAME_MAP = {name.upper(): name for name in LEADS}
SOURCE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def hash_file(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def validate_stem(stem, ecg_id):
    if not re.fullmatch(r"records100/\d{5}/\d{5}_lr", stem):
        raise ValueError("Unexpected waveform path: " + stem)
    if int(Path(stem).name.split("_")[0]) != ecg_id:
        raise ValueError("Waveform path/record ID mismatch")


def input_hashes():
    files = [REVIEW / "metadata" / "ptbxl_database.csv",
             REVIEW / "metadata" / "scp_statements.csv", PROTOCOL_PATH, CONFIG_PATH]
    return {str(p.relative_to(PROJECT)): hash_file(p) for p in files}


def development_metadata():
    """Never parse fold-10 diagnostic annotations. Keep IDs only for split audit."""
    previous = json.loads((REVIEW / "metadata_audit.json").read_text(encoding="utf-8"))
    for name, expected in previous["input_sha256"].items():
        if hash_file(REVIEW / "metadata" / name) != expected:
            raise ValueError("Metadata differs from recorded audit: " + name)
    with (REVIEW / "metadata" / "scp_statements.csv").open(encoding="utf-8-sig", newline="") as f:
        codes = list(csv.DictReader(f))
    key = next(iter(codes[0]))
    mapping = {r[key]: r["diagnostic_class"] for r in codes if r["diagnostic"] == "1.0"}
    records, excluded = [], []
    patients = {"train": set(), "validation": set(), "test": set()}
    raw_counts = {s: 0 for s in patients}
    with (REVIEW / "metadata" / "ptbxl_database.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            fold = int(row["strat_fold"])
            split = "train" if fold in range(1, 9) else "validation" if fold == 9 else "test" if fold == 10 else None
            if split is None:
                raise ValueError("Unexpected fold")
            patients[split].add(row["patient_id"])
            raw_counts[split] += 1
            if split == "test":
                continue  # Do not parse scp_codes or select test waveforms here.
            ecg_id = int(row["ecg_id"])
            validate_stem(row["filename_lr"], ecg_id)
            labels = sorted({mapping[c] for c in ast.literal_eval(row["scp_codes"]) if c in mapping})
            item = dict(ecg_id=ecg_id, patient_id=row["patient_id"], fold=fold,
                        split=split, filename_lr=row["filename_lr"], labels=labels)
            if not labels:
                excluded.append(dict(item, reason="no_diagnostic_superclass"))
            else:
                if not set(labels) <= set(CLASSES):
                    raise ValueError("Unknown target")
                records.append(item)
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if patients[a] & patients[b]:
            raise ValueError("Patient overlap between " + a + " and " + b)
    return dict(dataset="PTB-XL", version="1.0.3", input_sha256=input_hashes(),
                classes=CLASSES, canonical_leads=LEADS,
                test_diagnostic_codes_parsed=False, test_waveforms_selected=False,
                raw_split_counts=raw_counts,
                raw_patient_counts={s: len(p) for s, p in patients.items()},
                records=records, metadata_exclusions=excluded)


def load_plan():
    plan = json.loads((DATA / "development_plan.json").read_text(encoding="utf-8"))
    if plan["input_sha256"] != input_hashes():
        raise ValueError("Inputs changed since development plan; review and regenerate explicitly")
    return plan


def load_download_index():
    result = {}
    path = DATA / "download_manifest.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            result[entry["relative_path"]] = entry
    return result


def environment():
    import numpy, wfdb, scipy, pandas, requests, soundfile, matplotlib
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages={m.__name__: m.__version__ for m in [numpy, wfdb, scipy, pandas, requests, soundfile, matplotlib]},
                wfdb_location=str(Path(wfdb.__file__).resolve()),
                wfdb_source="https://pypi.org/project/wfdb/4.1.2/",
                model_libraries_imported=False)
