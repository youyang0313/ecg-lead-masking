"""Review label feasibility and split integrity without loading ECG waveforms.

Fold 10 contributes identifier/fold counts only. Its scp_codes are not parsed.
This script neither trains a model nor evaluates predictions.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import ast
import csv
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
META = ROOT / "metadata"
with (META / "scp_statements.csv").open(encoding="utf-8-sig", newline="") as handle:
    code_rows = list(csv.DictReader(handle))
code_field = next(iter(code_rows[0]))
codes = {row[code_field]:row for row in code_rows}
mapping = {code:row["diagnostic_class"] for code,row in codes.items() if row["diagnostic"] == "1.0"}
classes = ["NORM", "MI", "STTC", "CD", "HYP"]
if set(mapping.values()) != set(classes):
    raise ValueError("Unexpected superclass mapping")

splits = {name:dict(records=0, patients=set()) for name in ["train", "validation", "test"]}
folds = defaultdict(lambda:dict(records=0, patients=set()))
detail = {name:dict(eligible_records=0,eligible_patients=set(),unmapped_diagnostic_records=0,label_positives=Counter(),label_patterns=Counter(),zero_likelihood_code_occurrences=0,zero_likelihood_diagnostic_occurrences=0) for name in ["train", "validation"]}
record_ids = set()
patient_folds = defaultdict(set)
unknown_codes = Counter()
with (META / "ptbxl_database.csv").open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        record = row["ecg_id"]
        if not record or record in record_ids:
            raise ValueError("Missing or duplicate ECG ID")
        record_ids.add(record)
        patient = row["patient_id"]
        if not patient:
            raise ValueError("Missing patient ID")
        fold = int(row["strat_fold"])
        if not 1 <= fold <= 10:
            raise ValueError("Invalid fold")
        split = "train" if fold <= 8 else "validation" if fold == 9 else "test"
        splits[split]["records"] += 1
        splits[split]["patients"].add(patient)
        folds[fold]["records"] += 1
        folds[fold]["patients"].add(patient)
        patient_folds[patient].add(fold)
        if split == "test":
            continue  # Deliberately do not parse or summarize held-out labels.
        annotation = ast.literal_eval(row["scp_codes"])
        if not isinstance(annotation, dict):
            raise ValueError("Unexpected annotation format")
        unknown_codes.update(code for code in annotation if code not in codes)
        label_set = {mapping[code] for code in annotation if code in mapping}
        d = detail[split]
        d["zero_likelihood_code_occurrences"] += sum(float(v) == 0 for v in annotation.values())
        d["zero_likelihood_diagnostic_occurrences"] += sum(float(v) == 0 for k,v in annotation.items() if k in mapping)
        if not label_set:
            d["unmapped_diagnostic_records"] += 1
            continue
        d["eligible_records"] += 1
        d["eligible_patients"].add(patient)
        d["label_positives"].update(label_set)
        d["label_patterns"].update(["+".join(sorted(label_set))])

overlap = {}
names = list(splits)
for i, left in enumerate(names):
    for right in names[i+1:]:
        overlap[left+"_vs_"+right] = len(splits[left]["patients"] & splits[right]["patients"])
if any(overlap.values()) or any(len(fs)>1 for fs in patient_folds.values()):
    raise ValueError("Patient assigned to multiple folds/splits")
if unknown_codes:
    raise ValueError("Unmapped SCP codes in development metadata")

for name, d in detail.items():
    d["eligible_patients"] = len(d["eligible_patients"])
    d["label_positives"] = {c:d["label_positives"][c] for c in classes}
    d["label_prevalence"] = {c:round(d["label_positives"][c]/d["eligible_records"],6) for c in classes}
    d["multi_label_records"] = sum(n for pattern,n in d["label_patterns"].items() if "+" in pattern)
    d["norm_with_other_superclass_records"] = sum(n for pattern,n in d["label_patterns"].items() if "NORM" in pattern.split("+") and "+" in pattern)
    d["label_patterns"] = dict(sorted(d["label_patterns"].items()))

result = {
    "review_date":"2026-10-23",
    "dataset":"PTB-XL",
    "version":"1.0.3",
    "scope":"Public label metadata feasibility review; no waveform loading or model training",
    "test_policy":"Fold 10 identifier counts and split membership only; its scp_codes never parsed by this script",
    "label_rule":"Official example's code-presence aggregation into diagnostic_class; retain zero-likelihood entries as present statements, not as soft targets",
    "records_total":len(record_ids),
    "patients_total":len(patient_folds),
    "statement_code_count":len(codes),
    "statement_category_counts":{category:sum(r[category]=="1.0" for r in code_rows) for category in ["diagnostic","form","rhythm"]},
    "diagnostic_superclasses":classes,
    "diagnostic_subclass_count":len({r["diagnostic_subclass"] for r in code_rows if r["diagnostic"]=="1.0"}),
    "afib_codebook_entry":{k:codes["AFIB"][k] for k in ["description","diagnostic","form","rhythm","diagnostic_class"]},
    "raw_split_counts":{name:{"records":d["records"],"patients":len(d["patients"])} for name,d in splits.items()},
    "fold_counts":{str(f):{"records":d["records"],"patients":len(d["patients"])} for f,d in sorted(folds.items())},
    "development_label_summary":detail,
    "patient_overlap_between_splits":overlap,
    "unknown_development_codes":dict(unknown_codes),
    "input_sha256":{name:sha256((META/name).read_bytes()).hexdigest() for name in ["ptbxl_database.csv","scp_statements.csv","example_physionet.py"]},
    "limits":["Counts describe available annotations, not independently confirmed clinical disease.","No waveform/header inspection; no verification of actual lead quality or input ordering.","No held-out test label support counts or performance were used to choose tasks.","Category flags can overlap; diagnostic/form/rhythm counts need not sum to total statements."]
}
(ROOT / "metadata_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({key:result[key] for key in ["records_total","patients_total","statement_code_count","statement_category_counts","diagnostic_subclass_count","afib_codebook_entry","raw_split_counts","development_label_summary","patient_overlap_between_splits"]},ensure_ascii=False,indent=2))
