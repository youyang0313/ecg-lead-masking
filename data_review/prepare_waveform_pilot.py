"""Select and archive a training-only waveform sanity sample from official S3.

No model code or training. Selection uses metadata before any waveform inspection.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import ast
import csv
import json
import random
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
PILOT = ROOT / "waveform_pilot"
PILOT.mkdir(exist_ok=True)
SOURCE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
with (ROOT/"metadata"/"scp_statements.csv").open(encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))
code_column=next(iter(rows[0]))
mapping={r[code_column]:r["diagnostic_class"] for r in rows if r["diagnostic"]=="1.0"}
audited=json.loads((ROOT/"metadata_audit.json").read_text(encoding="utf-8"))
for name,expected in audited["input_sha256"].items():
    if sha256((ROOT/"metadata"/name).read_bytes()).hexdigest()!=expected:
        raise ValueError("Metadata changed since audit: "+name)

candidates=[]
with (ROOT/"metadata"/"ptbxl_database.csv").open(encoding="utf-8-sig",newline="") as f:
    for row in csv.DictReader(f):
        fold=int(row["strat_fold"])
        if fold not in range(1,9):
            continue
        labels=sorted({mapping[c] for c in ast.literal_eval(row["scp_codes"]) if c in mapping})
        if labels:
            candidates.append(dict(ecg_id=int(row["ecg_id"]),patient_id=row["patient_id"],fold=fold,filename_lr=row["filename_lr"],labels=labels))

random.Random(20261023).shuffle(candidates)
selected=[]
patients=set()
positive=Counter()
def add(row):
    selected.append(row)
    patients.add(row["patient_id"])
    positive.update(row["labels"])

for label in CLASSES:
    for row in candidates:
        if positive[label]>=20:
            break
        if row["patient_id"] not in patients and label in row["labels"]:
            add(row)
for row in candidates:
    if len(selected)>=100:
        break
    if row["patient_id"] not in patients:
        add(row)
if len(selected)!=100 or len(patients)!=100 or min(positive[c] for c in CLASSES)<20:
    raise ValueError("Pilot selection constraints not met")
for row in selected:
    if not re.fullmatch(r"records100/\d{5}/\d{5}_lr",row["filename_lr"]):
        raise ValueError("Unexpected waveform path")
    if int(PurePosixPath(row["filename_lr"]).name.split("_")[0])!=row["ecg_id"]:
        raise ValueError("ECG/path ID mismatch")

selection=dict(date="2026-10-23",dataset="PTB-XL",version="1.0.3",seed=20261023,scope="100 unique training patients; at least 20 positives per target; metadata-only purposive sample, not prevalence representative",waveform_inspection_used_for_selection=False,classes=CLASSES,label_positive_counts=dict(positive),records=selected)
selection_file=PILOT/"selection.json"
if selection_file.exists() and json.loads(selection_file.read_text(encoding="utf-8"))!=selection:
    raise ValueError("Existing pilot selection differs; do not replace it silently")
selection_file.write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")

def retrieve(item):
    row,suffix=item
    rel=row["filename_lr"]+suffix
    path=PILOT/"originals"/rel
    url=SOURCE+rel
    record=dict(ecg_id=row["ecg_id"],relative_path=str(path.relative_to(PILOT)),source_url=url,version="1.0.3",retrieved_on="2026-10-23")
    try:
        if path.exists():
            raw=path.read_bytes()
        else:
            with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=25) as response:
                raw=response.read()
            if suffix==".hea":
                if not raw.decode("ascii").splitlines()[0].startswith(PurePosixPath(rel).stem+" "):
                    raise ValueError("Unexpected header content")
            elif len(raw)!=24000:
                raise ValueError("Unexpected format-16 1000x12 byte length")
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        record.update(status="archived",bytes=len(raw),sha256=sha256(raw).hexdigest())
    except Exception as exc:
        record.update(status="failed",error=type(exc).__name__+": "+str(exc)[:200])
    return record

manifest_path=PILOT/"sources.json"
results={}
jobs=[(row,suffix) for row in selected for suffix in [".hea",".dat"]]
with ThreadPoolExecutor(max_workers=8) as pool:
    pending=[pool.submit(retrieve,job) for job in jobs]
    for future in as_completed(pending):
        record=future.result();results[record["relative_path"]]=record
        if len(results)%20==0 or record["status"]!="archived":
            print(json.dumps({"completed":len(results),"total":len(jobs),"latest_status":record["status"]}),flush=True)
manifest=dict(dataset="PTB-XL",version="1.0.3",official_project_url="https://physionet.org/content/ptb-xl/1.0.3/",scope="Training-only waveform/header pilot; public S3 bucket is listed on official project page",files=[results[k] for k in sorted(results)])
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
failed=[r for r in results.values() if r["status"]!="archived"]
print(json.dumps({"selected_records":len(selected),"selected_patients":len(patients),"label_positives":dict(positive),"archived_files":len(results)-len(failed),"failed_files":len(failed)},ensure_ascii=False),flush=True)
if failed:
    raise SystemExit(1)
