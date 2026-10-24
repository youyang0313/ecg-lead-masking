"""Check the fixed training-only sample with a narrow WFDB format-16 reader.

This is a sanity checker, not a general WFDB implementation or model pipeline.
Unsupported encodings/modifiers are rejected, never guessed.
"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import ast
import csv
import json
import re
import struct
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent
PILOT=ROOT/"waveform_pilot"
PROTOCOL=json.loads((ROOT.parent/"report"/"protocol_v1.json").read_text(encoding="utf-8"))
CONFIG=json.loads((ROOT.parent/"report"/"lead_configurations.json").read_text(encoding="utf-8"))
CANONICAL=CONFIG["canonical_names_for_this_document"]
NAME_MAP={s.upper():s for s in CANONICAL}

def read_record(stem):
    header=stem.with_suffix(".hea").read_text(encoding="ascii")
    lines=[s.strip() for s in header.splitlines() if s.strip() and not s.lstrip().startswith("#")]
    record=lines[0].split()
    if len(record)!=4 or record[0]!=stem.name or "/" in record[0]:
        raise ValueError("Unsupported record line")
    channels,fs,n=int(record[1]),float(record[2]),int(record[3])
    if (channels,fs,n)!=(12,100.0,1000) or len(lines)!=13:
        raise ValueError("Unexpected recording shape/rate")
    specs=[]
    for line in lines[1:]:
        fields=line.split()
        if len(fields)!=9 or fields[0]!=stem.name+".dat" or fields[1]!="16" or int(fields[7])!=0:
            raise ValueError("Only one multiplexed format-16 file with no modifiers is supported")
        gain=re.fullmatch(r"([+-]?[\d.]+(?:[eE][+-]?\d+)?)(?:\((-?\d+)\))?/([^/]+)",fields[2])
        if not gain or gain.group(3)!="mV":
            raise ValueError("Unsupported gain/unit")
        scale=float(gain.group(1)); baseline=int(gain.group(2)) if gain.group(2) is not None else int(fields[4])
        if not np.isfinite(scale) or scale<=0 or fields[8].upper() not in NAME_MAP:
            raise ValueError("Invalid gain or lead name")
        specs.append(dict(native_name=fields[8],name=NAME_MAP[fields[8].upper()],gain=scale,baseline=baseline,initial=int(fields[5]),checksum=int(fields[6])))
    names=[s["name"] for s in specs]
    if len(set(names))!=12 or set(names)!=set(CANONICAL):
        raise ValueError("Expected 12 unique lead names")
    raw=stem.with_suffix(".dat").read_bytes()
    if len(raw)!=channels*n*2:
        raise ValueError("Byte count differs from header")
    digital=np.frombuffer(raw,dtype="<i2").reshape(n,channels)
    # Separate scalar decoder confirms byte order/interleaving on the first frame.
    if tuple(digital[0])!=struct.unpack("<12h",raw[:24]):
        raise ValueError("Scalar and vector decoding differ")
    for i,s in enumerate(specs):
        if int(digital[0,i])!=s["initial"]:
            raise ValueError("Header initial-sample mismatch")
        if int(digital[:,i].astype(np.int64).sum())%65536!=s["checksum"]%65536:
            raise ValueError("WFDB checksum mismatch")
    if np.any(digital==np.iinfo(np.int16).min):
        raise ValueError("Reserved format-16 negative extreme requires missing-data review")
    physical=(digital.astype(np.float64)-np.array([s["baseline"] for s in specs]))/np.array([s["gain"] for s in specs])
    reorder=[names.index(lead) for lead in CANONICAL]
    physical=physical[:,reorder].T.astype(np.float32)
    if not np.isfinite(physical).all():
        raise ValueError("Non-finite physical signal")
    return physical,dict(native_order=[s["native_name"] for s in specs],canonical_order=CANONICAL,fs=fs,samples=n,unit="mV",gains=[s["gain"] for s in specs],baselines=[s["baseline"] for s in specs],initial_samples_checked=12,checksums_checked=12,constant_leads=[CANONICAL[i] for i in range(12) if np.ptp(physical[i])==0])

def apply_configuration(z,available):
    retained=np.array([lead in available for lead in CANONICAL],dtype=bool)
    result=z.copy()
    result[~retained]=0
    return result,retained

def main():
    selection=json.loads((PILOT/"selection.json").read_text(encoding="utf-8"))
    manifest=json.loads((PILOT/"sources.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["status"]!="archived" or sha256((PILOT/entry["relative_path"]).read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("Original-file source/hash verification failed")
    with (ROOT/"metadata"/"scp_statements.csv").open(encoding="utf-8-sig",newline="") as f:
        code_rows=list(csv.DictReader(f))
    key=next(iter(code_rows[0]));mapping={r[key]:r["diagnostic_class"] for r in code_rows if r["diagnostic"]=="1.0"}
    selected_ids={r["ecg_id"] for r in selection["records"]}
    metadata={}
    with (ROOT/"metadata"/"ptbxl_database.csv").open(encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            if int(row["ecg_id"]) in selected_ids:
                if int(row["strat_fold"]) not in range(1,9):
                    raise ValueError("Non-training sample in pilot")
                metadata[int(row["ecg_id"])]=row
    checks=[];signals=[];valid=[]
    for row in selection["records"]:
        check=dict(ecg_id=row["ecg_id"],fold=row["fold"],labels=row["labels"])
        try:
            meta=metadata[row["ecg_id"]]
            labels=sorted({mapping[k] for k in ast.literal_eval(meta["scp_codes"]) if k in mapping})
            if labels!=row["labels"] or meta["filename_lr"]!=row["filename_lr"] or meta["patient_id"]!=row["patient_id"]:
                raise ValueError("Record/label metadata mismatch")
            signal,info=read_record(PILOT/"originals"/row["filename_lr"])
            check.update(status="passed",**info)
            signals.append(signal);valid.append(row)
        except Exception as exc:
            check.update(status="failed",error=type(exc).__name__+": "+str(exc))
        checks.append(check)
    successful=len(signals)
    masking_count=0
    pilot_scaler=None
    if signals:
        stack=np.stack(signals)
        means=stack.astype(np.float64).mean(axis=(0,2))
        stds=stack.astype(np.float64).std(axis=(0,2),ddof=0)
        scales=np.maximum(stds,PROTOCOL["preprocessing"]["std_floor_mV"])
        pilot_scaler=dict(scope="PILOT-ONLY statistics from the selected training sample; not parameters for a full training run",mean_mV=means.tolist(),std_mV=stds.tolist())
        for signal,row in zip(signals,valid):
            z=((signal-means[:,None])/scales[:,None]).astype(np.float32)
            for configuration in CONFIG["configurations"]:
                masked,keep=apply_configuration(z,configuration["available"])
                if masked.shape!=(12,1000) or not np.isfinite(masked).all() or not np.array_equal(masked[keep],z[keep]) or not np.all(masked[~keep]==0):
                    raise ValueError("Input mask invariants failed")
                if keep.sum()!=len(configuration["available"]):
                    raise ValueError("Mask lead count differs")
                masking_count+=1
    orders=Counter(" / ".join(c["native_order"]) for c in checks if c["status"]=="passed")
    result=dict(date="2026-10-23",dataset="PTB-XL 1.0.3",sample_scope=selection["scope"],selected_records=len(selection["records"]),selected_patients=len({r["patient_id"] for r in selection["records"]}),passed_records=successful,failed_records=len(checks)-successful,original_files_hash_verified=len(manifest["files"]),signal_shape_per_record=[12,1000],sampling_hz=100,unit="mV",native_order_counts=dict(orders),canonical_order=CANONICAL,initial_samples_verified=sum(c.get("initial_samples_checked",0) for c in checks),channel_checksums_verified=sum(c.get("checksums_checked",0) for c in checks),record_configuration_checks=masking_count,label_positive_counts=selection["label_positive_counts"],pilot_scaler=pilot_scaler,decoder_scope="Narrow single-file multiplexed WFDB format 16 only, following official format specification; no general WFDB library cross-check",reference_urls=["https://physionet.org/physiotools/wag/header-5.htm","https://physionet.org/physiotools/wag/signal-5.htm"],models_trained=False,validation_or_test_waveforms_read=False,test_diagnostic_codes_parsed=False,limitations=["Purposive sample is not representative of whole-dataset signal quality or class prevalence.","Passing checks does not establish medical correctness of annotations or diagnostic adequacy of 100 Hz.","Final normalization must be fitted to the complete eligible training cohort; pilot parameters must not be reused as final parameters.","Unsupported WFDB encodings are rejected; adopt and cross-check a standard reader before a general data pipeline."],record_checks=checks)
    (PILOT/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["selected_records","passed_records","failed_records","original_files_hash_verified","native_order_counts","initial_samples_verified","channel_checksums_verified","record_configuration_checks","models_trained"]},ensure_ascii=False,indent=2))
    if successful!=len(selection["records"]):
        raise SystemExit(1)

if __name__=="__main__":
    main()
