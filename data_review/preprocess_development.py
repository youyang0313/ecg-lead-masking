"""Standard-WFDB cross-check and train-fitted development preprocessing.

No model creation/training. Test annotations and test waveforms are not used.
Physical float32 conversion matches the earlier pilot; statistics use float64.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
import time

from data_common import (CLASSES, CONFIG, DATA, LEADS, NAME_MAP, ORIGINALS,
                         PREPARED, PROTOCOL, REVIEW, environment, hash_file,
                         load_download_index, load_plan, write_json)
import numpy as np
import wfdb


def now():
    return datetime.now(timezone.utc).isoformat()


def read_standard(stem):
    record = wfdb.rdrecord(str(stem), physical=True, return_res=64)
    if (record.n_sig, record.fs, record.sig_len) != (12, 100, 1000):
        raise ValueError("Unexpected lead count/sampling/duration")
    if record.units != ["mV"] * 12:
        raise ValueError("Expected physical mV units")
    try:
        names = [NAME_MAP[n.upper()] for n in record.sig_name]
    except (KeyError, AttributeError) as exc:
        raise ValueError("Unexpected lead name") from exc
    if len(set(names)) != 12 or set(names) != set(LEADS):
        raise ValueError("Missing or duplicate expected lead names")
    signal = record.p_signal[:, [names.index(n) for n in LEADS]].T.astype(np.float32)
    if signal.shape != (12, 1000) or not np.isfinite(signal).all():
        raise ValueError("Unexpected shape or non-finite physical waveform")
    return signal, dict(native_leads=record.sig_name,
                        constant_leads=[LEADS[c] for c in range(12) if np.ptp(signal[c]) == 0])


def crosscheck_pilot():
    from check_waveform_pilot import read_record
    pilot = REVIEW / "waveform_pilot"
    manifest = json.loads((pilot / "sources.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if hash_file(pilot / entry["relative_path"]) != entry["sha256"]:
            raise ValueError("Pilot raw-file hash mismatch")
    rows = json.loads((pilot / "selection.json").read_text(encoding="utf-8"))["records"]
    checks = []
    for row in rows:
        stem = pilot / "originals" / row["filename_lr"]
        narrow, _ = read_record(stem)
        standard, info = read_standard(stem)
        difference = float(np.max(np.abs(narrow.astype(np.float64)-standard)))
        if not np.array_equal(narrow, standard):
            raise ValueError("WFDB and pilot reader disagree: " + str(row["ecg_id"]))
        checks.append(dict(ecg_id=row["ecg_id"], exact_float32_match=True,
                           max_absolute_difference_mV=difference, **info))
    result = dict(completed_on=now(), records_checked=len(checks),
                  exact_matches=len(checks), values_compared=len(checks)*12000,
                  comparison="physical mV, canonical lead order, float32 inputs",
                  environment=environment(), checks=checks,
                  standard_reader_documentation="https://wfdb.readthedocs.io/en/v4.1.2/io.html",
                  test_waveforms_read=False, models_trained=False)
    write_json(pilot / "wfdb_crosscheck.json", result)
    print(json.dumps({k: result[k] for k in ["records_checked", "exact_matches", "values_compared", "models_trained"]}, indent=2))


class RunningMoments:
    """Mergeable population moments, one independent channel per row."""
    def __init__(self):
        self.count = 0
        self.mean = np.zeros(12, dtype=np.float64)
        self.m2 = np.zeros(12, dtype=np.float64)

    def update(self, signal):
        x = signal.astype(np.float64)
        n = x.shape[1]
        mean = x.mean(axis=1)
        m2 = ((x-mean[:, None])**2).sum(axis=1)
        delta = mean-self.mean
        total = self.count+n
        self.m2 += m2+delta**2*self.count*n/total
        self.mean += delta*n/total
        self.count = total

    def std(self):
        if self.count == 0:
            raise ValueError("No training samples for normalization")
        return np.sqrt(self.m2/self.count)


def normalize(signal, mean, scale):
    return ((signal.astype(np.float64)-mean[:, None])/scale[:, None]).astype(np.float32)


def check_masks(z):
    for config in CONFIG["configurations"]:
        keep = np.array([n in config["available"] for n in LEADS])
        masked = z.copy()
        masked[~keep] = 0
        if not np.array_equal(masked[keep], z[keep]) or np.any(masked[~keep] != 0):
            raise ValueError("Mask altered retained signals or failed to zero absent channels")
        if not np.isfinite(masked).all():
            raise ValueError("Non-finite masked input")
    return len(CONFIG["configurations"])


def read_checked(row):
    try:
        signal, info = read_standard(ORIGINALS/row["filename_lr"])
        return row, signal, info, None
    except Exception as exc:
        return row, None, None, type(exc).__name__+": "+str(exc)[:300]


def iter_waveforms(rows):
    # Bound in-flight arrays and retain the exact cohort order for statistics
    # and output indexing, independently of filesystem completion order.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(rows), 128):
            yield from pool.map(read_checked, rows[start:start+128])


def verify_downloads(plan):
    index = load_download_index()
    print(json.dumps(dict(stage="verify_original_hashes", total_files=2*len(plan["records"]))), flush=True)
    relative_paths = []
    for row in plan["records"]:
        for suffix in [".hea", ".dat"]:
            relative_paths.append(row["filename_lr"]+suffix)
    def check(relative):
        entry = index.get(relative)
        path = ORIGINALS / relative
        if not entry or entry["status"] != "archived" or not path.exists():
            raise ValueError("Download incomplete: " + relative)
        if hash_file(path) != entry["sha256"]:
            raise ValueError("Original hash mismatch; restore before processing: " + relative)
        return relative
    with ThreadPoolExecutor(max_workers=24) as pool:
        for total, _ in enumerate(pool.map(check, relative_paths), 1):
            if total % 2000 == 0:
                progress = dict(stage="verify_original_hashes", checked=total,
                                total_files=2*len(plan["records"]))
                write_json(DATA/"preprocessing_progress.json", progress)
                print(json.dumps(progress), flush=True)
    return total


def process():
    print(json.dumps(dict(stage="load_locked_development_plan")), flush=True)
    plan = load_plan()
    started = time.monotonic()
    # A stale ready marker must never describe a partial rerun.
    marker = PREPARED / "READY.json"
    if marker.exists():
        raise ValueError("Prepared data already exist; verify before choosing an explicit versioned rebuild")
    PREPARED.mkdir(parents=True, exist_ok=True)
    verified = verify_downloads(plan)
    print(json.dumps(dict(raw_files_hash_verified=verified)), flush=True)
    moments = RunningMoments()
    eligible = {"train": [], "validation": []}
    failures, checks = [], []
    constants = {s: Counter() for s in eligible}
    orders = Counter()
    for i, (row, signal, info, error) in enumerate(iter_waveforms(plan["records"]), 1):
        if error:
            # Capture eligibility failures without changing inclusion by model/configuration.
            failures.append(dict(row, reason=error))
        else:
            eligible[row["split"]].append(row)
            constants[row["split"]].update(info["constant_leads"])
            orders[" / ".join(info["native_leads"])] += 1
            checks.append(dict(ecg_id=row["ecg_id"], split=row["split"], constant_leads=info["constant_leads"]))
            if row["split"] == "train":
                moments.update(signal)
        if i % 1000 == 0 or i == len(plan["records"]):
            print(json.dumps(dict(stage="eligibility_and_training_statistics", checked=i,
                                  total=len(plan["records"]), failures=len(failures))), flush=True)
    if any(not rows for rows in eligible.values()):
        raise ValueError("An eligible development split is empty")
    std = moments.std()
    floor = PROTOCOL["preprocessing"]["std_floor_mV"]
    scale = np.maximum(std, floor)
    training_ids = [r["ecg_id"] for r in eligible["train"]]
    train_ids_path = PREPARED / "scaler_training_ecg_ids.json"
    write_json(train_ids_path, training_ids)
    scaler = dict(completed_on=now(), scope="eligible complete TRAINING signals only",
                  canonical_leads=LEADS, mean_mV=moments.mean.tolist(),
                  population_std_mV=std.tolist(), applied_scale_mV=scale.tolist(),
                  std_floor_mV=floor, training_records=len(training_ids),
                  samples_per_lead=moments.count,
                  training_ids_sha256=hash_file(train_ids_path),
                  input_sha256=plan["input_sha256"],
                  validation_used_for_fitting=False, test_used_for_fitting=False)
    write_json(PREPARED / "scaler.json", scaler)
    outputs, normalized_audits = [], {}
    masks = 0
    for split, rows in eligible.items():
        signal_path = PREPARED / (split+".signals.npy")
        temp = PREPARED / (split+".signals.partial.npy")
        out = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                        shape=(len(rows), 12, 1000))
        labels = np.zeros((len(rows), len(CLASSES)), dtype=np.uint8)
        normalized_moments = RunningMoments()
        max_abs = 0.0
        for i, (row, signal, _, error) in enumerate(iter_waveforms(rows)):
            if error:
                raise ValueError("Previously eligible waveform failed on second read: "+error)
            z = normalize(signal, moments.mean, scale)
            if not np.isfinite(z).all():
                raise ValueError("Non-finite normalized signal")
            out[i] = z
            labels[i] = [c in row["labels"] for c in CLASSES]
            normalized_moments.update(z)
            max_abs = max(max_abs, float(np.max(np.abs(z))))
            masks += check_masks(z)
            if (i+1) % 1000 == 0 or i+1 == len(rows):
                out.flush()
                print(json.dumps(dict(stage="write_normalized", split=split, written=i+1,
                                      total=len(rows))), flush=True)
        out.flush()
        del out
        temp.replace(signal_path)
        np.save(PREPARED / (split+".labels.npy"), labels, allow_pickle=False)
        write_json(PREPARED / (split+".records.json"), rows)
        normalized_audits[split] = dict(records=len(rows), patients=len({r["patient_id"] for r in rows}),
                                       label_positive_counts=dict(zip(CLASSES, labels.sum(axis=0).astype(int).tolist())),
                                       mean=normalized_moments.mean.tolist(),
                                       population_std=normalized_moments.std().tolist(),
                                       max_absolute_standardized_value=max_abs)
        for path in [signal_path, PREPARED/(split+".labels.npy"), PREPARED/(split+".records.json")]:
            outputs.append(dict(file=path.name, bytes=path.stat().st_size, sha256=hash_file(path)))
    training_summary = normalized_audits["train"]
    if max(abs(v) for v in training_summary["mean"]) > 1e-5:
        raise ValueError("Normalized training means differ unexpectedly from zero")
    expected_std = std/scale
    if not np.allclose(training_summary["population_std"], expected_std, rtol=1e-5, atol=1e-5):
        raise ValueError("Normalized training standard deviations differ from expected values")
    write_json(PREPARED/"waveform_exclusions.json", failures)
    write_json(PREPARED/"record_checks.json", checks)
    for name in ["scaler.json", "scaler_training_ecg_ids.json", "waveform_exclusions.json", "record_checks.json"]:
        path = PREPARED/name
        outputs.append(dict(file=name, bytes=path.stat().st_size, sha256=hash_file(path)))
    audit = dict(completed_on=now(), status="preprocessing_complete_no_models",
                 dataset="PTB-XL 1.0.3", input_sha256=plan["input_sha256"],
                 raw_files_hash_verified=verified,
                 metadata_exclusions={s: sum(r["split"]==s for r in plan["metadata_exclusions"]) for s in eligible},
                 waveform_exclusions=len(failures), splits=normalized_audits,
                 constant_lead_record_counts={s: dict(c) for s, c in constants.items()},
                 constant_lead_policy="recorded, not excluded solely for this flag",
                 native_lead_orders=dict(orders), mask_invariant_checks=masks,
                 no_extra_filtering=True, normalization_fitted_on="train_only",
                 test_waveforms_read=False, test_diagnostic_codes_parsed=False,
                 models_built=False, models_trained=False, environment=environment(),
                 pipeline_source_sha256={name: hash_file(REVIEW/name) for name in
                                         ["data_common.py", "prepare_development.py", "preprocess_development.py", "test_preprocessing.py"]},
                 outputs=outputs, elapsed_seconds=round(time.monotonic()-started, 1))
    write_json(REVIEW/"preprocessing_audit.json", audit)
    write_json(marker, dict(status="complete", audit_sha256=hash_file(REVIEW/"preprocessing_audit.json"), outputs=outputs))
    print(json.dumps(dict(status=audit["status"], counts={s:v["records"] for s,v in normalized_audits.items()},
                          waveform_exclusions=len(failures), mask_checks=masks), indent=2))


def verify_prepared():
    plan = load_plan()
    audit = json.loads((REVIEW/"preprocessing_audit.json").read_text(encoding="utf-8"))
    ready = json.loads((PREPARED/"READY.json").read_text(encoding="utf-8"))
    if ready["audit_sha256"] != hash_file(REVIEW/"preprocessing_audit.json"):
        raise ValueError("Audit changed after ready marker")
    if audit["input_sha256"] != plan["input_sha256"]:
        raise ValueError("Audit/plan input mismatch")
    for name, expected in audit["pipeline_source_sha256"].items():
        if hash_file(REVIEW/name) != expected:
            raise ValueError("Pipeline source changed after preprocessing: " + name)
    for entry in ready["outputs"]:
        if hash_file(PREPARED/entry["file"]) != entry["sha256"]:
            raise ValueError("Prepared output hash mismatch: " + entry["file"])
    scaler = json.loads((PREPARED/"scaler.json").read_text(encoding="utf-8"))
    mean = np.array(scaler["mean_mV"])
    scale = np.array(scaler["applied_scale_mV"])
    patients = {}
    planned = {r["ecg_id"]: r for r in plan["records"]}
    excluded = json.loads((PREPARED/"waveform_exclusions.json").read_text(encoding="utf-8"))
    excluded_ids = {r["ecg_id"] for r in excluded}
    saved_ids = set()
    for split in ["train", "validation"]:
        rows = json.loads((PREPARED/(split+".records.json")).read_text(encoding="utf-8"))
        x = np.load(PREPARED/(split+".signals.npy"), mmap_mode="r", allow_pickle=False)
        y = np.load(PREPARED/(split+".labels.npy"), allow_pickle=False)
        if x.shape != (len(rows), 12, 1000) or x.dtype != np.float32 or y.shape != (len(rows), 5):
            raise ValueError("Prepared shapes/dtypes differ")
        if any(row != planned[row["ecg_id"]] or row["split"] != split for row in rows):
            raise ValueError("Saved record differs from the locked development plan")
        expected_labels = np.array([[c in row["labels"] for c in CLASSES] for row in rows], dtype=np.uint8)
        if not np.array_equal(y, expected_labels):
            raise ValueError("Full saved label array differs from planned labels")
        if len({r["ecg_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicated saved record ID")
        saved_ids.update(r["ecg_id"] for r in rows)
        patients[split] = {r["patient_id"] for r in rows}
        # Sample evenly through saved arrays, including both ends, and recompute
        # from original WFDB data to catch indexing/label/scaler integration errors.
        for i in np.unique(np.linspace(0, len(rows)-1, 20, dtype=int)):
            row = rows[i]
            signal, _ = read_standard(ORIGINALS/row["filename_lr"])
            if not np.array_equal(x[i], normalize(signal, mean, scale)):
                raise ValueError("Persisted waveform does not match source/scaler")
            expected = np.array([c in row["labels"] for c in CLASSES], dtype=np.uint8)
            if not np.array_equal(y[i], expected):
                raise ValueError("Persisted label/record alignment mismatch")
    if patients["train"] & patients["validation"]:
        raise ValueError("Prepared patient overlap")
    if saved_ids & excluded_ids or saved_ids | excluded_ids != set(planned):
        raise ValueError("Saved/excluded record IDs do not partition the development plan")
    result = dict(checked_on=now(), prepared_hashes_verified=len(ready["outputs"]),
                  source_recomputed_records=40, patient_overlap=0,
                  full_label_alignment_checked_records=len(saved_ids),
                  eligible_excluded_partition_checked=True,
                  pipeline_source_hashes_verified=len(audit["pipeline_source_sha256"]),
                  training_ids_match_scaler=(json.loads((PREPARED/"scaler_training_ecg_ids.json").read_text(encoding="utf-8")) ==
                                             [r["ecg_id"] for r in json.loads((PREPARED/"train.records.json").read_text(encoding="utf-8"))]),
                  no_model_training=True)
    if not result["training_ids_match_scaler"]:
        raise ValueError("Scaler cohort differs from saved training cohort")
    write_json(REVIEW/"preprocessing_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["pilot", "process", "verify"])
    action = parser.parse_args().action
    {"pilot": crosscheck_pilot, "process": process, "verify": verify_prepared}[action]()
