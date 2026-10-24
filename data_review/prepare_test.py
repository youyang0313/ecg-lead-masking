"""Explicit fold-10 release after development/model selection is complete.

Uses the existing diagnostic mapping, WFDB reader and TRAIN-fitted scaler.
No development artifacts or original protocol files are rewritten.
"""
import argparse
import ast
import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_common import (CLASSES, CONFIG_PATH, DATA, ORIGINALS, PREPARED,
                         PROJECT, PROTOCOL_PATH, REVIEW, hash_file, input_hashes,
                         validate_stem, write_json)
from prepare_development import now, retrieve
from preprocess_development import check_masks, normalize, read_standard
import numpy as np

OUT = DATA / 'test_prepared'
PLAN = DATA / 'test_plan.json'
SELECTED = PROJECT / 'experiments/train/20261022T183345197538Z/analysis/selected_models.json'


def fixed_models():
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    for path, expected in [(PROTOCOL_PATH, selected['protocol_sha256']),
                           (CONFIG_PATH, selected['configuration_sha256']),
                           (PREPARED/'scaler.json', selected['scaler_sha256']),
                           (PROJECT/'training/model.py', selected['architecture_source_sha256'])]:
        if hash_file(path) != expected:
            raise ValueError('Fixed input changed: '+str(path))
    if {(r['seed'], r['arm']) for r in selected['checkpoints']} != {
            (s, a) for s in [17, 29, 43] for a in ['A', 'B']}:
        raise ValueError('Unexpected checkpoint set')
    for r in selected['checkpoints']:
        if hash_file(r['checkpoint_path']) != r['checkpoint_sha256']:
            raise ValueError('Fixed checkpoint changed')
    return selected


def make_plan():
    fixed_models()  # MUST precede diagnostic annotation parsing.
    if PLAN.exists():
        return load_plan()
    development = json.loads((DATA/'development_plan.json').read_text(encoding='utf-8'))
    if input_hashes() != development['input_sha256']:
        raise ValueError('Metadata/design changed since development')
    release = dict(released_on=now(), selected_models_sha256=hash_file(SELECTED),
                   selection_fixed_on=json.loads(SELECTED.read_text(encoding='utf-8'))['fixed_on'],
                   analysis_protocol_sha256=hash_file(PROTOCOL_PATH),
                   reason='User authorized independent test after main training and model selection.',
                   changes_to_scientific_design=[], test_predictions_exist=False)
    write_json(DATA/'test_release.json', release)
    with (REVIEW/'metadata/scp_statements.csv').open(encoding='utf-8-sig', newline='') as f:
        codes = list(csv.DictReader(f))
    key = next(iter(codes[0]))
    mapping = {r[key]: r['diagnostic_class'] for r in codes if r['diagnostic'] == '1.0'}
    records, excluded, raw_patients, development_patients = [], [], set(), set()
    raw_count = 0
    with (REVIEW/'metadata/ptbxl_database.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if int(row['strat_fold']) != 10:
                development_patients.add(row['patient_id'])
                continue
            raw_count += 1
            raw_patients.add(row['patient_id'])
            ecg_id = int(row['ecg_id'])
            validate_stem(row['filename_lr'], ecg_id)
            labels = sorted({mapping[c] for c in ast.literal_eval(row['scp_codes']) if c in mapping})
            item = dict(ecg_id=ecg_id, patient_id=row['patient_id'], fold=10,
                        split='test', filename_lr=row['filename_lr'], labels=labels)
            if not set(labels) <= set(CLASSES):
                raise ValueError('Unknown diagnostic class')
            if labels:
                records.append(item)
            else:
                excluded.append(dict(item, reason='no_diagnostic_superclass'))
    if raw_patients & development_patients:
        raise ValueError('Test/development patient overlap')
    plan = dict(created_on=now(), input_sha256=input_hashes(),
                selected_models_sha256=hash_file(SELECTED), test_release_sha256=hash_file(DATA/'test_release.json'),
                dataset='PTB-XL 1.0.3', classes=CLASSES,
                raw_records=raw_count, raw_patients=len(raw_patients),
                patient_overlap_with_development=0, records=records, metadata_exclusions=excluded)
    write_json(PLAN, plan)
    print(json.dumps(dict(stage='test_plan', raw=raw_count, eligible_metadata=len(records),
                          excluded_empty_labels=len(excluded), patient_overlap=0)), flush=True)
    return plan


def load_plan():
    fixed_models()
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if plan['input_sha256'] != input_hashes() or plan['selected_models_sha256'] != hash_file(SELECTED):
        raise ValueError('Test plan input changed')
    return plan


def download_index():
    path = DATA/'test_download_manifest.jsonl'
    return {r['relative_path']: r for r in
            (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())} if path.exists() else {}


def fetch(workers):
    plan = load_plan()
    jobs = [(r, suffix) for r in plan['records'] for suffix in ['.hea', '.dat']]
    previous = download_index()
    counts = Counter()
    started = time.monotonic()
    with (DATA/'test_download_manifest.jsonl').open('a', encoding='utf-8') as log:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(retrieve, job, previous, {}) for job in jobs]
            for i, f in enumerate(as_completed(pending), 1):
                entry = f.result()
                log.write(json.dumps(entry, ensure_ascii=False)+'\n')
                counts[entry['status']] += 1
                if i % 250 == 0 or i == len(jobs):
                    log.flush()
                    print(json.dumps(dict(stage='download_test', completed=i, total=len(jobs),
                                          counts=dict(counts), seconds=round(time.monotonic()-started, 1))), flush=True)
    write_json(DATA/'test_download_summary.json', dict(completed_on=now(), counts=dict(counts),
               total_files=len(jobs), elapsed_seconds=time.monotonic()-started))
    if counts['failed']:
        raise RuntimeError('Incomplete acquisition; network failures are not cohort exclusions')


def process():
    plan = load_plan()
    if (OUT/'READY.json').exists():
        raise ValueError('Test data already prepared; use verify')
    OUT.mkdir(exist_ok=True)
    index = download_index()
    for row in plan['records']:
        for suffix in ['.hea', '.dat']:
            relative = row['filename_lr']+suffix
            e = index.get(relative)
            if not e or e['status'] != 'archived' or hash_file(ORIGINALS/relative) != e['sha256']:
                raise ValueError('Incomplete or altered original: '+relative)
    scaler = json.loads((PREPARED/'scaler.json').read_text(encoding='utf-8'))
    mean, scale = np.array(scaler['mean_mV']), np.array(scaler['applied_scale_mV'])
    rows, failures, checks = [], [], []
    def read(row):
        try:
            x, info = read_standard(ORIGINALS/row['filename_lr'])
            return row, x, info, None
        except Exception as exc:
            return row, None, None, type(exc).__name__+': '+str(exc)[:300]
    # Qualification is shared across all models/configurations and outcome-blind.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row, _, info, error in pool.map(read, plan['records']):
            if error:
                failures.append(dict(row, reason=error))
            else:
                rows.append(row)
                checks.append(dict(ecg_id=row['ecg_id'], **info))
    if not rows:
        raise ValueError('No eligible test records')
    x = np.lib.format.open_memmap(OUT/'test.signals.npy', mode='w+', dtype=np.float32,
                                  shape=(len(rows), 12, 1000))
    masks = 0
    for i, row in enumerate(rows):
        signal, _ = read_standard(ORIGINALS/row['filename_lr'])
        z = normalize(signal, mean, scale)
        if not np.isfinite(z).all():
            raise ValueError('Non-finite normalized test signal')
        masks += check_masks(z)
        x[i] = z
    x.flush()
    del x
    y = np.array([[c in r['labels'] for c in CLASSES] for r in rows], dtype=np.uint8)
    np.save(OUT/'test.labels.npy', y, allow_pickle=False)
    write_json(OUT/'test.records.json', rows)
    write_json(OUT/'waveform_exclusions.json', failures)
    write_json(OUT/'record_checks.json', checks)
    outputs = [dict(file=name, sha256=hash_file(OUT/name), bytes=(OUT/name).stat().st_size)
               for name in ['test.signals.npy', 'test.labels.npy', 'test.records.json',
                            'waveform_exclusions.json', 'record_checks.json']]
    audit = dict(completed_on=now(), records=len(rows), patients=len({r['patient_id'] for r in rows}),
                 raw_records=plan['raw_records'], metadata_exclusions=len(plan['metadata_exclusions']),
                 waveform_exclusions=len(failures), mask_invariant_checks=masks,
                 label_positive_counts=dict(zip(CLASSES, y.sum(axis=0).astype(int).tolist())),
                 patient_overlap_with_development=0, scaler_refitted=False,
                 scaler_sha256=hash_file(PREPARED/'scaler.json'), selected_models_sha256=hash_file(SELECTED),
                 plan_sha256=hash_file(PLAN), outputs=outputs,
                 preparation_source_sha256=hash_file(Path(__file__)))
    write_json(OUT/'audit.json', audit)
    write_json(OUT/'READY.json', dict(status='complete', audit_sha256=hash_file(OUT/'audit.json'), outputs=outputs))
    verify()
    print(json.dumps(audit, ensure_ascii=False), flush=True)


def verify():
    plan = load_plan()
    ready = json.loads((OUT/'READY.json').read_text(encoding='utf-8'))
    if ready['audit_sha256'] != hash_file(OUT/'audit.json'):
        raise ValueError('Test audit hash mismatch')
    for e in ready['outputs']:
        if hash_file(OUT/e['file']) != e['sha256']:
            raise ValueError('Test artifact changed')
    rows = json.loads((OUT/'test.records.json').read_text(encoding='utf-8'))
    x = np.load(OUT/'test.signals.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(OUT/'test.labels.npy', allow_pickle=False)
    planned = {r['ecg_id']: r for r in plan['records']}
    excluded = json.loads((OUT/'waveform_exclusions.json').read_text(encoding='utf-8'))
    if {r['ecg_id'] for r in rows} & {r['ecg_id'] for r in excluded} or len(rows)+len(excluded) != len(planned):
        raise ValueError('Invalid shared cohort partition')
    assert len({r['ecg_id'] for r in rows}) == len(rows)
    assert all(r == planned[r['ecg_id']] and r['fold'] == 10 for r in rows)
    assert x.shape == (len(rows), 12, 1000) and x.dtype == np.float32
    assert np.array_equal(y, [[c in r['labels'] for c in CLASSES] for r in rows])
    scaler = json.loads((PREPARED/'scaler.json').read_text(encoding='utf-8'))
    for i in np.unique(np.linspace(0, len(rows)-1, 30, dtype=int)):
        signal, _ = read_standard(ORIGINALS/rows[i]['filename_lr'])
        assert np.array_equal(x[i], normalize(signal, np.array(scaler['mean_mV']), np.array(scaler['applied_scale_mV'])))
    for split in ['train', 'validation']:
        development = json.loads((PREPARED/(split+'.records.json')).read_text(encoding='utf-8'))
        assert not {r['patient_id'] for r in rows} & {r['patient_id'] for r in development}
    write_json(OUT/'verification.json', dict(checked_on=now(), status='passed', records=len(rows),
               source_recomputed_records=30, patient_overlap=0, all_labels_aligned=True,
               scaler_not_refitted=True, output_hashes_verified=len(ready['outputs'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'fetch', 'process', 'verify'])
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error('workers must be 1..32')
    {'plan': make_plan, 'fetch': lambda: fetch(args.workers), 'process': process, 'verify': verify}[args.action]()
