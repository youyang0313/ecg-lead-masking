"""Post-hoc independent numerical audit using existing scikit-learn 0.24.2.

Does not select/retrain models or change the primary analysis.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parent.parent


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = Path(read(PROJECT/'training/latest_test.json')['directory'])
    manifest = read(out/'manifest.json')
    summary = read(out/'analysis/test_summary.json')
    fixed = read(out/'selected_models.json')
    assert digest(out/'test_predictions.npz') == manifest['prediction_sha256'] == summary['prediction_sha256']
    for item in fixed['checkpoints']:
        assert digest(Path(item['checkpoint_path'])) == item['checkpoint_sha256']
    with np.load(out/'test_predictions.npz', allow_pickle=False) as f:
        y, p, patients = f['targets'], f['probabilities'], f['patient_ids']
        keys = [(int(s), str(a)) for s, a in zip(f['seeds'], f['arms'])]
        configs, classes = f['configuration_ids'].tolist(), f['classes'].tolist()
    indices_a = [keys.index((s, 'A')) for s in [17, 29, 43]]
    indices_b = [keys.index((s, 'B')) for s in [17, 29, 43]]
    _, patient_index = np.unique(patients, return_inverse=True)
    patient_count = int(patient_index.max())+1
    max_auc_error = max_ap_error = 0.0
    auc = np.empty((6, 5, 5))
    for m in range(6):
        for c in range(5):
            for label in range(5):
                actual = summary['models'][m]['test_metrics'][configs[c]]['per_label'][classes[label]]
                auc[m, c, label] = roc_auc_score(y[:, label], p[m, c, :, label])
                max_auc_error = max(max_auc_error, abs(auc[m, c, label]-actual['auroc']))
                max_ap_error = max(max_ap_error, abs(average_precision_score(y[:, label], p[m, c, :, label])-actual['average_precision']))
    assert max(max_auc_error, max_ap_error) < 1e-12
    direct_gap = float((auc[indices_b, 1:]-auc[indices_a, 1:]).mean())
    assert abs(direct_gap-summary['primary']['estimate']) < 1e-12
    with np.load(out/'analysis/patient_bootstrap_draws.npz', allow_pickle=False) as f:
        draws = f['per_label_auroc']
    assert digest(out/'analysis/patient_bootstrap_draws.npz') == summary['bootstrap_draws_sha256']
    rng = np.random.default_rng(summary['bootstrap']['seed'])
    valid_index, invalid = 0, 0
    checked = []
    max_bootstrap_error = 0.0
    attempts = summary['bootstrap']['attempted_resamples']
    for attempt in range(attempts):
        sampled = rng.integers(0, patient_count, size=patient_count)
        weights = np.bincount(sampled, minlength=patient_count)[patient_index]
        positive = (weights[:, None]*y).sum(axis=0)
        negative = (weights[:, None]*(1-y)).sum(axis=0)
        if np.any(positive == 0) or np.any(negative == 0):
            invalid += 1
            continue
        if attempt in [0, attempts//2, attempts-1]:
            # Reconstruct the resampled recording cohort explicitly rather than
            # calling the custom weighted statistic under review.
            expanded = np.repeat(np.arange(len(y)), weights)
            reference = np.array([[[roc_auc_score(y[expanded, label], p[m, c, expanded, label])
                                    for label in range(5)] for c in range(5)] for m in range(6)])
            error = float(np.max(np.abs(reference-draws[valid_index])))
            max_bootstrap_error = max(max_bootstrap_error, error)
            assert error < 1e-12
            checked.append(attempt)
        valid_index += 1
    assert valid_index == len(draws) == summary['bootstrap']['valid_draws']
    assert invalid == summary['bootstrap']['invalid_draws']
    primary = (draws[:, indices_b, 1:]-draws[:, indices_a, 1:]).mean(axis=(1, 2, 3))
    complete = (draws[:, indices_b, 0]-draws[:, indices_a, 0]).mean(axis=(1, 2))
    assert np.allclose(np.percentile(primary, [2.5, 97.5]), summary['primary']['patient_bootstrap_95_ci'], rtol=0, atol=1e-14)
    assert np.allclose(np.percentile(complete, [2.5, 97.5]), summary['complete_input_change']['patient_bootstrap_95_ci'], rtol=0, atol=1e-14)
    result = dict(checked_on=datetime.now(timezone.utc).isoformat(), status='passed',
                  sklearn_version=sklearn.__version__, full_test_auc_and_ap_comparisons=300,
                  max_auroc_error=max_auc_error, max_ap_error=max_ap_error,
                  explicit_patient_resample_attempts_checked=checked,
                  explicit_resample_auroc_comparisons=len(checked)*150,
                  max_bootstrap_auroc_error=max_bootstrap_error,
                  all_2000_draw_validity_decisions_replayed=True,
                  primary_and_complete_percentile_intervals_recomputed=True,
                  fixed_checkpoint_hashes_verified=6, primary_estimate=direct_gap,
                  audit_source_sha256=digest(Path(__file__)))
    (out/'analysis/verify_test.py').write_bytes(Path(__file__).read_bytes())
    (out/'analysis/test_verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
