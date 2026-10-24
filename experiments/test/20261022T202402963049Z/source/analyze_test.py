"""Test estimates and protocol-specified paired patient bootstrap; no fitting."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from metrics import multilabel_metrics
from test_statistics import WeightedAUROC, effects, patient_weights, self_check
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def main():
    started = time.perf_counter()
    out = Path(read(PROJECT/'training/latest_test.json')['directory'])
    manifest = read(out/'manifest.json')
    assert manifest['status'] == 'complete'
    assert digest(out/'test_predictions.npz') == manifest['prediction_sha256']
    for name in ['analyze_test.py', 'test_statistics.py', 'metrics.py']:
        assert digest(Path(__file__).parent/name) == manifest['source_sha256'][name]
    selected = read(out/'selected_models.json')
    assert digest(out/'selected_models.json') == manifest['selected_models_sha256']
    protocol = read(PROJECT/'report/protocol_v1.json')
    assert digest(PROJECT/'report/protocol_v1.json') == selected['protocol_sha256']
    settings = protocol['evaluation']['confidence_interval']
    check = self_check()
    with np.load(out/'test_predictions.npz', allow_pickle=False) as f:
        p, y = f['probabilities'], f['targets']
        patients, ids = f['patient_ids'], f['ecg_ids']
        configurations, classes = f['configuration_ids'].tolist(), f['classes'].tolist()
        model_keys = [(int(s), str(a)) for s, a in zip(f['seeds'], f['arms'])]
    paired_seeds = protocol['training']['paired_seeds']
    assert classes == protocol['dataset']['classes'] and len(set(ids.tolist())) == len(y)
    assert configurations == ['S12', 'S6', 'S4', 'S3', 'S2']
    assert model_keys == [(r['seed'], r['arm']) for r in selected['checkpoints']]
    assert p.shape == (6, 5, len(y), 5) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    models = []
    for i, (seed, arm) in enumerate(model_keys):
        metrics = {c: multilabel_metrics(y, p[i, j], classes) for j, c in enumerate(configurations)}
        assert metrics == manifest['models'][i]['test_metrics']
        models.append(dict(seed=seed, arm=arm, selected_epoch=selected['checkpoints'][i]['selected_epoch'],
                           test_metrics=metrics))
    estimator = WeightedAUROC(y, p)
    auc = estimator(np.ones(len(y)))
    analysis = out/'analysis'
    analysis.mkdir(exist_ok=True)
    write(analysis/'statistical_checks.json', check)
    if auc is None:
        write(analysis/'test_summary.json', dict(status='undefined_primary', models=models,
              reason='At least one of the five fixed targets lacks positives or negatives.',
              independent_test_result=True, dataset_split='test_fold_10'))
        return
    expected = np.array([[[m['test_metrics'][c]['per_label'][label]['auroc'] for label in classes]
                          for c in configurations] for m in models])
    assert np.allclose(auc, expected, rtol=0, atol=1e-14)
    estimate = effects(auc, model_keys, paired_seeds, configurations)
    unique_patients, patient_index = np.unique(patients, return_inverse=True)
    rng = np.random.default_rng(settings['seed'])
    samples, invalid = [], 0
    # Patients are drawn once per attempt and shared across every seed/arm/configuration.
    for attempt in range(settings['attempted_resamples']):
        sampled = rng.integers(0, len(unique_patients), size=len(unique_patients))
        weights = patient_weights(patient_index, sampled)
        draw = estimator(weights)
        if draw is None:
            invalid += 1
        else:
            samples.append(draw)
        if (attempt+1) % 250 == 0:
            print(json.dumps(dict(stage='patient_bootstrap', attempted=attempt+1,
                                  valid=len(samples), invalid=invalid,
                                  seconds=round(time.perf_counter()-started, 1))), flush=True)
    if not samples:
        raise ValueError('No valid bootstrap draws; no numeric confidence interval')
    samples = np.stack(samples)
    np.savez_compressed(analysis/'patient_bootstrap_draws.npz', per_label_auroc=samples)
    bs_effects = [effects(draw, model_keys, paired_seeds, configurations) for draw in samples]
    alpha = (1-settings['level'])/2
    def interval(values):
        return np.quantile(values, [alpha, 1-alpha], method='linear').tolist() if np.lib.NumpyVersion(np.__version__) >= '1.22.0' else np.quantile(values, [alpha, 1-alpha], interpolation='linear').tolist()
    primary_ci = interval([r['primary'] for r in bs_effects])
    complete_ci = interval([r['complete'] for r in bs_effects])
    configuration_summary = {}
    for j, c in enumerate(configurations):
        entry = {}
        for arm in ['A', 'B']:
            indices = [i for i, key in enumerate(model_keys) if key[1] == arm]
            values = auc[indices, j].mean(axis=-1)
            entry[arm+'_mean'] = float(values.mean())
            entry[arm+'_seed_sd'] = float(values.std(ddof=1))
            entry[arm+'_macro_average_precision'] = float(np.mean([
                models[i]['test_metrics'][c]['macro_average_precision'] for i in indices]))
            entry[arm+'_mean_auroc_patient_ci_descriptive'] = interval(samples[:, indices, j].mean(axis=(1, 2)))
        entry['paired_B_minus_A'] = float(estimate['per_configuration'][j])
        entry['paired_difference_patient_ci_descriptive'] = interval([r['per_configuration'][j] for r in bs_effects])
        configuration_summary[c] = entry
    partial_means, partial_seed_sd = {}, {}
    for arm in ['A', 'B']:
        indices = [i for i, key in enumerate(model_keys) if key[1] == arm]
        values = auc[indices, 1:].mean(axis=(1, 2))
        partial_means[arm] = float(values.mean())
        partial_seed_sd[arm] = float(values.std(ddof=1))
    summary = dict(completed_on=datetime.now(timezone.utc).isoformat(), status='complete',
                   dataset_split='test_fold_10', independent_test_result=True,
                   records=len(y), patients=len(unique_patients),
                   positive_counts=dict(zip(classes, y.sum(axis=0).astype(int).tolist())),
                   model_keys=model_keys, paired_seeds=paired_seeds, models=models,
                   macro_auroc_by_configuration=configuration_summary,
                   partial_macro_auroc_means=partial_means, partial_macro_auroc_seed_sd=partial_seed_sd,
                   primary=dict(estimate=estimate['primary'], patient_bootstrap_95_ci=primary_ci,
                                paired_seed_effects=estimate['per_seed_partial'].tolist(),
                                paired_seed_effect_sd=float(estimate['per_seed_partial'].std(ddof=1)),
                                positive_gain_supported=bool(primary_ci[0] > 0)),
                   complete_input_change=dict(estimate=estimate['complete'], patient_bootstrap_95_ci=complete_ci,
                                              clinical_equivalence_tested=False),
                   bootstrap=dict(**settings, valid_draws=len(samples), invalid_draws=invalid,
                                  shared_patient_draws=True, seed_prediction_ensemble=False,
                                  interval_conditional_on_trained_seed_set=True),
                   interpretation='Primary interval assesses the fixed trained-seed set; secondary intervals are descriptive. No clinical safety/equivalence claim.',
                   inference_seconds_sum=manifest['inference_seconds_sum'],
                   statistics_elapsed_seconds=time.perf_counter()-started,
                   prediction_sha256=manifest['prediction_sha256'],
                   selected_models_sha256=manifest['selected_models_sha256'],
                   bootstrap_draws_sha256=digest(analysis/'patient_bootstrap_draws.npz'))
    write(analysis/'test_summary.json', summary)
    print(json.dumps(dict(status='test_analysis_complete', primary=summary['primary'],
                          complete=summary['complete_input_change'], partial_means=partial_means)), flush=True)


if __name__ == '__main__':
    main()
