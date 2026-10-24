"""One independent-test inference pass using the six already-fixed checkpoints."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from runtime import (CLASSES, CONFIG, PROJECT, PROTOCOL, PREPARED, configure,
                     hash_file, now, state_hash, write_json)
from experiment import evaluate
from model import ECGClassifier
import numpy as np
import torch

SELECTED = PROJECT/'experiments/train/20261022T183345197538Z/analysis/selected_models.json'
DATA = PREPARED.parent/'test_prepared'


def run():
    device = configure()
    selected = json.loads(SELECTED.read_text(encoding='utf-8'))
    for path, expected in [(PROJECT/'report/protocol_v1.json', selected['protocol_sha256']),
                           (PROJECT/'report/lead_configurations.json', selected['configuration_sha256']),
                           (PROJECT/'training/model.py', selected['architecture_source_sha256']),
                           (PREPARED/'scaler.json', selected['scaler_sha256'])]:
        assert hash_file(path) == expected, str(path)
    ready = json.loads((DATA/'READY.json').read_text(encoding='utf-8'))
    assert ready['status'] == 'complete' and ready['audit_sha256'] == hash_file(DATA/'audit.json')
    audit = json.loads((DATA/'audit.json').read_text(encoding='utf-8'))
    assert audit['selected_models_sha256'] == hash_file(SELECTED)
    for e in ready['outputs']:
        assert hash_file(DATA/e['file']) == e['sha256']
    data = dict(x=np.load(DATA/'test.signals.npy', mmap_mode='r', allow_pickle=False),
                y=np.load(DATA/'test.labels.npy', allow_pickle=False),
                rows=json.loads((DATA/'test.records.json').read_text(encoding='utf-8')))
    assert data['x'].shape == (len(data['rows']), 12, 1000)
    assert np.array_equal(data['y'], [[c in r['labels'] for c in CLASSES] for r in data['rows']])
    for split in ['train', 'validation']:
        dev = json.loads((PREPARED/(split+'.records.json')).read_text(encoding='utf-8'))
        assert not {r['patient_id'] for r in data['rows']} & {r['patient_id'] for r in dev}
    out = PROJECT/'experiments/test'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'
    source.mkdir()
    for name in ['evaluate_test.py', 'analyze_test.py', 'test_statistics.py', 'model.py', 'runtime.py', 'metrics.py', 'experiment.py']:
        shutil.copy2(PROJECT/'training'/name, source/name)
    shutil.copy2(PROJECT/'data_review/prepare_test.py', source/'prepare_test.py')
    shutil.copy2(PROJECT/'report/protocol_v1.json', source/'protocol_v1.json')
    shutil.copy2(PROJECT/'report/lead_configurations.json', source/'lead_configurations.json')
    shutil.copy2(SELECTED, out/'selected_models.json')
    manifest = dict(started_on=now(), status='running', selected_models_sha256=hash_file(SELECTED),
                    test_ready_sha256=hash_file(DATA/'READY.json'),
                    test_audit_sha256=hash_file(DATA/'audit.json'), records=len(data['rows']),
                    source_sha256={p.name: hash_file(p) for p in source.glob('*.py')},
                    scientific_design_changes=[], checkpoint_updates=False,
                    test_refitting=False, precision='float32; autocast disabled',
                    torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(0))
    write_json(out/'manifest.json', manifest)
    predictions, records = [], []
    for item in selected['checkpoints']:
        checkpoint_path = Path(item['checkpoint_path'])
        assert hash_file(checkpoint_path) == item['checkpoint_sha256']
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == item['seed'] and checkpoint['arm'] == item['arm']
        assert checkpoint['epoch'] == item['selected_epoch']
        model = ECGClassifier(PROTOCOL).to(device)
        model.load_state_dict(checkpoint['model_state'])
        assert state_hash(model) == item['selected_state_sha256']
        metrics, probabilities, seconds = evaluate(model, data, device)
        assert state_hash(model) == item['selected_state_sha256']
        predictions.append(probabilities)
        records.append(dict(**item, test_metrics=metrics, inference_seconds=seconds,
                            state_unchanged_after_inference=True))
        print(json.dumps(dict(stage='test_inference_complete', seed=item['seed'], arm=item['arm'],
                              seconds=round(seconds, 3))), flush=True)
        del model, checkpoint
    archive = out/'test_predictions.npz'
    np.savez_compressed(archive, probabilities=np.stack(predictions), targets=data['y'],
                        ecg_ids=np.array([r['ecg_id'] for r in data['rows']]),
                        patient_ids=np.array([r['patient_id'] for r in data['rows']]),
                        seeds=np.array([r['seed'] for r in records]), arms=np.array([r['arm'] for r in records]),
                        configuration_ids=np.array([c['id'] for c in CONFIG['configurations']]), classes=np.array(CLASSES))
    manifest.update(status='complete', completed_on=now(), models=records,
                    prediction_sha256=hash_file(archive), inference_seconds_sum=sum(r['inference_seconds'] for r in records))
    write_json(out/'manifest.json', manifest)
    write_json(PROJECT/'training/latest_test.json', dict(directory=str(out), manifest_sha256=hash_file(out/'manifest.json')))
    print(json.dumps(dict(status='test_inference_complete', directory=str(out))), flush=True)


if __name__ == '__main__':
    run()
