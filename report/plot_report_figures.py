"""Paper-size figures from frozen arrays/results; no training or test selection."""
from pathlib import Path
from hashlib import sha256
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'data_review'))
from preprocess_development import read_standard
from data_common import ORIGINALS


def main():
    out = ROOT/'paper/figures'
    out.mkdir(parents=True, exist_ok=True)
    run = Path(json.loads((PROJECT/'training/latest_test.json').read_text(encoding='utf-8'))['directory'])
    summary_path = run/'analysis/test_summary.json'
    s = json.loads(summary_path.read_text(encoding='utf-8'))
    assert json.loads((run/'analysis/test_verification.json').read_text(encoding='utf-8'))['status'] == 'passed'
    configs = ['S12', 'S6', 'S4', 'S3', 'S2']
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    files = []
    def save(fig, name):
        for extension in ['pdf', 'png']:
            p = out/(name+'.'+extension)
            fig.savefig(p, dpi=200, bbox_inches='tight')
            files.append(dict(file=p.name, sha256=sha256(p.read_bytes()).hexdigest()))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 2.65))
    for arm, color, offset, label in [('A', '#666666', -0.1, 'A: complete-input training'),
                                     ('B', '#0072B2', 0.1, 'B: complete + lead-masked training')]:
        values = np.array([s['macro_auroc_by_configuration'][c][arm+'_mean'] for c in configs])
        ci = np.array([s['macro_auroc_by_configuration'][c][arm+'_mean_auroc_patient_ci_descriptive'] for c in configs])
        ax.errorbar(np.arange(5)+offset, values, yerr=np.stack([values-ci[:, 0], ci[:, 1]-values]),
                    fmt='o', markersize=4, capsize=3, color=color, label=label)
    ax.axhline(0.5, color='#BBBBBB', ls='--', lw=0.7)
    ax.set(ylim=(0, 1), xlim=(-0.5, 4.5), ylabel='Macro-AUROC (mean of 3 seeds)',
           xticks=np.arange(5), xticklabels=['S12: all', 'S6: limb', 'S4: I/II/III/V2', 'S3: I/II/V2', 'S2: I/II'],
           xlabel='Prespecified input configuration')
    ax.legend(loc='lower left', frameon=False, fontsize=8)
    fig.tight_layout(pad=0.7)
    save(fig, 'test_discrimination')

    # Training ECG 12065 was the first metadata-selected pilot record, selected
    # before waveform inspection and never selected using test performance.
    rows = json.loads((PROJECT/'data/ptbxl_1_0_3/prepared/train.records.json').read_text(encoding='utf-8'))
    i = next(i for i, row in enumerate(rows) if row['ecg_id'] == 12065)
    row = rows[i]
    raw, _ = read_standard(ORIGINALS/row['filename_lr'])
    x = np.load(PROJECT/'data/ptbxl_1_0_3/prepared/train.signals.npy', mmap_mode='r')[i]
    leads = json.loads((ROOT/'lead_configurations.json').read_text(encoding='utf-8'))['canonical_names_for_this_document']
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 3.2), sharex=True)
    t = np.arange(1000)/100
    for axis, lead in zip(axes[:, 0], ['I', 'II', 'V2']):
        axis.plot(t, x[leads.index(lead)], lw=0.55, color='#0072B2')
        axis.set_ylabel(lead, rotation=0, labelpad=12)
    for axis, lead in zip(axes[:, 1], ['V1', 'V5', 'aVR']):
        axis.plot(t, x[leads.index(lead)], lw=0.5, color='#999999', label='Complete')
        axis.axhline(0, color='#0072B2', lw=0.8, label='S3 input')
        axis.set_ylabel(lead, rotation=0, labelpad=12)
    axes[0, 0].set_title('Retained in S3: I, II, V2', fontsize=9)
    axes[0, 1].set_title('Zeroed in S3 (blue); original (gray)', fontsize=8.5)
    for axis in axes.flat:
        axis.set_xlim(0, 10)
        axis.tick_params(labelsize=7)
    for axis in axes[-1]:
        axis.set_xlabel('Time (s)')
    fig.supylabel('Training-standardized amplitude (dimensionless)', fontsize=8, x=0.005)
    fig.tight_layout(pad=0.7, rect=(0.025, 0, 1, 1))
    save(fig, 'input_example')
    (out/'sources.json').write_text(json.dumps(dict(
        summary_sha256=sha256(summary_path.read_bytes()).hexdigest(),
        training_example_ecg_id=row['ecg_id'], training_example_fold=row['fold'],
        example_selection='First metadata-selected pilot record; selected before waveform inspection.',
        example_channels_displayed=['I', 'II', 'V2', 'V1', 'V5', 'aVR'],
        actual_source_shape=list(raw.shape), normalization='Full eligible TRAIN-only scaler, not pilot scaler.',
        generator_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), files=files), indent=2), encoding='utf-8')
    print(json.dumps(dict(figures=[f['file'] for f in files], training_example=row['ecg_id'])))


if __name__ == '__main__':
    main()
