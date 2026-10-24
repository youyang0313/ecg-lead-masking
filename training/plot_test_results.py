"""Standalone report figures from verified test summaries (no model execution)."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent


def main():
    run = Path(json.loads((PROJECT/'training/latest_test.json').read_text(encoding='utf-8'))['directory'])
    summary_path = run/'analysis/test_summary.json'
    s = json.loads(summary_path.read_text(encoding='utf-8'))
    verification = json.loads((run/'analysis/test_verification.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'passed'
    out = run/'figures'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={'width_ratios': [1.5, 1]})
    configs = list(s['macro_auroc_by_configuration'])
    x = np.arange(len(configs))
    for arm, offset, color, label in [('A', -0.12, '#666666', 'A: complete-input training'),
                                      ('B', 0.12, '#0072B2', 'B: mixed lead-masked training')]:
        values = np.array([s['macro_auroc_by_configuration'][c][arm+'_mean'] for c in configs])
        ci = np.array([s['macro_auroc_by_configuration'][c][arm+'_mean_auroc_patient_ci_descriptive'] for c in configs])
        axes[0].errorbar(x+offset, values, yerr=np.stack([values-ci[:, 0], ci[:, 1]-values]),
                         fmt='o', capsize=3, color=color, label=label)
    axes[0].set(xticks=x, xticklabels=['S12\nAll leads', 'S6\nLimb leads',
                 'S4\nI/II/III/V2', 'S3\nI/II/V2', 'S2\nI/II'], ylim=(0, 1),
                 ylabel='Macro-AUROC (three-seed mean)', xlabel='Prespecified input configuration',
                 title='Independent-test diagnostic discrimination')
    axes[0].axhline(0.5, color='#BBBBBB', lw=0.8, ls='--')
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    values = np.array([s['primary']['estimate'], s['complete_input_change']['estimate']])
    ci = np.array([s['primary']['patient_bootstrap_95_ci'], s['complete_input_change']['patient_bootstrap_95_ci']])
    axes[1].errorbar(values, [1, 0], xerr=np.stack([values-ci[:, 0], ci[:, 1]-values]),
                     fmt='o', color='#0072B2', capsize=4)
    axes[1].axvline(0, color='#888888', lw=0.8)
    axes[1].set(yticks=[1, 0], yticklabels=['Partial-input mean\n(primary)', 'Complete input\n(S12)'],
                 ylim=(-0.6, 1.6), xlabel='Paired B - A macro-AUROC change', title='Training-intervention effects')
    for i in range(2):
        axes[1].annotate(f'{values[i]:+.4f} [{ci[i, 0]:+.4f}, {ci[i, 1]:+.4f}]',
                          (values[i], 1-i), xytext=(0, 14), textcoords='offset points', ha='center', fontsize=9)
    axes[1].margins(x=0.3)
    fig.text(0.05, 0.025, f"PTB-XL 1.0.3 fold 10: {s['records']:,} recordings, {s['patients']:,} patients. "
             '95% paired patient-bootstrap intervals; conditional on three trained seeds.\n'
             'Configuration intervals are descriptive. Lead identity changes across configurations; points do not define a lead-count dose response.', fontsize=8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    files = []
    for extension in ['png', 'pdf', 'svg']:
        path = out/('independent_test_results.'+extension)
        fig.savefig(path, dpi=220, bbox_inches='tight')
        files.append(dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    plt.close(fig)
    (out/'figure_sources.json').write_text(json.dumps(dict(
        summary_path=str(summary_path), summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), files=files), indent=2), encoding='utf-8')
    print(str(out/'independent_test_results.png'))


if __name__ == '__main__':
    main()
