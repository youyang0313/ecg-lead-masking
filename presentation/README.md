# Lectrace presentation

The 10-minute execution-explanation presentation of the project, as a Python program stepped
through in the Lectrace viewer.

| File | What it is |
| --- | --- |
| `01_ecg_lead_masking.py` | the lecture; `main()` is what Lectrace traces |
| `_helpers.py` | data loading, the archived model class, BatchNorm hooks, repair, AUROC, charts (not traced) |
| `prepare_assets.py` | writes `assets/results.json` and copies two figures from the experiments' result files |
| `assets/` | every number the talk shows that is not computed live, plus two figures |
| `PRESENTATION_SCRIPT.md` | word-for-word spoken script with a click cue for every step, timings and likely questions |
| `lectrace.toml` | site title |

## Structure

Problem and approach (about 2.5 minutes), then four execution states, then the key result and its
healthcare interpretation:

1. **The input**: a zero-filled ECG breaks the Einthoven relation; deriving limb leads restores it.
2. **The forward pass**: BatchNorm input statistics of the complete-input model drift under missing
   leads; the masked model drifts less; per-label AUROC shows the consequence.
3. **The repair**: deriving limb leads and re-estimating BatchNorm statistics recovers most of the gap
   without changing a weight.
4. **The result**: the prespecified effect, and where masked training still beats the repair.

States 1–3 run the archived seed-17 models (S12-only selection) on the full PTB-XL test set on CPU;
the build takes about 25 seconds.

## Run

Requires Python 3.11+ with `lectrace`, `numpy` and `torch` (CPU is enough), plus the project's data
and archived checkpoints in `../data` and `../experiments`.

```
python prepare_assets.py
lectrace serve 01_ecg_lead_masking.py      # present from http://localhost:7000
lectrace build 01_ecg_lead_masking.py -o _site
```

Everything that touches the data runs directly in `main()`: Lectrace records every local variable
of functions called from `main()`, which would serialize the whole test set at each step.
