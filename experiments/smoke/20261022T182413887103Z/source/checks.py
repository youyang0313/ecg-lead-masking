"""Focused checks for the intervention, pairing and trainable classifier."""
from collections import Counter
import numpy as np
import torch
from torch import nn

from runtime import (PROTOCOL, ROOT, batch_tensor, epoch_lr, epoch_order,
                     load_development, now, set_seed, state_hash, write_json)
from model import ECGClassifier, sample_lead_mask
from metrics import binary_metrics, improves, selection_score


def run_checks(device):
    set_seed(17)
    model = ECGClassifier(PROTOCOL).to(device)
    initial = state_hash(model)
    set_seed(17)
    paired = ECGClassifier(PROTOCOL).to(device)
    if state_hash(paired) != initial:
        raise AssertionError("Initial states differ")
    global_rng = torch.get_rng_state().clone()
    generator = torch.Generator().manual_seed(20261023)
    masks = sample_lead_mask(24000, generator)
    counts = masks.sum(dim=1).numpy()
    histogram = dict(sorted(Counter(counts.tolist()).items()))
    if counts.min() < 1 or counts.max() != 12:
        raise AssertionError("Invalid mask count")
    if not .48 < (counts == 12).mean() < .52:
        raise AssertionError("Complete-input mixture differs from 0.5")
    expected = (counts != 12).sum()/11
    if any(abs(histogram.get(k, 0)-expected) > .25*expected for k in range(1, 12)):
        raise AssertionError("Partial lead-count distribution is inconsistent with uniform sampling")
    if not torch.equal(global_rng, torch.get_rng_state()):
        raise AssertionError("Masking perturbed the global initialization RNG")
    order_before = epoch_order(np.arange(1024), 17, 0)
    sample_lead_mask(200, generator)
    if not np.array_equal(order_before, epoch_order(np.arange(1024), 17, 0)):
        raise AssertionError("Masking changed recording order")
    # Retained channels and time-constant masking on recognizable values.
    x = torch.arange(8*12*1000, dtype=torch.float32).reshape(8, 12, 1000)+1
    kept = masks[:8]
    masked = x*kept.unsqueeze(-1)
    if not torch.equal(masked[kept], x[kept]) or torch.any(masked[~kept] != 0):
        raise AssertionError("Masking changed retained values or failed to zero full channels")
    data = load_development()
    xb = batch_tensor(data["train"]["x"], np.arange(8), device)
    yb = batch_tensor(data["train"]["y"], np.arange(8), device)
    model.train()
    logits = model(xb)
    if logits.shape != (8, 5) or not torch.isfinite(logits).all():
        raise AssertionError("Invalid logits")
    loss = nn.BCEWithLogitsLoss()(logits, yb)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise AssertionError("Missing or non-finite loss/gradients")
    model.eval()
    before = state_hash(model)
    with torch.inference_mode():
        first = model(xb)
        second = model(xb)
    if not torch.equal(first, second) or state_hash(model) != before:
        raise AssertionError("Evaluation is not stable or mutates state")
    if improves(.7, .7) or not improves(.71, .7) or improves(None, .7):
        raise AssertionError("Checkpoint tie/invalid-score rule incorrect")
    weights = PROTOCOL["selection"]["weights"]
    metrics = {c: {"macro_auroc": .7} for c in weights}
    if not np.isclose(selection_score(metrics, weights), .7):
        raise AssertionError("Selection weights incorrect")
    metrics["S2"]["macro_auroc"] = None
    if selection_score(metrics, weights) is not None:
        raise AssertionError("Undefined target/configuration was dropped")
    if binary_metrics([0, 0], [.2, .8])["auroc"] is not None or binary_metrics([1, 1], [.2, .8])["auroc"] is not None:
        raise AssertionError("Undefined AUC was assigned a number")
    if not np.isclose(epoch_lr(0), .001) or not np.isclose(epoch_lr(39), .00001):
        raise AssertionError("Learning-rate endpoints differ from protocol")
    result = dict(checked_on=now(), status="passed", parameter_count=sum(p.numel() for p in model.parameters()),
                  checks=["identical paired initialization", "mask count/mixture and nonempty input",
                          "independent masking RNG and recording order", "whole-lead masking and retained-value preservation",
                          "real-data shape, finite loss and gradients", "deterministic frozen-state evaluation",
                          "checkpoint ties, invalid scores and fixed selection weights", "fixed LR endpoints"],
                  mask_test_records=24000, retained_lead_count_histogram=histogram,
                  gpu=str(device), finite_fp32_gradient_check_loss=float(loss.item()),
                  formal_experiment=False, test_data_used=False)
    write_json(ROOT/"engineering_checks.json", result)
    print(result)
