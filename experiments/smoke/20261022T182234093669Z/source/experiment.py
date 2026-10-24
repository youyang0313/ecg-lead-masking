"""Explicit actions: check, smoke (short engineering run), train (full protocol).

Smoke checkpoints are stored separately and are never loaded by main training.
Only training and validation data are supported. No test evaluation is exposed.
"""
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import argparse
import json
import time
import traceback

from runtime import (CLASSES, CONFIG, LEADS, PREPARED, PROJECT, PROTOCOL, ROOT,
                     batch_tensor, configure, epoch_lr, epoch_order, hash_file,
                     load_development, now, provenance, set_seed, state_hash, write_json)
from model import ECGClassifier, sample_lead_mask
from metrics import improves, multilabel_metrics, selection_score
import numpy as np
import torch
from torch import nn


def evaluate(model, data, device):
    before = state_hash(model)
    model.eval()
    metrics, predictions = {}, []
    batch_size = PROTOCOL["training"]["batch_size"]
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for configuration in CONFIG["configurations"]:
            keep = torch.tensor([n in configuration["available"] for n in LEADS], device=device).view(1, 12, 1)
            scores = []
            for start in range(0, len(data["y"]), batch_size):
                indices = np.arange(start, min(start+batch_size, len(data["y"])))
                x = batch_tensor(data["x"], indices, device)*keep
                probabilities = torch.sigmoid(model(x)).float()
                if not torch.isfinite(probabilities).all():
                    raise FloatingPointError("Non-finite validation output")
                scores.append(probabilities.cpu().numpy())
            scores = np.concatenate(scores)
            metrics[configuration["id"]] = multilabel_metrics(data["y"], scores, CLASSES)
            predictions.append(scores)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    if state_hash(model) != before:
        raise AssertionError("Evaluation mutated model parameters or BatchNorm statistics")
    return metrics, np.stack(predictions), seconds


def train_epoch(model, optimizer, scaler, data, indices, seed, epoch, arm, mask_generator, device):
    model.train()
    order = epoch_order(indices, seed, epoch)
    lr = epoch_lr(epoch)
    for group in optimizer.param_groups:
        group["lr"] = lr
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    batch_size = PROTOCOL["training"]["batch_size"]
    total_loss = 0.0
    counts = Counter()
    finite_grad_norms = []
    overflow_steps = optimizer_steps = 0
    scale_before = float(scaler.get_scale())
    torch.cuda.synchronize()
    started = time.perf_counter()
    for batch_number, start in enumerate(range(0, len(order), batch_size), 1):
        chosen = order[start:start+batch_size]
        x = batch_tensor(data["x"], chosen, device)
        y = batch_tensor(data["y"], chosen, device)
        keep = (sample_lead_mask(len(chosen), mask_generator) if arm == "B" else
                torch.ones((len(chosen), 12), dtype=torch.bool))
        counts.update(keep.sum(dim=1).tolist())
        x = x*keep.to(device, non_blocking=True).unsqueeze(-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = loss_fn(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = float(torch.sqrt(sum(p.grad.detach().float().square().sum()
                                    for p in model.parameters() if p.grad is not None)).item())
        if np.isfinite(norm):
            finite_grad_norms.append(norm)
            optimizer_steps += 1
        else:
            overflow_steps += 1
        # Dynamic scaling skips optimizer updates on overflowing gradients.
        # There is no gradient clipping, loss repair or masked-label removal.
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item())*len(chosen)
        if batch_number % 50 == 0:
            print(json.dumps(dict(stage="training", arm=arm, seed=seed, epoch=epoch+1,
                                  batch=batch_number, total_batches=(len(order)+batch_size-1)//batch_size)), flush=True)
    torch.cuda.synchronize()
    if not optimizer_steps:
        raise FloatingPointError("No finite-gradient optimizer step in this epoch")
    return dict(epoch=epoch+1, learning_rate=lr, training_records=len(order),
                mean_bce=total_loss/len(order), training_seconds=time.perf_counter()-started,
                batches=(len(order)+batch_size-1)//batch_size,
                finite_gradient_optimizer_steps=optimizer_steps,
                overflow_skipped_steps=overflow_steps,
                finite_gradient_norm_min=min(finite_grad_norms),
                finite_gradient_norm_max=max(finite_grad_norms),
                grad_scale_start=scale_before, grad_scale_end=float(scaler.get_scale()),
                retained_lead_count_histogram={str(k): v for k, v in sorted(counts.items())},
                recording_order_sha256=sha256(order.astype("<i8").tobytes()).hexdigest())


def save_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def run_arm(arm, seed, epochs, indices, data, destination, device, mode):
    destination.mkdir(parents=True, exist_ok=False)
    torch.cuda.empty_cache()
    set_seed(seed)
    model = ECGClassifier(PROTOCOL).to(device)
    initial_hash = state_hash(model)
    training = PROTOCOL["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"],
                                 betas=tuple(training["betas"]), eps=training["epsilon"])
    scaler = torch.amp.GradScaler("cuda")
    mask_generator = torch.Generator(device="cpu").manual_seed(seed+1000003)
    torch.cuda.reset_peak_memory_stats()
    history = []
    best_score = None
    checkpoint_path = destination/"best.pt"
    for epoch in range(epochs):
        record = train_epoch(model, optimizer, scaler, data["train"], indices,
                             seed, epoch, arm, mask_generator, device)
        metrics, predictions, seconds = evaluate(model, data["validation"], device)
        score = selection_score(metrics, PROTOCOL["selection"]["weights"])
        record.update(validation_seconds=seconds, validation_metrics=metrics,
                      selection_score=score)
        if improves(score, best_score):
            best_score = score
            payload = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                           scaler_state=scaler.state_dict(), mask_rng_state=mask_generator.get_state(),
                           torch_rng_state=torch.get_rng_state(), cuda_rng_states=torch.cuda.get_rng_state_all(),
                           protocol=PROTOCOL, protocol_sha256=hash_file(PROJECT/"report"/"protocol_v1.json"),
                           epoch=epoch+1, seed=seed, arm=arm, mode=mode,
                           formal_experiment=(mode == "train"), initial_state_sha256=initial_hash,
                           selected_state_sha256=state_hash(model), selection_score=score,
                           validation_metrics=metrics)
            save_checkpoint(checkpoint_path, payload)
            temp = destination/"validation_predictions.tmp"
            with temp.open("wb") as f:
                np.savez_compressed(f, probabilities=predictions, targets=data["validation"]["y"],
                                    ecg_ids=np.array([r["ecg_id"] for r in data["validation"]["rows"]]),
                                    configuration_ids=np.array([c["id"] for c in CONFIG["configurations"]]))
            temp.replace(destination/"validation_predictions.npz")
        history.append(record)
        write_json(destination/"history.json", history)
        print(json.dumps(dict(stage="epoch_complete", mode=mode, seed=seed, arm=arm,
                              epoch=epoch+1, mean_bce=record["mean_bce"],
                              train_seconds=round(record["training_seconds"], 3),
                              validation_seconds=round(seconds, 3), selection_score=score,
                              overflow_skipped_steps=record["overflow_skipped_steps"])), flush=True)
    if best_score is None:
        raise ValueError("No valid checkpoint; all fixed-five-label validation scores were undefined")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    trained_hash = state_hash(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored = ECGClassifier(PROTOCOL).to(device)
    restored.load_state_dict(checkpoint["model_state"])
    if state_hash(restored) != checkpoint["selected_state_sha256"]:
        raise AssertionError("Checkpoint state readback mismatch")
    restored_metrics, restored_predictions, _ = evaluate(restored, data["validation"], device)
    with np.load(destination/"validation_predictions.npz", allow_pickle=False) as saved:
        identical = np.array_equal(restored_predictions, saved["probabilities"])
    if not identical or selection_score(restored_metrics, PROTOCOL["selection"]["weights"]) != best_score:
        raise AssertionError("Checkpoint readback changed validation predictions or selection score")
    result = dict(arm=arm, seed=seed, mode=mode, formal_experiment=(mode == "train"),
                  completed_epochs=epochs, planned_main_epochs=training["epoch_budget"],
                  training_records=len(indices), validation_records=len(data["validation"]["y"]),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  initial_state_sha256=initial_hash, trained_state_sha256=trained_hash,
                  parameters_updated=(trained_hash != initial_hash),
                  selected_epoch=checkpoint["epoch"], selected_validation_score=best_score,
                  checkpoint_readback_predictions_exact=identical,
                  checkpoint_sha256=hash_file(checkpoint_path),
                  train_validation_peak_allocated_bytes=peak_allocated,
                  train_validation_peak_reserved_bytes=peak_reserved,
                  history_path=str(destination/"history.json"), checkpoint_path=str(checkpoint_path))
    write_json(destination/"summary.json", result)
    del restored, model, optimizer, scaler, checkpoint
    torch.cuda.empty_cache()
    return result, history


def smoke_indices(training, count=1024):
    # Metadata-independent random sample, supplemented with the observed
    # constant-channel record and largest training amplitude for numerical stress.
    maximum, largest_index = -1.0, None
    for start in range(0, len(training["y"]), 256):
        block = np.abs(training["x"][start:start+256])
        flat = int(np.argmax(block))
        value = float(block.reshape(-1)[flat])
        if value > maximum:
            maximum, largest_index = value, start+flat//12000
    checks = json.loads((PREPARED/"record_checks.json").read_text(encoding="utf-8"))
    constant_ids = {r["ecg_id"] for r in checks if r["split"] == "train" and r["constant_leads"]}
    stress = {i for i, row in enumerate(training["rows"]) if row["ecg_id"] in constant_ids}
    stress.add(largest_index)
    remaining = np.array([i for i in range(len(training["y"])) if i not in stress])
    chosen = np.random.default_rng(20261023).choice(remaining, count-len(stress), replace=False)
    indices = np.array(sorted(set(chosen.tolist()) | stress), dtype=np.int64)
    return indices, dict(selection_seed=20261023, sampling_scope="engineering-only random training subset plus numerical stress records",
                         stress_ecg_ids=[training["rows"][i]["ecg_id"] for i in sorted(stress)],
                         largest_training_absolute_standardized_value=maximum,
                         selected_ecg_ids=[training["rows"][i]["ecg_id"] for i in indices])


def execute(mode, device):
    data = load_development()
    epochs = 2 if mode == "smoke" else PROTOCOL["training"]["epoch_budget"]
    seeds = [17] if mode == "smoke" else PROTOCOL["training"]["paired_seeds"]
    if mode == "smoke":
        indices, selection = smoke_indices(data["train"])
    else:
        indices, selection = np.arange(len(data["train"]["y"])), dict(scope="all eligible training records")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = PROJECT/"experiments"/mode/stamp
    directory.mkdir(parents=True, exist_ok=False)
    # Freeze runnable sources with each run, including the exact protocol.
    snapshot = directory/"source"
    snapshot.mkdir()
    for path in list(ROOT.glob("*.py"))+list(ROOT.glob("*.ps1"))+[PROJECT/"report"/"protocol_v1.json", PROJECT/"report"/"lead_configurations.json"]:
        (snapshot/path.name).write_bytes(path.read_bytes())
    manifest = dict(started_on=now(), status="running", mode=mode,
                    formal_experiment=(mode == "train"), epochs=epochs, seeds=seeds,
                    training_records=len(indices), validation_records=len(data["validation"]["y"]),
                    short_run_warning="Engineering scores are developmental values, not formal evidence for the research hypothesis" if mode == "smoke" else None,
                    selection=selection, environment=provenance(), results=[])
    write_json(directory/"manifest.json", manifest)
    all_histories = []
    try:
        for seed in seeds:
            paired = []
            for arm in ["A", "B"]:
                result, history = run_arm(arm, seed, epochs, indices, data,
                                          directory/("seed_"+str(seed)+"_"+arm), device, mode)
                manifest["results"].append(result)
                paired.append((result, history))
                all_histories.extend(history)
                write_json(directory/"manifest.json", manifest)
            if paired[0][0]["initial_state_sha256"] != paired[1][0]["initial_state_sha256"]:
                raise AssertionError("Paired models did not have identical initial state")
            if [e["recording_order_sha256"] for e in paired[0][1]] != [e["recording_order_sha256"] for e in paired[1][1]]:
                raise AssertionError("Paired models did not have identical epoch recording order")
        manifest.update(status="complete", completed_on=now(),
                        paired_initialization_and_order_verified=True)
        if mode == "smoke":
            step_seconds = np.mean([h["training_seconds"]/h["batches"] for h in all_histories])
            val_seconds = np.mean([h["validation_seconds"] for h in all_histories])
            batches = (len(data["train"]["y"])+PROTOCOL["training"]["batch_size"]-1)//PROTOCOL["training"]["batch_size"]
            epoch_estimate = float(step_seconds*batches+val_seconds)
            manifest["planning_estimate"] = dict(full_epoch_including_five_config_validation_seconds=epoch_estimate,
                                                  six_40_epoch_runs_hours=epoch_estimate*40*6/3600,
                                                  limitation="Extrapolated from short subset runs, includes cold-start effects; not measured full-run time")
    except Exception as exc:
        manifest.update(status="failed", failed_on=now(), error=type(exc).__name__+": "+str(exc))
        (directory/"failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(directory/"manifest.json", manifest)
        raise
    write_json(directory/"manifest.json", manifest)
    write_json(ROOT/("latest_"+mode+".json"), dict(directory=str(directory), manifest=str(directory/"manifest.json"), status=manifest["status"]))
    print(json.dumps(dict(status="complete", mode=mode, directory=str(directory),
                          parameters=manifest["results"][0]["parameter_count"],
                          planning_estimate=manifest.get("planning_estimate")), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["check", "smoke", "train"])
    action = parser.parse_args().action
    device = configure()
    if action == "check":
        from checks import run_checks
        run_checks(device)
    else:
        execute(action, device)
