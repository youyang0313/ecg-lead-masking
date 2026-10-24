"""Audit completed main training and summarize validation results only.

No training or test-data loading. Run using the existing PyTorch 2.6 environment.
Validation summaries describe checkpoint selection data, not independent tests.
"""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import argparse
import json
import math

from runtime import (CLASSES, CONFIG, PREPARED, PROJECT, PROTOCOL, ROOT, hash_file,
                     now, parameter_hash, state_hash, write_json)
from metrics import multilabel_metrics, selection_score
from model import ECGClassifier
import numpy as np
import torch


def close(a, b):
    if a is None or b is None or not math.isclose(a, b, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("Numerical mismatch: "+str((a, b)))


def verify(directory):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or not manifest["formal_experiment"] or manifest["mode"] != "train":
        raise ValueError("The full main run is not complete")
    seeds = PROTOCOL["training"]["paired_seeds"]
    epochs = PROTOCOL["training"]["epoch_budget"]
    ntrain = len(json.loads((PREPARED/"train.records.json").read_text(encoding="utf-8")))
    validation_rows = json.loads((PREPARED/"validation.records.json").read_text(encoding="utf-8"))
    validation_y = np.load(PREPARED/"validation.labels.npy", allow_pickle=False)
    if manifest["seeds"] != seeds or manifest["epochs"] != epochs or manifest["training_records"] != ntrain:
        raise AssertionError("Run budget/cohort differs from the protocol")
    expected_keys = {(s, a) for s in seeds for a in ["A", "B"]}
    if {(r["seed"], r["arm"]) for r in manifest["results"]} != expected_keys or len(manifest["results"]) != 6:
        raise AssertionError("Incomplete or duplicated paired runs")
    for name, expected in manifest["environment"]["source_sha256"].items():
        if hash_file(ROOT/name) != expected or hash_file(directory/"source"/name) != expected:
            raise AssertionError("Executed source changed: "+name)
    inputs = manifest["environment"]["input_and_protocol_sha256"]
    for path, key in [(PROJECT/"report"/"protocol_v1.json", "protocol_v1.json"),
                      (PROJECT/"report"/"lead_configurations.json", "lead_configurations.json"),
                      (PREPARED/"READY.json", "prepared_READY.json")]:
        if hash_file(path) != inputs[key]:
            raise AssertionError("Run input changed: "+key)
    ready = json.loads((PREPARED/"READY.json").read_text(encoding="utf-8"))
    for entry in ready["outputs"]:
        if hash_file(PREPARED/entry["file"]) != entry["sha256"]:
            raise AssertionError("Prepared data changed during training: "+entry["file"])
    configurations = [c["id"] for c in CONFIG["configurations"]]
    histories, metrics_by_run, model_records, selected_models = {}, {}, [], []
    masks = {"A": np.zeros(13, dtype=np.int64), "B": np.zeros(13, dtype=np.int64)}
    total_optimizer_steps = total_overflow = 0
    all_training_seconds, all_validation_seconds = [], []
    initial_states = {}
    batch_size = PROTOCOL["training"]["batch_size"]
    batches = (ntrain+batch_size-1)//batch_size
    for result in manifest["results"]:
        seed, arm = result["seed"], result["arm"]
        arm_directory = Path(result["checkpoint_path"]).parent
        history = json.loads((arm_directory/"history.json").read_text(encoding="utf-8"))
        if len(history) != epochs or result["completed_epochs"] != epochs or result["training_records"] != ntrain:
            raise AssertionError("Incomplete epoch budget")
        histories[(seed, arm)] = history
        for i, h in enumerate(history):
            if h["epoch"] != i+1 or h["training_records"] != ntrain or h["batches"] != batches:
                raise AssertionError("Epoch/sample/batch accounting mismatch")
            hi, lo = PROTOCOL["training"]["learning_rate"], PROTOCOL["training"]["minimum_learning_rate"]
            close(h["learning_rate"], lo+(hi-lo)*(1+math.cos(math.pi*i/(epochs-1)))/2)
            order = np.random.default_rng(np.random.SeedSequence([seed, i, 20261023])).permutation(np.arange(ntrain))
            if sha256(order.astype("<i8").tobytes()).hexdigest() != h["recording_order_sha256"]:
                raise AssertionError("Recorded order differs from declared seed/epoch")
            if not all(np.isfinite(h[k]) for k in ["mean_bce", "finite_gradient_norm_min", "finite_gradient_norm_max", "training_seconds", "validation_seconds"]):
                raise AssertionError("Non-finite training diagnostic")
            if h["finite_gradient_optimizer_steps"]+h["overflow_skipped_steps"] != batches:
                raise AssertionError("Optimizer/overflow accounting mismatch")
            histogram = {int(k): v for k, v in h["retained_lead_count_histogram"].items()}
            if sum(histogram.values()) != ntrain or not set(histogram) <= set(range(1, 13)):
                raise AssertionError("Mask/sample accounting mismatch")
            if arm == "A" and histogram != {12: ntrain}:
                raise AssertionError("Complete-input arm saw partial inputs")
            for k, v in histogram.items():
                masks[arm][k] += v
            total_optimizer_steps += h["finite_gradient_optimizer_steps"]
            total_overflow += h["overflow_skipped_steps"]
            all_training_seconds.append(h["training_seconds"])
            all_validation_seconds.append(h["validation_seconds"])
            close(selection_score(h["validation_metrics"], PROTOCOL["selection"]["weights"]), h["selection_score"])
        best_index = max(range(epochs), key=lambda i: history[i]["selection_score"])
        if result["selected_epoch"] != best_index+1:
            raise AssertionError("Checkpoint selection or earliest-tie rule mismatch")
        if hash_file(arm_directory/"best.pt") != result["checkpoint_sha256"]:
            raise AssertionError("Checkpoint bytes changed")
        checkpoint = torch.load(arm_directory/"best.pt", map_location="cpu", weights_only=True)
        if not checkpoint["formal_experiment"] or checkpoint["mode"] != "train" or checkpoint["epoch"] != best_index+1:
            raise AssertionError("Wrong checkpoint mode/epoch")
        torch.manual_seed(seed)
        model = ECGClassifier(PROTOCOL)
        if state_hash(model) != result["initial_state_sha256"]:
            raise AssertionError("Model did not start from the declared fresh initialization")
        initial_states[(seed, arm)] = result["initial_state_sha256"]
        model.load_state_dict(checkpoint["model_state"])
        if state_hash(model) != checkpoint["selected_state_sha256"] or parameter_hash(model) == result["initial_parameters_sha256"]:
            raise AssertionError("Checkpoint state/weight update mismatch")
        if not result["checkpoint_readback_predictions_exact"] or not result["parameters_updated"]:
            raise AssertionError("Trainer checkpoint readback/weight check failed")
        selected_steps = sum(h["finite_gradient_optimizer_steps"] for h in history[:best_index+1])
        if any(int(s["step"].item()) != selected_steps for s in checkpoint["optimizer_state"]["state"].values()):
            raise AssertionError("Checkpoint optimizer counter differs from its selected epoch")
        with np.load(arm_directory/"validation_predictions.npz", allow_pickle=False) as saved:
            probabilities = saved["probabilities"]
            if probabilities.shape != (5, len(validation_y), 5) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
                raise AssertionError("Validation prediction shape/range invalid")
            if not np.array_equal(saved["targets"], validation_y) or not np.array_equal(saved["ecg_ids"], [r["ecg_id"] for r in validation_rows]):
                raise AssertionError("Validation labels/IDs changed order")
            if saved["configuration_ids"].tolist() != configurations:
                raise AssertionError("Prediction configuration order changed")
            metrics = {c: multilabel_metrics(validation_y, probabilities[i], CLASSES) for i, c in enumerate(configurations)}
        for c in configurations:
            for key in ["macro_auroc", "macro_average_precision"]:
                close(metrics[c][key], history[best_index]["validation_metrics"][c][key])
            for label in CLASSES:
                for key in ["auroc", "average_precision"]:
                    close(metrics[c]["per_label"][label][key], checkpoint["validation_metrics"][c]["per_label"][label][key])
        close(selection_score(metrics, PROTOCOL["selection"]["weights"]), result["selected_validation_score"])
        metrics_by_run[(seed, arm)] = metrics
        model_records.append(dict(seed=seed, arm=arm, selected_epoch=best_index+1,
                                  checkpoint_sha256=result["checkpoint_sha256"], validation_metrics=metrics))
        selected_models.append(dict(seed=seed, arm=arm, selected_epoch=best_index+1,
                                    checkpoint_path=str(arm_directory/"best.pt"),
                                    checkpoint_sha256=result["checkpoint_sha256"],
                                    selected_state_sha256=checkpoint["selected_state_sha256"]))
        del model, checkpoint
    for seed in seeds:
        if initial_states[(seed, "A")] != initial_states[(seed, "B")]:
            raise AssertionError("Paired initialization differs")
        if [h["recording_order_sha256"] for h in histories[(seed, "A")]] != [h["recording_order_sha256"] for h in histories[(seed, "B")]]:
            raise AssertionError("Paired epoch order differs")
    if len({initial_states[(s, "A")] for s in seeds}) != len(seeds):
        raise AssertionError("Distinct seeds shared initialization")
    complete_fraction = float(masks["B"][12]/masks["B"].sum())
    if not .49 < complete_fraction < .51:
        raise AssertionError("Observed complete-input mixture inconsistent with protocol")
    validation_summary = {}
    for configuration in configurations:
        a = np.array([metrics_by_run[(s, "A")][configuration]["macro_auroc"] for s in seeds])
        b = np.array([metrics_by_run[(s, "B")][configuration]["macro_auroc"] for s in seeds])
        validation_summary[configuration] = dict(A_mean=float(a.mean()), A_seed_sd=float(a.std(ddof=1)),
                                                  B_mean=float(b.mean()), B_seed_sd=float(b.std(ddof=1)),
                                                  mean_paired_B_minus_A=float((b-a).mean()))
    paired_gaps = [float(np.mean([metrics_by_run[(s, "B")][c]["macro_auroc"]-metrics_by_run[(s, "A")][c]["macro_auroc"]
                                 for c in configurations if c != "S12"])) for s in seeds]
    output = directory/"analysis"
    output.mkdir(exist_ok=True)
    discarded = manifest.get("discarded_attempt")
    extra_steps = extra_skips = 0
    if discarded:
        recovery_source = directory/"recovery"/"recover_main.py"
        if hash_file(recovery_source) != manifest["recovery_driver_sha256"]:
            raise AssertionError("Recovery driver snapshot changed")
        extra_history = json.loads((Path(discarded["directory"])/"history.json.tmp").read_text(encoding="utf-8"))
        if len(extra_history) != 40 or not discarded["rerun_selected_predictions_exact"]:
            raise AssertionError("Recovery accounting/reproduction incomplete")
        extra_steps = sum(h["finite_gradient_optimizer_steps"] for h in extra_history)
        extra_skips = sum(h["overflow_skipped_steps"] for h in extra_history)
    verification = dict(checked_on=now(), status="passed", runs=6, epochs_per_run=epochs,
                        completed_epochs=6*epochs, training_records=ntrain, validation_records=len(validation_y),
                        finite_gradient_optimizer_steps=total_optimizer_steps, overflow_skipped_steps=total_overflow,
                        source_and_input_hashes_verified=True, checkpoint_and_optimizer_states_verified=True,
                        fresh_initialization_and_paired_epoch_order_verified=True,
                        all_saved_validation_metrics_recomputed=True, mask_complete_fraction_B=complete_fraction,
                        recovery_source_verified=bool(discarded),
                        discarded_attempt_optimizer_steps=extra_steps,
                        discarded_attempt_overflow_skips=extra_skips,
                        verification_source_sha256=hash_file(Path(__file__)),
                        retained_count_histograms={a: {str(k): int(v) for k,v in enumerate(values) if v} for a, values in masks.items()},
                        test_data_used=False)
    write_json(output/"training_verification.json", verification)
    (output/"verify_main.py").write_bytes(Path(__file__).read_bytes())
    summary = dict(created_on=now(), dataset_split="validation_fold_9", independent_test_result=False,
                   interpretation="Checkpoint-selected validation performance only; no hypothesis test or patient-bootstrap interval. Test data remain unused.",
                   paired_seeds=seeds, macro_auroc_by_configuration=validation_summary,
                   paired_validation_partial_gaps=paired_gaps,
                   mean_validation_partial_gap=float(np.mean(paired_gaps)),
                   validation_partial_gap_seed_sd=float(np.std(paired_gaps, ddof=1)),
                   wall_seconds=(datetime.fromisoformat(manifest["completed_on"])-datetime.fromisoformat(manifest["started_on"])).total_seconds(),
                   training_seconds_sum=float(sum(all_training_seconds)), validation_seconds_sum=float(sum(all_validation_seconds)),
                   discarded_attempt=manifest.get("discarded_attempt"),
                   actual_epochs_including_discarded_attempt=manifest.get("actual_epochs_including_discarded_attempt", 6*epochs),
                   full_training_epoch_seconds_range=[min(all_training_seconds), max(all_training_seconds)],
                   peak_allocated_MiB=max(r["train_validation_peak_allocated_bytes"] for r in manifest["results"])/1024**2,
                   peak_reserved_MiB=max(r["train_validation_peak_reserved_bytes"] for r in manifest["results"])/1024**2,
                   model_records=model_records)
    write_json(output/"validation_summary.json", summary)
    write_json(output/"selected_models.json", dict(fixed_on=now(),
                scope="Validation-selected models fixed before test diagnostic annotations or waveforms are used; local record, not external preregistration",
                protocol_sha256=inputs["protocol_v1.json"],
                configuration_sha256=inputs["lead_configurations.json"],
                architecture_source_sha256=manifest["environment"]["source_sha256"]["model.py"],
                scaler_path=str(PREPARED/"scaler.json"), scaler_sha256=hash_file(PREPARED/"scaler.json"),
                classes=CLASSES, expected_test_fold=10, test_eligibility_not_yet_computed=True,
                input_configurations=configurations, configuration_specific_finetuning=False,
                seed_prediction_ensemble=False, checkpoints=selected_models))
    write_json(ROOT/"latest_main_verification.json", dict(directory=str(directory), verification=str(output/"training_verification.json"),
                                                         validation_summary=str(output/"validation_summary.json"), status="passed"))
    print(json.dumps(verification, indent=2))
    print(json.dumps({k: summary[k] for k in ["macro_auroc_by_configuration", "mean_validation_partial_gap", "wall_seconds", "peak_allocated_MiB"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path)
    directory = parser.parse_args().directory
    if directory is None:
        directory = Path(json.loads((ROOT/"latest_train.json").read_text(encoding="utf-8"))["directory"])
    verify(directory)
