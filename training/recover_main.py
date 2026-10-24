"""Recover one Windows log-save interruption without changing training rules.

Preserve four completed runs and the failed attempt. Rerun seed-43 A from its
fresh initialization in a separate directory, then run seed-43 B. Only JSON
persistence is wrapped in bounded PermissionError retries.
"""
from pathlib import Path
import argparse
import json
import time

import runtime
import experiment
import numpy as np


def recover(directory):
    manifest_path = directory/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "failed" or {(r["seed"], r["arm"]) for r in manifest["results"]} != {(17,"A"),(17,"B"),(29,"A"),(29,"B")}:
        raise ValueError("Recovery is restricted to the recorded four-run-complete failure")
    if "PermissionError" not in manifest["error"]:
        raise ValueError("Unexpected failure; do not silently recover")
    for name, expected in manifest["environment"]["source_sha256"].items():
        if runtime.hash_file(runtime.ROOT/name) != expected:
            raise ValueError("Training source changed: "+name)
    recovery = directory/"recovery"
    recovery.mkdir(exist_ok=False)
    (recovery/"manifest_before_recovery.json").write_bytes(manifest_path.read_bytes())
    (recovery/"recover_main.py").write_bytes(Path(__file__).read_bytes())
    old_history_path = directory/"seed_43_A"/"history.json.tmp"
    discarded_history = json.loads(old_history_path.read_text(encoding="utf-8"))
    if len(discarded_history) != 40 or discarded_history[-1]["epoch"] != 40:
        raise ValueError("Unexpected interrupted-attempt history")
    original_write = runtime.write_json
    io_retries = []
    def resilient_write(path, payload):
        for attempt in range(10):
            try:
                return original_write(path, payload)
            except PermissionError:
                io_retries.append(dict(path=str(path), attempt=attempt+1, recorded_on=runtime.now()))
                if attempt == 9:
                    raise
                time.sleep(.1*(attempt+1))
    # Both modules imported the original function; replace only the persistence
    # binding, leaving model, gradients, masking, seeds and selection untouched.
    runtime.write_json = resilient_write
    experiment.write_json = resilient_write
    device = runtime.configure()
    data = runtime.load_development()
    indices = np.arange(len(data["train"]["y"]))
    manifest.update(status="recovering", recovery_started_on=runtime.now(),
                    recovery_reason="Windows PermissionError replacing seed_43_A/history.json after epoch 40",
                    discarded_attempt=dict(directory=str(directory/"seed_43_A"),
                                            complete_temporary_history_epochs=40,
                                            training_seconds=sum(h["training_seconds"] for h in discarded_history),
                                            validation_seconds=sum(h["validation_seconds"] for h in discarded_history),
                                            excluded_from_selected_six_models=True),
                    recovery_driver_sha256=runtime.hash_file(Path(__file__)))
    resilient_write(manifest_path, manifest)
    paired = []
    for arm, name in [("A", "seed_43_A_retry"), ("B", "seed_43_B")]:
        result, history = experiment.run_arm(arm, 43, 40, indices, data, directory/name, device, "train")
        result["rerun_after_io_failure"] = arm == "A"
        if arm == "A":
            with np.load(directory/"seed_43_A"/"validation_predictions.npz", allow_pickle=False) as old:
                with np.load(directory/name/"validation_predictions.npz", allow_pickle=False) as new:
                    repeated_exactly = bool(np.array_equal(old["probabilities"], new["probabilities"]))
            if not repeated_exactly:
                raise AssertionError("Same-source recovery changed selected validation predictions")
            manifest["discarded_attempt"]["rerun_selected_predictions_exact"] = repeated_exactly
        manifest["results"].append(result)
        paired.append((result, history))
        resilient_write(manifest_path, manifest)
    if paired[0][0]["initial_state_sha256"] != paired[1][0]["initial_state_sha256"] or [h["recording_order_sha256"] for h in paired[0][1]] != [h["recording_order_sha256"] for h in paired[1][1]]:
        raise AssertionError("Recovered pair initialization/order mismatch")
    manifest.update(status="complete", completed_on=runtime.now(),
                    paired_initialization_and_order_verified=True,
                    scientific_design_amendments=[], json_io_retries=io_retries,
                    selected_model_epochs=240, actual_epochs_including_discarded_attempt=280)
    resilient_write(manifest_path, manifest)
    resilient_write(runtime.ROOT/"latest_train.json", dict(directory=str(directory), manifest=str(manifest_path), status="complete"))
    print(json.dumps(dict(status="complete", directory=str(directory), completed_models=6,
                          discarded_attempt_predictions_reproduced=True, json_io_retries=len(io_retries)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    recover(parser.parse_args().directory)
