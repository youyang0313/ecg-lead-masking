"""Read-only verification of archived smoke artifacts; no torch or training."""
from hashlib import sha256
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    latest = json.loads((ROOT/"latest_smoke.json").read_text(encoding="utf-8"))
    directory = Path(latest["directory"])
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["formal_experiment"]:
        raise AssertionError("Expected completed engineering-only run")
    source_ok = all(digest(ROOT/name) == value and digest(directory/"source"/name) == value
                    for name, value in manifest["environment"]["source_sha256"].items())
    if not source_ok:
        raise AssertionError("Current/snapshotted training source differs from the executed source")
    previous_directories = sorted(p for p in directory.parent.iterdir()
                                  if p.is_dir() and p.name < directory.name and (p/"manifest.json").exists())
    previous = previous_directories[-1] if previous_directories else None
    order_lists, histories, repeated = [], [], []
    peak = 0
    for result in manifest["results"]:
        arm_directory = directory/("seed_"+str(result["seed"])+"_"+result["arm"])
        history = json.loads((arm_directory/"history.json").read_text(encoding="utf-8"))
        histories.extend(history)
        order_lists.append([e["recording_order_sha256"] for e in history])
        if result["initial_parameters_sha256"] == result["trained_parameters_sha256"]:
            raise AssertionError("Trainable weights did not change")
        if not result["checkpoint_readback_predictions_exact"] or digest(arm_directory/"best.pt") != result["checkpoint_sha256"]:
            raise AssertionError("Checkpoint/prediction verification missing or checkpoint hash changed")
        peak = max(peak, result["train_validation_peak_allocated_bytes"])
        with np.load(arm_directory/"validation_predictions.npz", allow_pickle=False) as predictions:
            if predictions["probabilities"].shape != (5, 2146, 5):
                raise AssertionError("Unexpected validation prediction shape")
            if previous:
                with np.load(previous/arm_directory.name/"validation_predictions.npz", allow_pickle=False) as old:
                    repeated.append(bool(np.array_equal(predictions["probabilities"], old["probabilities"])))
    if len(order_lists) != 2 or order_lists[0] != order_lists[1]:
        raise AssertionError("Paired epoch recording orders differ")
    if manifest["results"][0]["initial_state_sha256"] != manifest["results"][1]["initial_state_sha256"]:
        raise AssertionError("Paired initial states differ")
    if repeated and not all(repeated):
        raise AssertionError("Repeat engineering runs changed validation predictions")
    if any(e["overflow_skipped_steps"] for e in histories):
        raise AssertionError("This recorded engineering result claims zero overflow steps")
    verification = dict(status="passed", current_and_archived_source_matches_run=source_ok,
                        trainable_weights_updated_and_checkpoint_predictions_exact=True,
                        paired_initialization_and_recording_order_identical=True,
                        repeat_run_predictions_exact=all(repeated) if repeated else None,
                        finite_gradient_optimizer_steps=sum(e["finite_gradient_optimizer_steps"] for e in histories),
                        overflow_skipped_steps=sum(e["overflow_skipped_steps"] for e in histories),
                        peak_allocated_MiB=peak/1024**2,
                        train_epoch_seconds_range=[min(e["training_seconds"] for e in histories),
                                                   max(e["training_seconds"] for e in histories)],
                        formal_experiments_completed=0, test_data_used=False, run_directory=str(directory))
    (ROOT/"engineering_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
