"""Plan/download official training and validation waveforms, with resumable hashes.

No test-waveform selection and no model code. Run `plan` before `fetch`.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from hashlib import sha256
import argparse
import json
import threading
import time

from data_common import (DATA, ORIGINALS, REVIEW, SOURCE, development_metadata,
                         hash_file, load_download_index, load_plan, write_json)

LOCAL = threading.local()


def now():
    return datetime.now(timezone.utc).isoformat()


def make_plan():
    plan = development_metadata()
    plan["created_on"] = now()
    path = DATA / "development_plan.json"
    if path.exists():
        previous = load_plan()
        if previous["records"] != plan["records"] or previous["metadata_exclusions"] != plan["metadata_exclusions"]:
            raise ValueError("Existing development cohort differs")
        plan = previous
    else:
        write_json(path, plan)
    print(json.dumps(dict(records=dict(Counter(r["split"] for r in plan["records"])),
                          metadata_exclusions=dict(Counter(r["split"] for r in plan["metadata_exclusions"])),
                          test_selected=False), indent=2))


def retrieve(job, previous, pilot):
    import requests
    row, suffix = job
    relative = row["filename_lr"] + suffix
    dest = ORIGINALS / relative
    entry = dict(ecg_id=row["ecg_id"], split=row["split"], relative_path=relative,
                 source_url=SOURCE + relative, version="1.0.3")
    try:
        known = previous.get(relative)
        if dest.exists() and known and known.get("status") == "archived" and hash_file(dest) == known["sha256"]:
            return dict(known, acquisition="verified_existing", verified_on=now())
        pilot_source = REVIEW / "waveform_pilot" / "originals" / relative
        pilot_known = pilot.get("originals/" + relative)
        if pilot_source.exists() and pilot_known and hash_file(pilot_source) == pilot_known["sha256"]:
            raw = pilot_source.read_bytes()
            acquisition = "copied_verified_pilot_original"
        else:
            if not hasattr(LOCAL, "session"):
                LOCAL.session = requests.Session()
                LOCAL.session.headers["User-Agent"] = "NTUDL-PTBXL-development-data/1.0"
            for attempt in range(4):
                try:
                    response = LOCAL.session.get(entry["source_url"], timeout=(10, 45))
                    response.raise_for_status()
                    raw = response.content
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            acquisition = "official_s3_download"
        if suffix == ".hea":
            if not raw.decode("ascii").splitlines()[0].startswith(Path(relative).stem + " "):
                raise ValueError("Unexpected record header")
        elif len(raw) == 0:
            raise ValueError("Empty waveform file")
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_suffix(dest.suffix + ".part")
        temp.write_bytes(raw)
        temp.replace(dest)
        entry.update(status="archived", bytes=len(raw), sha256=sha256(raw).hexdigest(),
                     acquisition=acquisition, retrieved_on=now())
    except Exception as exc:
        entry.update(status="failed", error=type(exc).__name__ + ": " + str(exc)[:250], attempted_on=now())
    return entry


def fetch(workers):
    plan = load_plan()
    previous = load_download_index()
    pilot = json.loads((REVIEW / "waveform_pilot" / "sources.json").read_text(encoding="utf-8"))
    pilot = {e["relative_path"].replace("\\", "/"): e for e in pilot["files"] if e["status"] == "archived"}
    jobs = [(r, suffix) for r in plan["records"] for suffix in [".hea", ".dat"]]
    counters = Counter()
    started = time.monotonic()
    # Append only in this main thread; each line is independently usable on resume.
    with (DATA / "download_manifest.jsonl").open("a", encoding="utf-8") as log:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(retrieve, job, previous, pilot) for job in jobs]
            for i, future in enumerate(as_completed(pending), 1):
                entry = future.result()
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
                counters[entry["status"]] += 1
                if i % 250 == 0 or i == len(jobs):
                    log.flush()
                    progress = dict(completed=i, total=len(jobs), counts=dict(counters), elapsed_seconds=round(time.monotonic()-started, 1))
                    write_json(DATA / "download_progress.json", progress)
                    print(json.dumps(progress), flush=True)
    summary = dict(completed_on=now(), total_files=len(jobs), counts=dict(counters),
                   official_project_url="https://physionet.org/content/ptb-xl/1.0.3/",
                   official_s3_prefix=SOURCE, test_waveforms_downloaded=False,
                   elapsed_seconds=round(time.monotonic()-started, 1))
    write_json(DATA / "download_summary.json", summary)
    if counters["failed"]:
        raise SystemExit("Download incomplete; retry fetch before processing. Network errors are not cohort exclusions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["plan", "fetch"])
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    make_plan() if args.action == "plan" else fetch(args.workers)
