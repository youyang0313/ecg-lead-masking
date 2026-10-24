"""Archive a fixed allowlist of PTB-XL metadata; never fetch waveform files."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen
from time import monotonic
import gzip
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent
DEST = ROOT / "metadata"
FILES = ["ptbxl_database.csv", "scp_statements.csv", "example_physionet.py", "ptbxl_v103_changelog.txt", "LICENSE.txt"]
BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
DEST.mkdir(parents=True, exist_ok=True)

def fetch(name):
    # Official project page also exposes this public bucket; use it for the
    # larger CSV because the main website throttles individual transfers.
    url = ("https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/" if name=="ptbxl_database.csv" else BASE) + name
    path = DEST / name
    if path.exists():
        raw = path.read_bytes()
    else:
        with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0", "Accept-Encoding":"gzip"}), timeout=25) as response:
            chunks=[]
            received=0
            started=monotonic()
            last_report=started
            while True:
                chunk=response.read(8192)
                if not chunk:
                    break
                chunks.append(chunk)
                received+=len(chunk)
                if monotonic()-last_report>8:
                    print(json.dumps({"name":name,"transport_bytes_received":received}),flush=True)
                    last_report=monotonic()
                if monotonic()-started>100:
                    raise TimeoutError("Metadata transfer exceeded 100-second review budget")
            raw = b"".join(chunks)
            if response.headers.get("Content-Encoding","").lower()=="gzip":
                raw=gzip.decompress(raw)
        text = raw.decode("utf-8-sig")
        if "<html" in text[:1000].lower() or "checking your browser" in text.lower():
            raise ValueError("Web page returned for metadata: " + name)
        if name == "ptbxl_database.csv" and "scp_codes" not in text.splitlines()[0]:
            raise ValueError("Unexpected database CSV header")
        if name == "scp_statements.csv" and "diagnostic_class" not in text.splitlines()[0]:
            raise ValueError("Unexpected statement CSV header")
        path.write_bytes(raw)
    return dict(name=name, source_url=url, dataset_version="1.0.3", retrieved_on_client_date="2026-10-23", bytes=len(raw), sha256=sha256(raw).hexdigest(), local_path=str(path.relative_to(ROOT)))

records = list(ThreadPoolExecutor(max_workers=3).map(fetch, FILES))
(ROOT / "sources.json").write_text(json.dumps({"scope":"Metadata and documentation only; no ECG waveform or header files", "files":records}, indent=2), encoding="utf-8")
for record in records:
    print(json.dumps(record))
