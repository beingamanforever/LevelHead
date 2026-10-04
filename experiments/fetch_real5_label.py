"""Download the real captures to label for the orientation benchmark : every photograph of
Real5-OmniDocBench-Skew and Real5-OmniDocBench-Screen-Photography (both sets in full).
Uses curl (the local Python has no CA bundle); resumable, existing files are skipped.
Usage: python fetch_real5_label.py <out_dir>
"""
import json, subprocess, sys, tempfile, time
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "PaddlePaddle/Real5-OmniDocBench"; SETS = ["Real5-OmniDocBench-Skew", "Real5-OmniDocBench-Screen-Photography"]
OUT = Path(sys.argv[1])

def listing(folder):
    files, url = [], f"https://huggingface.co/api/datasets/{REPO}/tree/main/{folder}"
    while url:
        with tempfile.NamedTemporaryFile() as h:
            body = subprocess.run(["curl", "-sS", "-D", h.name, url], capture_output=True, text=True, check=True).stdout
            head = open(h.name).read()
        files += [f["path"] for f in json.loads(body) if f["type"] == "file" and f["path"].lower().endswith((".png", ".jpg", ".jpeg"))]
        link = next((l for l in head.splitlines() if l.lower().startswith("link:") and 'rel="next"' in l), None)
        url = link.split("<", 1)[1].split(">", 1)[0] if link else None
    return files

VIEW = OUT.parent / (OUT.name + "_view")   # 2,000-pixel JPEG copies for the labelling tool; angles do not depend on scale

def view(dst):
    out = VIEW / dst.parent.name / (dst.stem + ".jpg")
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["sips", "-Z", "2000", "-s", "format", "jpeg", "-s", "formatOptions", "85", str(dst), "--out", str(out)], check=True, capture_output=True)

def fetch(path):
    dst = OUT / path.split("/")[0].replace("Real5-OmniDocBench-", "") / path.split("/")[-1]
    if dst.exists() and dst.stat().st_size > 0: view(dst); return 0
    dst.parent.mkdir(parents=True, exist_ok=True); tmp = dst.with_name(dst.name + ".part")
    for attempt in range(6):   # transient HTTP/2 stream errors happen; retry each file on its own, over HTTP/1.1
        r = subprocess.run(["curl", "-sSL", "--http1.1", "--retry", "3", "-o", str(tmp), f"https://huggingface.co/datasets/{REPO}/resolve/main/{quote(path)}"], capture_output=True)
        if r.returncode == 0: tmp.rename(dst); view(dst); return 1
        time.sleep(5 * (attempt + 1))
    print("FAILED", path, flush=True); return -1

if __name__ == "__main__":
    todo = {s: listing(s) for s in SETS}; print({s: len(v) for s, v in todo.items()}, flush=True)
    allf = [p for v in todo.values() for p in v]
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(fetch, allf)):
            if i % 250 == 0: print("fetched", i, "of", len(allf), flush=True)
    json.dump(todo, open(OUT / "sources.json", "w"), indent=1); print("FETCH_DONE", len(allf))
