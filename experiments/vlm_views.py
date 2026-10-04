"""VLM backbones through OpenRouter, written in the cache_views.py record format so score.py
scores CNNs and VLMs with the same code.

Per page, 12 calls: raw and level in four counterclockwise turns (np.rot90), the identity view of
crop, jit90, jit80, and a blank page of the same size. The model answers which way the top of the
text points; the four options are shuffled per call with a seed from (page, view, turn), as in
RotBench, so no answer letter carries a fixed meaning. Answers map to "turned k quarter turns
counterclockwise" (k = 0..3), the cache convention. Probabilities come from the first token's top
logprobs when the model supports them; otherwise the answer is one-hot with a symmetric floor
(0.94 / 0.02), under which the log-mean vote is exactly a majority vote.
"crop" holds only its identity row; the other three rows are uniform and unused for VLMs.
VIEWS=core keeps only raw and level (8 calls per page) for the extra model families; the control views are then
filled with uniform rows and marked "views": "core", so only the classify-first and LevelHead rules are scored.
DETAIL=low sends OpenAI models the 512-pixel image (their high-detail image tokens cost about 13 times more).
Usage: MODEL=qwen/qwen2.5-vl-72b-instruct THREADS=12 [VIEWS=core] [DETAIL=low] python vlm_views.py
"""
import base64, hashlib, io, json, math, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, requests
from PIL import Image

MODEL = os.environ["MODEL"]; THREADS = int(os.environ.get("THREADS", "12")); LIMIT = int(os.environ.get("LIMIT", "0"))
VIEWS = os.environ.get("VIEWS", "all"); DETAIL = os.environ.get("DETAIL")
B = Path(os.environ.get("BENCH", "bench")); OUT = B / "vlm" / f"cache_vlm_{MODEL.replace('/', '__')}.jsonl"
KEY = os.environ["OPENROUTER_API_KEY"]; API = "https://openrouter.ai/api/v1"
OPTIONS = [("up: the text is upright and reads normally", 0),
           ("left: the page is turned a quarter turn counterclockwise, so lines of text run from bottom to top", 1),
           ("down: the page is upside down", 2),
           ("right: the page is turned a quarter turn clockwise, so lines of text run from top to bottom", 3)]
LETTERS = "ABCD"; lock = threading.Lock(); spend = {"usd": 0.0, "calls": 0, "logprobs": 0, "letter": 0, "failed": 0}
LOGPROBS = any(m["id"] == MODEL and "logprobs" in m.get("supported_parameters", []) for m in requests.get(f"{API}/models", timeout=60).json()["data"])
if os.environ.get("NOLOGPROBS") == "1": LOGPROBS = False   # providers that rarely return them: use letters only, so the vote is a clean majority

def prompt(order):
    lines = "\n".join(f"{LETTERS[i]}) {OPTIONS[o][0]}" for i, o in enumerate(order))
    return ("This is a scanned or photographed document page. It may be rotated. Look at the text and decide "
            f"which way the top of the text points in this image.\n{lines}\nReply with only the letter (A, B, C or D), nothing else.")

def b64(img):
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()

def ask(img, text):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 16, "reasoning": {"enabled": False}, "usage": {"include": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(img)}"} | ({"detail": DETAIL} if DETAIL else {})}, {"type": "text", "text": text}]}]}
    if os.environ.get("REASONING") == "1":   # models whose endpoint cannot disable reasoning: let it think, read the final letter
        body.pop("reasoning"); body["max_tokens"] = 2048
    if LOGPROBS: body |= {"logprobs": True, "top_logprobs": 5}  # 5 = the cap of Alibaba's Qwen endpoint; four letters fit
    if LOGPROBS and os.environ.get("REQUIRE") == "1": body["provider"] = {"require_parameters": True}   # route only to providers that return log-probabilities
    err = None
    for attempt in range(6):
        try:
            r = requests.post(f"{API}/chat/completions", json=body, timeout=180, headers={"Authorization": f"Bearer {KEY}"})
            if r.status_code in (408, 429) or r.status_code >= 500: raise RuntimeError(f"http {r.status_code}")
            r.raise_for_status(); d = r.json()
            if "choices" not in d: raise RuntimeError(str(d.get("error", d))[:200])
            return d
        except Exception as e:  # noqa: BLE001  transient network/provider errors: back off and retry
            err = e; time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"gave up: {err}")

def probs(img, tag):
    order = list(range(4)); random.Random(hashlib.md5(tag.encode()).hexdigest()).shuffle(order)
    d = ask(img, prompt(order)); ch = d["choices"][0]; text = (ch["message"].get("content") or "").strip()
    with lock: spend["calls"] += 1; spend["usd"] += float((d.get("usage") or {}).get("cost") or 0)
    letter = {}
    for t in (((ch.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []):
        tok = t["token"].strip().strip("(").upper()
        if tok in LETTERS: letter[tok] = letter.get(tok, 0) + math.exp(t["logprob"])
    if letter:
        kind = "logprobs"; p_letter = np.array([letter.get(c, 0.0) for c in LETTERS]) + 1e-4
    elif m := re.fullmatch(r"\(?([ABCD])\)?\.?", text) or re.search(r"\b([ABCD])\)", text):
        kind = "letter"; p_letter = np.full(4, 0.02); p_letter[LETTERS.index(m.group(1))] = 0.94
    else:
        kind = "failed"; p_letter = np.full(4, 0.25)
    with lock: spend[kind] += 1
    p = np.zeros(4)
    for i, o in enumerate(order): p[OPTIONS[o][1]] = p_letter[i]
    return (p / p.sum()).round(6).tolist()

def turn(img, k):  # np.rot90 convention: k counterclockwise quarter turns
    return Image.fromarray(np.ascontiguousarray(np.rot90(np.asarray(img), k)))

def page(m):
    v = Path(m["views"]); load = lambda n: Image.open(v / f"{n}.jpg").convert("RGB"); tag = f"{m['dataset']}|{m['id']}"
    rec = {k: m[k] for k in ("dataset", "split", "id", "path", "label", "skew")}; t = time.perf_counter()
    for name in ("raw", "level"):
        img = load(name); rec[name] = [probs(turn(img, k), f"{tag}|{name}|{k}") for k in range(4)]
    if VIEWS == "core":
        rec |= {"crop": [[0.25] * 4] * 4, "jit": [[0.25] * 4] * 2, "blank": [0.25] * 4, "views": "core"}
    else:
        rec["crop"] = [probs(load("crop"), f"{tag}|crop")] + [[0.25] * 4] * 3
        rec["jit"] = [probs(load(n), f"{tag}|{n}") for n in ("jit90", "jit80")]
        rec["blank"] = probs(load("blank"), f"{tag}|blank")
    rec["t_page"] = time.perf_counter() - t
    return rec

def safe(m):  # one failed page must not stop the run; a rerun picks it up
    try: return page(m)
    except Exception as e:  # noqa: BLE001
        print("PAGE_FAILED", m["dataset"], m["id"], str(e)[:200], flush=True); return None

if __name__ == "__main__":
    subset = json.load(open(B / "vlm" / "subset.json")); skews = json.load(open(B / "skew.json"))
    for m in subset: s = skews[m["path"]]; m["skew"] = s if abs(s) >= 1 else 0.0
    if LIMIT: subset = subset[::max(1, len(subset) // LIMIT)][:LIMIT]
    key = lambda m: (m["dataset"], m["split"], m["id"])   # ORB ids repeat across splits
    done = {key(json.loads(l)) for l in open(OUT)} if OUT.exists() else set()
    todo = [m for m in subset if key(m) not in done]; print(MODEL, "logprobs", LOGPROBS, "todo", len(todo), flush=True)
    with open(OUT, "a") as out, ThreadPoolExecutor(THREADS) as pool:
        for n, rec in enumerate(pool.map(safe, todo)):
            if rec is None: continue
            out.write(json.dumps(rec) + "\n"); out.flush()
            if n % 25 == 0: print(MODEL, n, len(todo), {k: round(v, 3) for k, v in spend.items()}, flush=True)
    print("VLM_DONE", MODEL, spend, flush=True)
