"""A local open VLM as orientation backbone, with exact probabilities for the four answer letters.

Same prompt, per-call option shuffle and image bytes as vlm_views.py (VIEWS=core: raw and level views in four
counterclockwise turns), but the model runs on our GPU, so the four letters' probabilities are read from the full
next-token distribution instead of a provider's top-5 log-probabilities.
Default model: microsoft/Phi-4-multimodal-instruct (Microsoft, 5.6B), native to transformers via its refs/pr/70 weights.
Usage: MODEL=microsoft/Phi-4-multimodal-instruct REVISION=refs/pr/70 [LIMIT=20] python local_vlm.py
"""
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

MODEL = os.environ.get("MODEL", "microsoft/Phi-4-multimodal-instruct"); REV = os.environ.get("REVISION") or None; LIMIT = int(os.environ.get("LIMIT", "0")); CHUNK = int(os.environ.get("CHUNK", "2"))
B = Path(os.environ.get("BENCH", "bench")); OUT = B / "vlm" / f"cache_vlm_{MODEL.replace('/', '__')}.jsonl"
OPTIONS = [("up: the text is upright and reads normally", 0),
           ("left: the page is turned a quarter turn counterclockwise, so lines of text run from bottom to top", 1),
           ("down: the page is upside down", 2),
           ("right: the page is turned a quarter turn clockwise, so lines of text run from top to bottom", 3)]
LETTERS = "ABCD"

def prompt(order):   # identical to vlm_views.py
    lines = "\n".join(f"{LETTERS[i]}) {OPTIONS[o][0]}" for i, o in enumerate(order))
    return ("This is a scanned or photographed document page. It may be rotated. Look at the text and decide "
            f"which way the top of the text points in this image.\n{lines}\nReply with only the letter (A, B, C or D), nothing else.")

proc = AutoProcessor.from_pretrained(MODEL, revision=REV)
if hasattr(proc, "image_processor") and "longest_edge" in (getattr(proc.image_processor, "size", None) or {}):
    proc.image_processor.size = {"longest_edge": 2 * 364}   # 2 x 2 tiles + a global view: legible text at a quarter of the default cost
try: model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16)
except ValueError: model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16)   # Phi-4-multimodal registers as a causal LM
model = model.to("cuda").eval()
letter_ids = [proc.tokenizer.encode(c, add_special_tokens=False)[-1] for c in LETTERS]
assert len(set(letter_ids)) == 4, letter_ids

@torch.inference_mode()
def probs(images, tags):
    orders = []
    for t in tags:
        o = list(range(4)); random.Random(hashlib.md5(t.encode()).hexdigest()).shuffle(o); orders.append(o)
    if getattr(proc, "chat_template", None):
        texts = [proc.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(o)}]}], add_generation_prompt=True) for o in orders]
    else:   # Phi-4-multimodal's native processor ships no template; this is its documented chat format
        texts = [f"<|user|>{getattr(proc, 'image_token', '<|image|>')}{prompt(o)}<|end|><|assistant|>" for o in orders]
    logits = []
    for i in range(0, len(texts), CHUNK):   # small chunks: high-resolution crops make long sequences
        batch = proc(text=texts[i:i + CHUNK], images=[[im] for im in images[i:i + CHUNK]], return_tensors="pt", padding=True).to("cuda")
        logits.append(model(**batch, logits_to_keep=1).logits[:, -1, :].float())   # only the next-token logits
    logits = torch.cat(logits)
    p_letter = torch.softmax(logits[:, letter_ids], dim=-1).cpu().numpy()   # exact distribution over the four letters
    out = []
    for o, pl in zip(orders, p_letter):
        p = np.zeros(4)
        for i, k in enumerate(o): p[OPTIONS[k][1]] = pl[i]
        out.append((p / p.sum()).round(6).tolist())
    return out

turn = lambda img, k: Image.fromarray(np.ascontiguousarray(np.rot90(np.asarray(img), k)))   # counterclockwise quarter turns

if __name__ == "__main__":
    proc.tokenizer.padding_side = "left"   # the next-token logits sit at the last position for every row
    subset = json.load(open(B / "vlm" / "subset.json")); skews = json.load(open(B / "skew.json"))
    for m in subset: s = skews[m["path"]]; m["skew"] = s if abs(s) >= 1 else 0.0
    if LIMIT: subset = subset[::max(1, len(subset) // LIMIT)][:LIMIT]
    key = lambda m: (m["dataset"], m["split"], m["id"])
    done = {key(json.loads(l)) for l in open(OUT)} if OUT.exists() else set()
    todo = [m for m in subset if key(m) not in done]; print(MODEL, "todo", len(todo), flush=True); t0 = time.time()
    with open(OUT, "a") as out:
        for n, m in enumerate(todo):
            v = Path(m["views"]); tag = f"{m['dataset']}|{m['id']}"
            imgs, tags = [], []
            for name in ("raw", "level"):
                img = Image.open(v / f"{name}.jpg").convert("RGB")
                imgs += [turn(img, k) for k in range(4)]; tags += [f"{tag}|{name}|{k}" for k in range(4)]
            p = probs(imgs, tags)
            rec = {k: m[k] for k in ("dataset", "split", "id", "path", "label", "skew")} | {"raw": p[:4], "level": p[4:],
                   "crop": [[0.25] * 4] * 4, "jit": [[0.25] * 4] * 2, "blank": [0.25] * 4, "views": "core"}
            out.write(json.dumps(rec) + "\n"); out.flush()
            if n % 100 == 0: print(MODEL, n, len(todo), f"{(time.time() - t0) / (n + 1):.2f}s/page", flush=True)
    print("LOCAL_VLM_DONE", MODEL)
