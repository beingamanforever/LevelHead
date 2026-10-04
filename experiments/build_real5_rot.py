"""Real5-Rot: real photographs turned by known angles, so orientation accuracy can be scored on real captures.

Base photos (2,000-pixel copies made by fetch_real5_label.py):
  * every Real5 Screen-Photography photo; their own tilt is about 0.5 deg (hand labels: median 0.45), so the base
    angle is taken as 0, or as the hand label where one exists;
  * every hand-labelled Real5 Skew photo, with its labelled angle as the base angle.
Each base photo gives two images: Real5-Rot4, turned clockwise by a balanced seeded quarter turn (exact pixel
transpose), and Real5-RotAny, turned clockwise by a seeded uniform angle in [0, 360) and cropped to its largest inscribed
upright rectangle (no filled corners). Label = base angle + applied angle (mod 360), the ODB-any convention.
Usage: python build_real5_rot.py <view_dir> <labels.jsonl> <out_dir>
"""
import json, math, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

VIEW, LABELS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
labels = {}
for line in open(LABELS): r = json.loads(line); labels[r["image"]] = r
base = [(f"Screen-Photography/{p.name}", labels.get(f"Screen-Photography/{p.name}", {}).get("theta", 0.0) or 0.0, "screen")
        for p in sorted((VIEW / "Screen-Photography").glob("*.jpg")) if labels.get(f"Screen-Photography/{p.name}", {}).get("flag") is None]
base += [(k, r["theta"], "skew") for k, r in sorted(labels.items()) if k.startswith("Skew/") and r.get("theta") is not None]

rng = random.Random(20261003); order = list(range(len(base))); rng.shuffle(order)
quarter = {i: 90 * (rank % 4) for rank, i in enumerate(order)}; angle = {i: rng.uniform(0, 360) for i in range(len(base))}
def inscribed(w, h, ang):
    """Largest axis-aligned rectangle inside a w x h image turned by ang degrees."""
    a = math.radians(ang % 180); sa, ca = abs(math.sin(a)), abs(math.cos(a)); wl, ws = (w, h) if w >= h else (h, w)
    if ws <= 2 * sa * ca * wl or abs(sa - ca) < 1e-10:
        x = 0.5 * ws; wr, hr = (x / sa, x / ca) if w >= h else (x / ca, x / sa)
    else:
        c2 = ca * ca - sa * sa; wr, hr = (w * ca - h * sa) / c2, (h * ca - w * sa) / c2
    return abs(wr), abs(hr)

TRANSPOSE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}   # clockwise turns
for d in ("rot4", "rotany"): (OUT / d).mkdir(parents=True, exist_ok=True)
manifest = []
for i, (rel, beta, split) in enumerate(base):
    img = Image.open(VIEW / rel).convert("RGB"); name = f"{split}_{Path(rel).stem}".replace(" ", "_")
    q = quarter[i]; a = angle[i]
    out4 = OUT / "rot4" / f"{name}.jpg"; outa = OUT / "rotany" / f"{name}.jpg"
    (img if q == 0 else img.transpose(TRANSPOSE[q])).save(out4, quality=92)
    # crop the turned photo to its largest inscribed upright rectangle: filled corners would add straight edges at 0 deg
    # that no real tilted capture has, and both deskewers lock onto them
    rot = img.rotate(-a, resample=Image.BICUBIC, expand=True); wr, hr = inscribed(img.width, img.height, a); W, H = rot.size
    rot.crop(((W - wr) / 2, (H - hr) / 2, (W + wr) / 2, (H + hr) / 2)).save(outa, quality=92)
    manifest += [{"dataset": "Real5-Rot4", "split": split, "id": name, "file": f"rot4/{name}.jpg", "label": round((beta + q) % 360, 2), "base": beta, "applied": q},
                 {"dataset": "Real5-RotAny", "split": split, "id": name, "file": f"rotany/{name}.jpg", "label": round((beta + a) % 360, 2), "base": beta, "applied": round(a, 3)}]
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=0)
from collections import Counter
print("REAL5_ROT_DONE", Counter((m["dataset"], m["split"]) for m in manifest))
