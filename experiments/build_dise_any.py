"""DISE-any: the DISE 2021 test split (Pham et al., ICIP 2022; 2,800 scanned pages with known skew in [-45, 45] deg,
built from DISEC 2013, RVL-CDIP and RDCL 2017) turned clockwise by a balanced seeded quarter turn (exact transpose).

The DISE ground truth g, written in each filename as name[g].png, is the angle jdeskew returns, i.e. the
counterclockwise tilt; our labels are clockwise, so the base angle is -g. Label = (-g + quarter turn) mod 360,
the ODB-any convention. Output layout matches Real5-Rot, so skew_real5_rot.py and cache_real5rot.py run unchanged.
Usage: python build_dise_any.py <dise2021_45/test> <out_dir>
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path
from PIL import Image

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
files = sorted(SRC.glob("*.png")); (OUT / "img").mkdir(parents=True, exist_ok=True)
rng = random.Random(20261004); order = list(range(len(files))); rng.shuffle(order)
quarter = {i: 90 * (rank % 4) for rank, i in enumerate(order)}
TRANSPOSE = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180, 270: Image.Transpose.ROTATE_90}   # clockwise turns
manifest = []
for i, f in enumerate(files):
    base = -float(re.search(r"\[(.+)\]", f.stem).group(1)); q = quarter[i]; name = f"{i:04d}"
    img = Image.open(f).convert("RGB"); img.thumbnail((2000, 2000))
    (img if q == 0 else img.transpose(TRANSPOSE[q])).save(OUT / "img" / f"{name}.png")
    manifest.append({"dataset": "DISE-any", "split": "test", "id": name, "file": f"img/{name}.png", "src": f.name,
                     "label": round((base + q) % 360, 2), "base": base, "applied": q})
json.dump(manifest, open(OUT / "manifest.json", "w"), indent=0)
print("DISE_ANY_DONE", len(manifest), Counter(m["applied"] for m in manifest))
