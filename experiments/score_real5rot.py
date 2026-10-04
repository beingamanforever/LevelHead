"""Score Real5-Rot caches (cache_real5rot.py): orientation accuracy (within 15 deg) on real photographs turned by
known angles, for classify-first, classify-first + vote, and LevelHead with either deskewer (jdeskew, the paper's
Step 1; profile, the camera-capture Step 1). Also by tilt off the nearest quarter turn on Real5-RotAny.
Usage: python score_real5rot.py <cache_dir> > ../results/real5rot.md
"""
import json, sys
from pathlib import Path
import numpy as np

lg = lambda x: np.log(np.clip(np.asarray(x, float), 1e-12, 1))
frame = lambda p: np.array([np.roll(np.asarray(p, float)[k], -k) for k in range(4)])
pred = lambda s, skew: (skew + 90 * ((-int(np.argmax(s))) % 4)) % 360
err = lambda a, b: abs((a - b + 180) % 360 - 180)
RULES = {"classify-first": lambda r: pred(lg(r["raw"][0]), r["skew"]), "classify-first + vote": lambda r: pred(lg(frame(r["raw"])).mean(0), r["skew"]),
         "LevelHead (jdeskew)": lambda r: pred(lg(frame(r["level"])).mean(0), r["skew"]), "LevelHead (profile)": lambda r: pred(lg(frame(r["level_p"])).mean(0), r["skew_p"])}
NAMES = {"hf_mansee__swin-tiny-patch4-window7-224-img_orientation": "Swin-T", "hf_amaye15__Beit-Base-Image-Orientation-Fixer": "BEiT-B", "doctr": "docTR", "ppdoc": "PP-LCNet"}

for f in sorted(Path(sys.argv[1]).glob("cache_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]; name = NAMES.get(f.stem[6:], f.stem[6:])
    print(f"\n## {name}\n\n| Set | photos | " + " | ".join(RULES) + " |\n|---|---:|" + "---:|" * len(RULES))
    for ds in ("Real5-Rot4", "Real5-RotAny"):
        for sp in ("screen", "skew"):
            g = [r for r in rows if r["dataset"] == ds and r["split"] == sp]
            print(f"| {ds} {sp} | {len(g)} | " + " | ".join(f"{100 * np.mean([err(fn(r), r['label'] % 360) <= 15 for r in g]):.1f}" for fn in RULES.values()) + " |")
    g = [r for r in rows if r["dataset"] == "Real5-RotAny"]; tilt = np.array([abs(((r["label"] % 90) + 45) % 90 - 45) for r in g])
    for lo, hi in ((0, 15), (15, 30), (30, 46)):
        sel = [r for r, t in zip(g, tilt) if lo <= t < hi]
        print(f"| RotAny, tilt {lo}-{min(hi, 45)} deg | {len(sel)} | " + " | ".join(f"{100 * np.mean([err(fn(r), r['label'] % 360) <= 15 for r in sel]):.1f}" for fn in RULES.values()) + " |")
