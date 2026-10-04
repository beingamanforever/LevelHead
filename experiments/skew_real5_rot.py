"""Tilt estimates for Real5-Rot from both deskewers (Step 1): jdeskew as in skew_pass.py (height 2048, +-45 deg)
and the projection profile of deskew_real.py. Writes {file: {"jdeskew": deg, "profile": deg}}; both are the
counterclockwise rotation that levels the rows (the clockwise tilt of the page), modulo 90.
Usage: python skew_real5_rot.py <real5rot_dir>
"""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from jdeskew.estimator import get_angle
from deskew_real import profile_angle

D = Path(sys.argv[1])

def one(m):
    im = Image.open(D / m["file"]).convert("RGB")
    return m["file"], {"jdeskew": round(float(get_angle(np.asarray(im), vertical_image_shape=2048, angle_max=45)), 3), "profile": round(float(profile_angle(im)), 3)}

if __name__ == "__main__":
    man = json.load(open(D / "manifest.json")); out = D / "skew.json"
    done = json.load(open(out)) if out.exists() else {}
    todo = [m for m in man if m["file"] not in done]
    with ProcessPoolExecutor(4) as ex:
        for n, (f, s) in enumerate(ex.map(one, todo, chunksize=8)):
            done[f] = s
            if n % 200 == 0: json.dump(done, open(out, "w")); print("skew", n, len(todo), flush=True)
    json.dump(done, open(out, "w")); print("SKEW_DONE", len(done))
