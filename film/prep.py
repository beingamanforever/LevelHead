"""Collect real LevelHead outputs for the video into data/scene.json and assets/.

Everything comes from the paper's own runs (~/Documents/LevelHead): the teaser page (ODB-any 1617, turned 152 deg),
the method page (ODB-any 0304, turned 121 deg, Qwen3-VL-32B's four answers), a public Real5 photo for the Hann taper,
the accuracy-by-angle curve (PP-LCNet on ODB-any), the 30-45 deg tilt results for all eleven backbones, and the
qualitative pages of Figure 4. Run with the LevelHead venv: ~/Documents/LevelHead/.venv/bin/python prep.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LH = Path.home() / "Documents/LevelHead"
FIGS = LH / "paper/figs"
sys.path.insert(0, str(FIGS))
from style import NAMES, load  # noqa: E402  (the paper's own loader and backbone names)

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
lg = lambda x: np.log(np.clip(np.asarray(x, float), 1e-12, 1))
frame = lambda p: np.array([np.roll(np.asarray(p, float)[k], -k) for k in range(4)])


def page(src, out, long_side=1500):
    im = Image.open(src).convert("RGB"); im.thumbnail((long_side, long_side)); im.save(ASSETS / out, quality=92)
    return list(im.size)


def spectrum(gray, taper):
    """jdeskew's binarised log spectrum and its radial projection over -45..45 deg (as in fig_hann.py)."""
    h = 1024; img = cv2.resize(gray, None, fx=h / gray.shape[0], fy=h / gray.shape[0])
    s = max(img.shape); sq = np.full((s, s), 255, np.uint8); sq[:img.shape[0], :img.shape[1]] = img
    b = cv2.adaptiveThreshold(~sq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10).astype(float)
    if taper:
        win = np.zeros_like(b); win[:img.shape[0], :img.shape[1]] = np.outer(np.hanning(img.shape[0]), np.hanning(img.shape[1])); b *= win
    m = np.abs(np.fft.fftshift(np.fft.fft2(b))); c = m.shape[0] // 2; r = np.arange(c)
    th = np.linspace(-45, 45, 361)
    li = np.array([m[c + (r * np.cos(t)).astype(np.int32), c + (-r * np.sin(t)).astype(np.int32)].sum() for t in np.radians(th)])
    lm = np.log1p(m); k = c // 2; crop = cv2.GaussianBlur(lm[c - k:c + k, c - k:c + k], (0, 0), 1.2)
    lo, hi = np.percentile(crop, 55), np.percentile(crop, 99.8)
    v = cv2.resize(np.clip((crop - lo) / (hi - lo), 0, 1), (512, 512), interpolation=cv2.INTER_AREA) ** 1.25
    # a round lens: pale blue paper, energy in the paper's blue, transparent outside the circle
    paper, ink = np.array([243, 247, 252.]), np.array([31, 95, 173.])
    rgb = paper * (1 - v[..., None]) + ink * v[..., None]
    yy, xx = np.mgrid[:512, :512]; alpha = np.clip(255 - (np.hypot(xx - 255.5, yy - 255.5) - 253) * 255, 0, 255)
    vis = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")
    return vis, (li / li.max()).round(4).tolist(), float(th[np.argmax(li)])


def main():
    ASSETS.mkdir(exist_ok=True)
    A = FIGS / "assets"
    scene = {}
    # teaser: one page through both orders, docTR's actual reads
    ocr = json.load(open(A / "p1617_ocr.json"))
    snip = lambda a: " ".join(ocr[a].split())[:64]
    scene["teaser"] = {"size": page(A / "p1617_upright.jpg", "teaser.jpg"), "turned": 152.0, "skew": -27.49,
                       "cf": {"undo": 332.5, "ned": 0.876, "text": snip("332.5")}, "vlm_cf": {"undo": 62.5, "ned": 0.997, "text": snip("62.5")},
                       "lh": {"undo": 152.5, "ned": 0.015, "text": snip("152.5")}}
    # method: Qwen3-VL-32B's four answers on the level page (paper convention, from make_method_assets.py)
    p_ccw = np.array([[0.001, 0.949, 0.002, 0.047], [1.0, 0.0, 0.0, 0.0], [0.001, 0.987, 0.0, 0.011], [0.0, 0.542, 0.422, 0.035]])
    P = np.array([[p_ccw[-j % 4][-k % 4] for k in range(4)] for j in range(4)])
    Q = np.array([[P[j][(k + j) % 4] for k in range(4)] for j in range(4)])
    G = np.exp(np.log(np.clip(Q, 1e-6, 1)).mean(0)); G /= G.sum()
    up = Image.open(A / "p0304_upright.jpg").convert("RGB"); up.thumbnail((2200, 2200))
    raw = up.rotate(-121.0, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    vis, prof, peak = spectrum(np.asarray(raw.convert("L")), taper=True); vis.save(ASSETS / "spectrum.png")
    scene["method"] = {"size": page(A / "p0304_upright.jpg", "method.jpg"), "turned": 121.0, "skew": 31.042, "k": int(G.argmax()),
                       "raw": P.round(3).tolist(), "rolled": Q.round(3).tolist(), "posterior": G.round(3).tolist(), "profile": prof, "peak": peak}
    # Hann taper on a public Real5 photo where the released jdeskew answers 0
    photo = LH / "data/jdeskew_failures/03_screen_tilt-26.5.jpg"
    g = cv2.cvtColor(cv2.imread(str(photo)), cv2.COLOR_BGR2GRAY)
    v0, p0, a0 = spectrum(g, taper=False); v1, p1, a1 = spectrum(g, taper=True)
    v0.save(ASSETS / "hann_off.png"); v1.save(ASSETS / "hann_on.png")
    scene["hann"] = {"size": page(photo, "photo.jpg", 1200), "truth": -26.5, "off": {"angle": a0, "profile": p0}, "on": {"angle": a1, "profile": p1}}
    # accuracy by angle (Figure 5): PP-LCNet on ODB-any, 15 deg bins
    recs = [r for r in map(json.loads, open(LH / "results/raw/server/cache_ppdoc.jsonl")) if r["dataset"] == "ODB-any"]
    pred = lambda r, s: (r["skew"] + 90 * ((-int(np.argmax(s))) % 4)) % 360
    err = lambda a, b: abs((a - b + 180) % 360 - 180)
    ang = np.array([r["label"] % 360 for r in recs]); b = np.minimum((ang // 15).astype(int), 23)
    rules = {"cf": lambda r: lg(r["raw"][0]), "vote": lambda r: lg(frame(r["raw"])).mean(0), "lh": lambda r: lg(frame(r["level"])).mean(0)}
    curve = {}
    for name, fn in rules.items():
        ok = np.array([err(pred(r, fn(r)), r["label"] % 360) <= 15 for r in recs], float)
        curve[name] = [round(100 * ok[b == i].mean(), 1) for i in range(24)]
    scene["angle"] = {"pages": len(recs), **curve}
    # accuracy at 30-45 deg of tilt for all eleven backbones (Figure 1, right)
    rows = [("cnn_scores", k) for k in ("doctr", "hf_amaye15__Beit-Base-Image-Orientation-Fixer", "hf_mansee__swin-tiny-patch4-window7-224-img_orientation", "ppdoc")] + \
           [("vlm_scores", k) for k in ("qwen3-vl-8b-instruct", "qwen3-vl-30b-a3b-instruct", "qwen3-vl-32b-instruct", "qwen3-vl-235b-a22b-instruct", "gpt-4o-mini", "gemma-4-31b-it", "deepseek-v4.1-flash")]
    tilt = []
    for src, k in rows:
        cell = [x for x in load(src)["tilt"][k] if x["lo"] == 30][0]
        tilt.append([NAMES[k].replace("†", ""), round(cell["classify_first"], 1), round(cell["level_first"], 1)])
    scene["tilt"] = tilt
    # qualitative pages (Figure 4): input, classify-first output, LevelHead output
    gallery = []
    for lv in range(5):
        stem = f"L{lv}_w0"
        for part in ("in", "cf", "lh"):
            shutil.copy(A / "levels" / f"{stem}_{part}.jpg", ASSETS / f"{stem}_{part}.jpg")
        gallery.append({"stem": stem, **{p: list(Image.open(ASSETS / f"{stem}_{p}.jpg").size) for p in ("in", "cf", "lh")}})
    scene["gallery"] = gallery
    (HERE / "data/scene.json").write_text(json.dumps(scene))
    # sanity: the paper's printed numbers
    assert abs(G[1] - 0.996) < 0.002 and abs(scene["method"]["skew"] - 31.0) < .1, G
    assert abs(peak - 31) < 2, peak
    print("posterior", G.round(3), "spectrum peak", peak, "| hann off/on", round(a0, 1), round(a1, 1))
    print("tilt", tilt)
    print("angle lh min", min(curve["lh"]), "cf min", min(curve["cf"]))


if __name__ == "__main__":
    main()
