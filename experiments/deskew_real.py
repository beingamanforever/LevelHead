"""Deskewer accuracy on hand-labelled real photographs (Real5 Skew and Screen-Photography, labels from label_tool.py).

Compares jdeskew (the paper's Step 1, global Fourier spectrum) with a projection-profile deskewer on the centre of
the page: binarise the darkest 15% of pixels in the central two thirds, rotate over candidate angles, and keep the
angle whose row sums change most sharply (text rows horizontal). The centre crop keeps the desk and the perspective
of the page edges out of the estimate. Nothing here is tuned on the labels.
Usage: python deskew_real.py <labels.jsonl> <view_dir> > ../results/deskew_real.md
"""
import json, sys
import numpy as np
from PIL import Image
from jdeskew.estimator import get_angle

fold = lambda d: ((d + 45) % 90) - 45

def hann_angle(img, height=2048, angle_max=45, num=20):
    """jdeskew's radial projection with two fixes. (1) The page is Hann-tapered before the DFT: jdeskew pads with white and
    then thresholds, so a photo darker than white gains an ink line along the padding, a bright cross at 0/90 deg that
    wins over the text; the taper fades the page boundary and the line. (2) A peak at the edge of the search range is returned as is; the released code maps it to 0,
    a rule meant for blank pages, which here are the pages with a flat profile."""
    import cv2
    from jdeskew.estimator import _ensure_gray, _ensure_optimal_square
    a = np.asarray(img); a = cv2.resize(a, None, fx=height / a.shape[0], fy=height / a.shape[0])
    g = _ensure_optimal_square(_ensure_gray(a))
    b = cv2.adaptiveThreshold(~g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10).astype(float)
    win = np.zeros_like(b); win[:a.shape[0], :a.shape[1]] = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))   # taper the page, not the padding
    m = np.abs(np.fft.fftshift(np.fft.fft2(b * win))); c = m.shape[0] // 2; r = np.arange(c)
    tr = np.linspace(-angle_max, angle_max, int(angle_max * num * 2)) / 180 * np.pi   # the released code's grid, same float ops
    li = np.array([m[c + (r * np.cos(t)).astype(np.int32), c + (-r * np.sin(t)).astype(np.int32)].sum() for t in tr])
    return 0.0 if li.max() == li.min() else float(tr[np.argmax(li)] / np.pi * 180)

def profile_angle(img, span=45, step=0.5):
    """Counterclockwise rotation that levels the text rows, i.e. the clockwise tilt of the page, searched in +-span."""
    g = img.convert("L"); g.thumbnail((1000, 1000)); w, h = g.size; g = g.crop((w // 6, h // 6, 5 * w // 6, 5 * h // 6))
    a = np.asarray(g, float); ink = Image.fromarray(((a < np.percentile(a, 15)) * 255).astype(np.uint8))
    score = lambda ang: np.var(np.diff(np.asarray(ink.rotate(ang, resample=Image.BILINEAR), float).sum(1)))
    return max(np.arange(-span, span + 1e-9, step), key=score)

if __name__ == "__main__":
    labels = {}
    for line in open(sys.argv[1]): r = json.loads(line); labels[r["image"]] = r
    rows = [r for r in labels.values() if r.get("theta") is not None]
    res = {"jdeskew": {}, "jdeskew+Hann": {}, "profile": {}}
    for r in rows:
        im = Image.open(f"{sys.argv[2]}/{r['image']}").convert("RGB"); t = fold(r["theta"]); sub = r["image"].split("/")[0]
        est = {"jdeskew": float(get_angle(np.asarray(im), vertical_image_shape=2048, angle_max=45)), "jdeskew+Hann": float(hann_angle(im)), "profile": float(profile_angle(im))}
        for k, s in est.items(): res[k].setdefault(sub, []).append(abs(fold(t - s)))
    print("| Deskewer | Set | photos | true median tilt | median error (deg) | within 2 deg | above 5 deg |\n|---|---|---:|---:|---:|---:|---:|")
    tilt = {sub: np.median([abs(fold(r["theta"])) for r in rows if r["image"].startswith(sub)]) for sub in res["jdeskew"]}
    for k in res:
        for sub, e in sorted(res[k].items()):
            e = np.array(e); print(f"| {k} | {sub} | {len(e)} | {tilt[sub]:.1f} | {np.median(e):.2f} | {100 * (e <= 2).mean():.0f}% | {100 * (e > 5).mean():.0f}% |")
