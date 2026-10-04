"""Smallest end-to-end checks: the deskewer reads a known tilt, the vote cancels a class bias, and the whole
pipeline recovers known angles with a toy backbone that only knows where the page's title block is."""
import numpy as np
from PIL import Image, ImageDraw

from levelhead import LevelHead, estimate_tilt, rolled_vote


def page():
    im = Image.new("RGB", (900, 1200), "white"); d = ImageDraw.Draw(im)
    d.rectangle((120, 80, 780, 200), fill="black")                     # title block marks the top
    for y in range(260, 1120, 28):                                      # rows of "text"
        for x in range(100, 800, 46):
            d.rectangle((x, y, x + 34, y + 12), fill="black")
    return im


def toy_backbone(images):
    """Counterclockwise quarter turns from where the heaviest band of ink sits: top 0, left 1, bottom 2, right 3."""
    out = []
    for im in images:
        a = np.asarray(im.convert("L")) < 128; h, w = a.shape
        bands = [a[: h // 6].mean(), a[:, : w // 6].mean(), a[-h // 6:].mean(), a[:, -w // 6:].mean()]
        p = np.full(4, 0.05); p[int(np.argmax(bands))] = 0.85; out.append(p)
    return np.array(out)


def test_tilt():
    for t in (-30, -7, 12, 33):
        est = estimate_tilt(page().rotate(-t, expand=True, fillcolor="white"))   # rotate(-t) turns the page t degrees clockwise
        assert abs(est - t) < 1.0, (t, est)


def test_bias_cancels():
    rng = np.random.default_rng(0); p = rng.dirichlet(np.ones(4), size=4)
    biased = p * np.array([20.0, 1, 1, 1]); biased /= biased.sum(1, keepdims=True)
    assert np.argmax(rolled_vote(p)) == np.argmax(rolled_vote(biased))


def test_pipeline():
    lh = LevelHead(toy_backbone)
    for theta in (0, 25, 100, 200, 290, 330):
        pred = lh.predict(page().rotate(-theta, expand=True, fillcolor="white")).angle
        assert abs((pred - theta + 180) % 360 - 180) <= 2, (theta, pred)


if __name__ == "__main__":
    test_tilt(); test_bias_cancels(); test_pipeline(); print("ok")
