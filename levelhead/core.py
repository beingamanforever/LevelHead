"""Step 2 and 3: vote over the four quarter turns of the level page, rolled back, with a geometric mean.

A backbone is any callable that maps a list of RGB images to an (N, 4) array of probabilities, where column k is
"this image is turned k quarter turns counterclockwise". Views are made with numpy.rot90, so view j of a page
turned c quarter turns is turned c + j; rolling row j back by j puts every view in the page's frame, and the mean
of the log-probabilities (a product of experts) cancels any input-independent class bias of the backbone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PIL import Image

from .deskew import estimate_tilt, level

Backbone = Callable[[Sequence[Image.Image]], np.ndarray]


def quarter_views(image: Image.Image) -> list[Image.Image]:
    """The page turned 0, 1, 2 and 3 quarter turns counterclockwise."""
    a = np.asarray(image.convert("RGB"))
    return [Image.fromarray(np.ascontiguousarray(np.rot90(a, k))) for k in range(4)]


def rolled_vote(probs: np.ndarray, prior: Sequence[float] | None = None, temperature: float = 1.0, floor: float = 1e-4) -> np.ndarray:
    """Posterior over counterclockwise quarter turns of the page from the four views' probabilities (4 x 4)."""
    p = np.asarray(probs, dtype=float)
    rolled = np.stack([np.roll(p[j], -j) for j in range(4)])           # view j read at its own offset
    score = np.log(np.clip(rolled, floor, 1.0)).mean(0) / temperature    # geometric mean of the four experts
    if prior is not None:
        score = score + np.log(np.clip(np.asarray(prior, dtype=float), floor, 1.0))
    post = np.exp(score - score.max())
    return post / post.sum()


@dataclass
class Prediction:
    angle: float          # clockwise angle the page is turned by, in [0, 360); rotate counterclockwise by it to fix
    tilt: float           # clockwise tilt read by the deskewer, modulo 90
    quarter_turns: int    # clockwise quarter turns of the level page
    posterior: np.ndarray  # over clockwise quarter turns 0, 90, 180, 270


class LevelHead:
    """Level first, then vote. Works with any orientation backbone and any deskewer that reads the tilt modulo 90."""

    def __init__(self, backbone: Backbone, deskewer: Callable[[Image.Image], float] = estimate_tilt, prior=None, temperature: float = 1.0):
        self.backbone, self.deskewer, self.prior, self.temperature = backbone, deskewer, prior, temperature

    def predict(self, image: Image.Image) -> Prediction:
        image = image.convert("RGB")
        tilt = float(self.deskewer(image))
        tilt = tilt if abs(tilt) >= 1.0 else 0.0                          # level pages pass through untouched
        probs = np.asarray(self.backbone(quarter_views(level(image, tilt))), dtype=float)
        ccw = rolled_vote(probs, None, self.temperature)
        cw = ccw[[0, 3, 2, 1]]                                             # counterclockwise k == clockwise (4 - k) % 4
        if self.prior is not None:
            cw = cw * np.asarray(self.prior, dtype=float); cw = cw / cw.sum()
        k = int(np.argmax(cw))
        return Prediction(angle=(tilt + 90 * k) % 360, tilt=tilt, quarter_turns=k, posterior=cw)

    def correct(self, image: Image.Image, fill=(255, 255, 255)) -> Image.Image:
        """The page turned upright."""
        return image.convert("RGB").rotate(self.predict(image).angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)


def classify_first(backbone: Backbone, image: Image.Image, deskewer: Callable[[Image.Image], float] = estimate_tilt) -> float:
    """The baseline order: classify the page as given (one view), then deskew. Returns the clockwise angle."""
    p = np.asarray(backbone([image.convert("RGB")]), dtype=float)[0]
    tilt = float(deskewer(image)); tilt = tilt if abs(tilt) >= 1.0 else 0.0
    return (tilt + 90 * ((-int(np.argmax(p))) % 4)) % 360
