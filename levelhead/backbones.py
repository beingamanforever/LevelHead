"""Backbone adapters. Each returns a callable images -> (N, 4) probabilities, column k = turned k quarter turns counterclockwise."""
from __future__ import annotations

import re
from typing import Sequence

import numpy as np
from PIL import Image


def probe_convention(raw, pages: Sequence[Image.Image]) -> list[int]:
    """Learn which output index means "turned k quarter turns counterclockwise" from upright pages turned by known
    amounts. `raw(images)` returns (N, C) probabilities; returns, for k = 0..3, the output index to read."""
    votes = None
    for page in pages:
        a = np.asarray(page.convert("RGB"))
        p = np.asarray(raw([Image.fromarray(np.ascontiguousarray(np.rot90(a, k))) for k in range(4)]))
        votes = np.zeros((4, p.shape[1])) if votes is None else votes
        for k in range(4):
            votes[k, int(np.argmax(p[k]))] += 1
    order = [int(np.argmax(votes[k])) for k in range(4)]
    if len(set(order)) != 4:
        raise ValueError(f"no consistent convention: {votes.tolist()}")
    return order


class HFOrientationClassifier:
    """A Hugging Face image-classification model with four orientation labels.

    `order[k]` is the output index meaning "turned k quarter turns counterclockwise". By default it is read from
    label names such as "90_degree", assuming counterclockwise degrees (true for
    mansee/swin-tiny-patch4-window7-224-img_orientation); check a new model with `probe_convention`.
    """

    def __init__(self, repo: str = "mansee/swin-tiny-patch4-window7-224-img_orientation", order: Sequence[int] | None = None, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForImageClassification.from_pretrained(repo).to(self.device).eval()
        labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        if order is None:
            degrees = [int(re.search(r"\d+", label).group()) % 360 for label in labels]
            order = [degrees.index(90 * k) for k in range(4)]
        self.order = list(order)

    def raw(self, images: Sequence[Image.Image]) -> np.ndarray:
        with self.torch.inference_mode():
            x = self.proc(images=[im.convert("RGB") for im in images], return_tensors="pt")["pixel_values"].to(self.device)
            return self.torch.softmax(self.model(pixel_values=x).logits.float(), 1).cpu().numpy()

    def __call__(self, images: Sequence[Image.Image]) -> np.ndarray:
        return self.raw(images)[:, self.order]
