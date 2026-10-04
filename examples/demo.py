"""Correct the orientation of one page with LevelHead and a Hugging Face orientation classifier.
Usage: python examples/demo.py page.jpg [--out upright.jpg] [--model mansee/swin-tiny-patch4-window7-224-img_orientation]
"""
import argparse

from PIL import Image

from levelhead import LevelHead, classify_first
from levelhead.backbones import HFOrientationClassifier

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("--out", default="upright.jpg")
ap.add_argument("--model", default="mansee/swin-tiny-patch4-window7-224-img_orientation")
args = ap.parse_args()

backbone = HFOrientationClassifier(args.model)
page = Image.open(args.image).convert("RGB")
lh = LevelHead(backbone)
pred = lh.predict(page)
print(f"tilt {pred.tilt:+.1f} deg, quarter turns {pred.quarter_turns}, page turned {pred.angle:.1f} deg clockwise")
print("P(turn) over 0/90/180/270:", " ".join(f"{p:.3f}" for p in pred.posterior))
print(f"classify-first would say {classify_first(backbone, page):.1f} deg")
lh.correct(page).save(args.out); print("wrote", args.out)
