"""Rebuild the LevelHead rotation benchmarks from a local OmniDocBench copy.

We release labels and this script, not images: OmniDocBench stays under its own terms.
  ODB-4    : all pages, each turned clockwise by a balanced, seeded quarter turn
  ODB-any  : all pages, each turned clockwise by a seeded uniform angle in [0, 360), white fill
Seeds and rendering match the paper runs exactly (prep_data.py).
Usage: python build_bench.py /path/to/OmniDocBench/data out_dir
"""
import json, random, sys
from pathlib import Path
from PIL import Image

SEED = 20261003

def build(odb, out):
    pages = sorted((odb / "images").glob("*")); labels = []
    rng = random.Random(SEED); order = list(range(len(pages))); rng.shuffle(order)
    quarter = {i: 90 * (rank % 4) for rank, i in enumerate(order)}
    angles = {i: rng.uniform(0, 360) for i in range(len(pages))}
    for kind in ("odb4", "odbany"): (out / kind).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pages):
        img = Image.open(p).convert("RGB"); img.thumbnail((2200, 2200))
        img.rotate(-quarter[i], expand=True).save(out / "odb4" / f"{i:04d}.png")
        img.rotate(-angles[i], resample=Image.BICUBIC, expand=True, fillcolor="white").save(out / "odbany" / f"{i:04d}.png")
        labels += [{"set": "ODB-4", "id": f"{i:04d}", "source_image": p.name, "clockwise_deg": quarter[i]},
                   {"set": "ODB-any", "id": f"{i:04d}", "source_image": p.name, "clockwise_deg": round(angles[i], 3)}]
    json.dump(labels, open(out / "labels.json", "w"), indent=0)
    print(len(labels), "labelled images in", out)

if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))
