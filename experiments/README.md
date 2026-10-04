# Experiments

Scripts behind the paper's new benchmarks, the VLM runs and the deskewer study.
They expect a working directory `BENCH` (default `./bench`) that holds the prepared views; paths are set at the top of each script.

| Script | What it does |
|---|---|
| `build_bench.py` | Rebuilds **ODB-4** and **ODB-any** bit for bit from a local OmniDocBench copy (seed 20261003). |
| `fetch_real5_label.py` | Downloads the Real5-OmniDocBench Skew and Screen-Photography photographs. |
| `label_tool.py` | Keyboard-driven web tool with a ruler overlay for hand-labelling the tilt of a photograph. |
| `build_real5_rot.py` | Builds **Real5-Rot**: each photograph in a known quarter turn (Rot4) and at a known angle, cropped to its largest upright rectangle (RotAny). |
| `build_dise_any.py` | Builds **DISE-any**: the 2,800 DISE 2021 test scans in balanced quarter turns. |
| `deskew_real.py` | The corrected jdeskew (`hann_angle`) and a projection-profile baseline, scored on hand-labelled photographs. |
| `skew_real5_rot.py` | Tilt estimates of both deskewers for every Real5-Rot / DISE-any image. |
| `score_real5rot.py` | Classify-first, classify-first + vote and LevelHead accuracy, by capture and by tilt. |
| `vlm_views.py` | Any VLM on OpenRouter as a backbone: shuffled options per call, letter probabilities from log-probabilities (`OPENROUTER_API_KEY` from the environment). |
| `local_vlm.py` | A local open VLM as a backbone, with exact probabilities for the four answer letters. |

Labels are released, images are not: every source dataset stays under its own terms.
Ground truth: `labels/real5_hand_labels.jsonl` (clockwise tilt of 139 Real5 photographs, one annotator), `labels/real5_rot.json` and `labels/dise_any.json` (label = clockwise angle the image is turned by).
