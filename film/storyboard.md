# LevelHead film - storyboard

1920 × 1080, 60 fps, about 99 seconds, in the LevelHead page style: ink and slate with one blue (`#1F5FAD`, the paper's colour), Geist and Geist Mono.

- Cold open in dark ink, letterboxed to 2.39:1, so the white pages carry the frame; it opens onto the light film through an iris centred on the hero page.
- Light backdrop: a soft blue-grey base with a slowly drifting key light, a cool falloff at the edges, and out-of-focus paper sheets drifting far behind that shift with every scene change. Every page sits on a tinted, dotted stage.
- Fine film grain over everything (soft-light), which also hides banding in the gradients.
- Type: chip pills, 64 px titles that slide up out of a mask, 24 px subtitles, larger and darker mono labels.
- Diagrams: module cards with line icons; smooth gradient connectors with filled heads, and light pulses that travel along them when data flows.
- The Fourier spectrum is a round blue lens that opens like an iris, with the energy along each ray drawn as a polar curve around it.
- A protractor dial turns slowly behind everything and snaps a quarter turn at every scene change.
- Real pages are drawn upright and turned live, so every rotation on screen is the exact angle from the paper's runs.
- Scene lengths come from the narration (`data/timeline.json`); each beat lands on the word that names it.
- Sound: effects are synthesised in `sfx.py` (whooshes, an impact, riser, hits, ticks, pops, mallet notes, chimes, error tones) and placed from the cue sheet that `video.html` builds from the same beats (`node render.mjs cues`), so each sound lands on its frame.

| # | Scene | What is on screen | Where it comes from |
|---|---|---|---|
| 1 | Open | Black, then a slit opens to a letterboxed frame: the camera rushes through a tunnel of real pages turned every way, with speed streaks, depth blur and a rolling horizon. It brakes, and the OmniDocBench page slams to a stop at 152° with a flash and a shockwave; a ring of four arrows cycles without choosing. The voice asks the question, and "Which way is up?" slams in word by word. | The teaser page of Figure 1 (ODB-any 1617); the tunnel uses the method, Real5 and Figure 4 pages |
| 2 | The usual order | SAM 3-style pipeline: page → Classifier (PP-LCNet) → Deskew (jdeskew). "What it was trained on" shows four level pages. On "the guess", the classifier flags "180° off"; the deskewer levels the page, which settles upside down with a red ✗, and docTR's real output types out: "sppauuop p.aypz…", edit distance 0.876. | Figure 1's classify-first row and docTR's actual read |
| 3 | The idea | The method page tumbles to 121°. A dial around it draws 90° (k = 1) in blue, then 31° (s) in light blue. "θ = 90°·k + s". Text rows glow on the page; a glyph "R" turns through four quarter turns. | Eq. 1 and the method page of Figure 2 (ODB-any 0304) |
| 4 | Step 1 · Level | A ghost of the page flows into its log-magnitude spectrum. A ray sweeps −45°…45° while a cursor draws the energy curve, and locks on the peak: s = 31.0°. The page swings level, still a quarter turn off. | The spectrum computed from that page; the ray is drawn along the visible streak (the tilt is read modulo 90°) |
| 5 | Step 1 · Photos | A Real5 phone photo. The released jdeskew's spectrum shows a bright cross and reads 0.0° ✗; a Hann window fades the photo's border, and the tapered spectrum reads −25.8° ✓ (true −26.5°). | The paper's Hann fix (Appendix D), recomputed on a public Real5 photo |
| 6 | Step 2 · Ask | The level page splits into four views (+0°, +90°, +180°, +270°), which flow into a frozen backbone; four answer charts grow. | Qwen3-VL-32B's real answers on that page |
| 7 | Step 3 · Combine | The charts slide left and each rolls back by its own turn. They converge on a ⊗ (geometric mean × prior) and the posterior grows to 0.996 on 90°. The first view is flagged "confidently wrong: 0.95 on 270°". The page returns and turns upright: "turned back 90° + 31° = 121°" ✓. | Figure 2's numbers (method_data.tex) |
| 8 | Any backbone | A LevelHead core with eleven backbones docking: four classifiers (docTR, PP-LCNet, Swin-T, BEiT-B) and seven VLMs (four Qwen3-VL sizes, GPT-4o-mini, Gemma-4-31B, DeepSeek-V4.1-Flash). | The backbones of Table 1 |
| 9 | Results | Accuracy by angle (PP-LCNet, 1,651 ODB-any pages): classify-first dips at every diagonal; LevelHead draws flat on top and the gain fills in. Dumbbells: every backbone at 30–45° of tilt, +16 to +60. | Figure 5 data and Figure 1's right panel, recomputed from the results |
| 10 | Real pages | Figure 4 animated: five difficulty levels, input, classify-first output ✗ and LevelHead output ✓. Pills: phone photos at any angle 69 → 90%, scans at any angle 65 → 91%. | Figure 4 pages; the abstract's numbers (Real5-Rot RotAny and DISE-any, mean of the four classifiers) |
| 11 | Test scores | Two cards, two metrics. Guessing the angle (accuracy %, higher is better): all eleven backbones, classify-first in grey and LevelHead's gain in blue; classifiers +7.2 points, VLMs +38.1. Reading the page (OCR edit distance, lower is better, ODB-any): docTR, PP-OCRv5 and Qwen3-VL-8B as given, after classify-first and after LevelHead, which reaches the oracle line. | Table 1's Avg. column; Table 3 with PP-LCNet as the backbone |
| 12 | Close | A page icon arrives at 121°, levels by 31°, turns upright by 90°. "LevelHead", "Deskew before you classify.", the GitHub repository and four facts. | - |

## Rules kept

- No author name, affiliation or venue, because the paper is under double-blind review. The film links only the public repository, which already exists.
- No private data: every page is from a public benchmark (OmniDocBench, Real5, and the Figure 4 pages).
