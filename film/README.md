# The LevelHead film

Source of the 99-second film: [watch it](https://beingamanforever.github.io/LevelHead/#film) or open `docs/levelhead-film.mp4`.
Every frame is drawn by `video.html` from real outputs of the paper's runs, and headless Chrome steps through it frame by frame.

| File | Role |
|---|---|
| `video.html` | the film: `window.seek(t)` draws the frame at time t with no state carried between frames; it also builds the sound-effect cue sheet |
| `render.mjs` | renders the frames with Chrome (playwright-core), encodes them with ffmpeg, and exports the cue sheet |
| `audio.py` | narration (ElevenLabs) and music (Lyria) through OpenRouter, scene timing from the narration, and the final mix |
| `sfx.py` | sound effects synthesised with numpy: whooshes, the impact, riser, hits, ticks, mallet notes and chimes |
| `prep.py` | collects `assets/` and `data/scene.json` from the paper's runs; it needs the run outputs, which are not released |
| `data/` | narration script, scene timing, the real outputs on screen, and the cue sheet |
| `assets/`, `audio/` | public pages (OmniDocBench, Real5 and the Figure 4 pages), recorded narration and music |
| `script.md`, `storyboard.md` | the narration and the shot list, with the source of every number |

## Build

Needs Node 18+, Python 3 with numpy, ffmpeg and Google Chrome (set `CHROME` to its path outside macOS).

```bash
cd film && npm install
node render.mjs stills 3 30 86      # spot-check frames -> out/still-<t>.png
node render.mjs video 60 8          # 1080p60 picture with 8 workers -> out/video-silent.mp4
node render.mjs cues                # sound-effect cue sheet -> data/cues.json
python audio.py sfx                 # sound effects -> audio/sfx.wav
python audio.py mix                 # picture + narration + music + effects -> out/levelhead-demo.mp4
```

The narration and music are committed, so a rebuild needs no API key.
To re-voice or re-score, set `OPENROUTER_API_KEY` and run `python audio.py vo` (which also re-times the scenes; rerun `cues` and `sfx` after it) or `python audio.py music`.
