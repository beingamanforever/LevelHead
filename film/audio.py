"""Narration, music and the final mix, through OpenRouter.

  python audio.py samples         the first line in four ElevenLabs voices -> audio/sample-*.mp3
  python audio.py vo [voice]      narration per scene -> audio/vo-*.mp3, data/timeline.json
  python audio.py beats           re-time scenes from the existing narration
  python audio.py music           instrumental bed -> audio/music.mp3
  python audio.py sfx             data/cues.json (node render.mjs cues) -> audio/sfx.wav
  python audio.py mix             out/video-silent.mp4 + audio -> out/levelhead-demo.mp4

The key is read from OPENROUTER_API_KEY or ~/.openrouter_env and never printed.
"""

import base64
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
AUDIO = HERE / "audio"
NARRATION = json.loads((HERE / "data/narration.json").read_text())
ORDER = list(NARRATION)          # scene order is the order of data/narration.json
LEAD, TAIL = 0.12, 0.28         # seconds of air before and after each line: the voice runs on without gaps
TTS_MODEL = "elevenlabs/eleven-multilingual-v2"   # ElevenLabs' long-form narration model
MUSIC_MODEL = "google/lyria-3-pro-preview"
VOICES = ["george", "brian", "alice", "sarah"]
SPEED = 1.1                     # ElevenLabs speaking rate: a brisk, natural read
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    env = Path.home() / ".openrouter_env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("OPENROUTER_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("No OpenRouter key: put OPENROUTER_API_KEY in ~/.openrouter_env")
    return key


def post(path, body):
    req = urllib.request.Request("https://openrouter.ai/api/v1" + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"authorization": f"Bearer {api_key()}", "content-type": "application/json"})
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as error:
        sys.exit(f"OpenRouter {path} failed: {error.code} {error.read()[:400]!r}")


def tts(text, voice):
    with post("/audio/speech", {"model": TTS_MODEL, "input": text, "voice": voice, "response_format": "mp3", "speed": SPEED}) as resp:
        return resp.read()


def duration(path):
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out).groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def pause_ends(path):
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path), "-af", "silencedetect=noise=-38dB:d=0.12", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return [float(x) for x in re.findall(r"silence_end: ([\d.]+)", out)]


def beat_time(text, phrase, total, ends):
    """When `phrase` starts: its share of the characters, snapped to the pause before it when it opens a clause."""
    i = text.find(phrase)
    if i < 0:
        sys.exit(f"beat phrase {phrase!r} not in {text!r}")
    guess = total * i / len(text)
    if i and text[i - 2 : i] in (". ", ", ", ": "):
        near = [e for e in ends if abs(e - guess) < 0.45]
        if near:
            return min(near, key=lambda e: abs(e - guess)) - 0.05
    return guess


def samples():
    AUDIO.mkdir(exist_ok=True)
    for voice in VOICES:
        (AUDIO / f"sample-{voice}.mp3").write_bytes(tts(NARRATION[ORDER[0]]["say"], voice))
        print(f"audio/sample-{voice}.mp3")


def voice_over(voice, regenerate=True):
    AUDIO.mkdir(exist_ok=True)
    timeline, beats, starts_at, t = {}, {}, {}, 0.0
    for scene in ORDER:
        spec = NARRATION[scene]
        out = AUDIO / f"vo-{scene}.mp3"
        if regenerate or not out.exists():
            out.write_bytes(tts(spec["say"], voice))
        length, ends, lead = duration(out), pause_ends(out), spec.get("lead", LEAD)   # a scene may hold silence before its line
        timeline[scene] = round(max(spec["min"], lead + length + TAIL), 3)
        beats[scene] = {name: round(lead + beat_time(spec["say"], phrase, length, ends), 3) for name, phrase in spec["beats"].items()}
        starts_at[scene] = round(t + lead, 3)
        t += timeline[scene]
        print(f"{scene:8s} line {length:5.2f} s  scene {timeline[scene]:5.2f} s  beats {beats[scene]}")
    (HERE / "data/timeline.json").write_text(json.dumps({**timeline, "beats": beats, "vo_at": starts_at, "voice": f"{TTS_MODEL}:{voice}"}, indent=1))
    print(f"total {t:.2f} s")


def music():
    cues = json.loads((HERE / "data/cues.json").read_text())
    length = round(cues["total"] - cues["music_at"])   # the bed enters on the opening impact
    prompt = (f"Instrumental only, about {length} seconds. Cinematic, modern electronic score for a launch film about a research method "
              "that turns any rotated document page upright. It enters on a huge low impact: open with a deep sub hit and a "
              "dark, tense synth pulse for the first eight seconds, then brighten into a clean, confident rhythmic arpeggio with "
              "warm pads and deep bass, subtle glitchy ticks, steadily building; swell to an uplifting peak in the last twenty "
              "seconds and resolve on a warm sustained chord. No vocals, no drops, elegant and precise.")
    body = {"model": MUSIC_MODEL, "messages": [{"role": "user", "content": prompt}],
            "modalities": ["text", "audio"], "audio": {"format": "mp3"}, "stream": True}
    chunks, fmt = [], "mp3"
    with post("/chat/completions", body) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:]).get("choices", [{}])[0].get("delta", {})
            audio = delta.get("audio") or {}
            if audio.get("data"):
                chunks.append(audio["data"])
                fmt = audio.get("format", fmt)
    if not chunks:
        sys.exit("no audio came back from the music model")
    raw_file = AUDIO / f"music-raw.{fmt}"
    raw_file.write_bytes(base64.b64decode("".join(chunks)))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(raw_file), "-b:a", "192k", str(AUDIO / "music.mp3")], check=True)
    print("audio/music.mp3", round(duration(AUDIO / "music.mp3"), 2), "s")


def sfx():
    import sfx as synth
    cues = json.loads((HERE / "data/cues.json").read_text())
    synth.track(cues["cues"], cues["total"], AUDIO / "sfx.wav")
    print("audio/sfx.wav", len(cues["cues"]), "cues")


def mix():
    tl = json.loads((HERE / "data/timeline.json").read_text())
    music_at = json.loads((HERE / "data/cues.json").read_text())["music_at"]
    total = sum(tl[s] for s in ORDER)
    inputs, filters, labels = ["-i", str(HERE / "out/video-silent.mp4")], [], []
    for i, scene in enumerate(ORDER, start=1):
        inputs += ["-i", str(AUDIO / f"vo-{scene}.mp3")]
        ms = int(tl["vo_at"][scene] * 1000)
        filters.append(f"[{i}:a]aresample=48000,pan=stereo|c0=c0|c1=c0,adelay={ms}|{ms}[v{i}]")
        labels.append(f"[v{i}]")
    filters.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0[vo]")
    final = "[vo]"
    music_file = AUDIO / "music.mp3"
    if music_file.exists():
        m = len(inputs) // 2
        inputs += ["-i", str(music_file)]
        # The bed dips under the voice (sidechain) and fades out with the end card.
        # a bed a little shorter than the cut is stretched to fit (up to 10%, pitch kept), so it never stops early
        # the bed drops in on the opening impact, after the cold open
        ms = int(music_at * 1000)
        stretch = f"atempo={max(0.9, duration(music_file) / (total - music_at)):.4f}," if duration(music_file) < total - music_at else ""
        filters.append(f"[{m}:a]aresample=48000,aformat=channel_layouts=stereo,{stretch}volume=0.30,afade=t=in:d=0.08,adelay={ms}|{ms},"
                       f"afade=t=out:st={total - 2.5:.2f}:d=2.5[bed]")
        filters.append("[vo]asplit=2[vo1][vo2]")
        filters.append("[bed][vo2]sidechaincompress=threshold=0.03:ratio=6:attack=40:release=450[ducked]")
        filters.append("[vo1][ducked]amix=inputs=2:normalize=0[mixed]")
        final = "[mixed]"
    sfx_file = AUDIO / "sfx.wav"
    if sfx_file.exists():
        inputs += ["-i", str(sfx_file)]
        filters.append(f"{final}[{len(inputs) // 2 - 1}:a]amix=inputs=2:normalize=0[withfx]")
        final = "[withfx]"
    filters.append(f"{final}loudnorm=I=-16:TP=-1.5:LRA=11,apad,atrim=0:{total:.3f}[out]")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
                    "-map", "0:v", "-map", "[out]", "-c:v", "libx264", "-preset", "slow", "-crf", "20",   # grain makes crf 16 three times larger with no visible gain
                   
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                    str(HERE / "out/levelhead-demo.mp4")], check=True)
    print("out/levelhead-demo.mp4", round(duration(HERE / "out/levelhead-demo.mp4"), 2), "s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "vo"
    if cmd in ("vo", "beats"):
        # `beats` re-times the scenes from the narration already on disk
        voice_over(sys.argv[2] if len(sys.argv) > 2 else "george", regenerate=cmd == "vo")
    else:
        {"samples": samples, "music": music, "sfx": sfx, "mix": mix}[cmd]()
