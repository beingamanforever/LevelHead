"""Sound effects synthesised from noise and sines, then placed on the film's cue sheet.

Each effect is a mono numpy array at 48 kHz; track() pans and sums them into audio/sfx.wav.
Cues come from video.html (window.CUES), so every sound lands on the frame that triggers it.
"""

import wave
from functools import lru_cache

import numpy as np

SR = 48000
TAU = 2 * np.pi


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _rng(seed):
    return np.random.default_rng(seed)


def _band(x, lo, hi):
    """Brick-wall band-pass in the frequency domain, with soft 1/3-octave edges."""
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR) + 1e-9
    gain = 1 / (1 + (lo / f) ** 6) / (1 + (f / hi) ** 6)
    return np.fft.irfft(spec * gain, len(x))


def _sweep_noise(dur, fc, bw, amp, seed):
    """Noise whose pass band follows fc(u), u in 0..1: overlap-added random-phase frames."""
    n, hop = 2048, 256
    frames = int(dur * SR / hop) + 1
    out = np.zeros(frames * hop + n)
    f = np.fft.rfftfreq(n, 1 / SR) + 1
    win, rng = np.hanning(n), _rng(seed)
    for i in range(frames):
        u = min(1.0, i / max(1, frames - 1))
        mag = np.exp(-0.5 * ((np.log2(f) - np.log2(fc(u))) / bw(u)) ** 2)
        frame = np.fft.irfft(mag * np.exp(1j * rng.uniform(0, TAU, len(f))), n) * win
        out[i * hop:i * hop + n] += frame * amp(u)
    return _norm(out[n // 2:n // 2 + int(dur * SR)])


def _norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def _reverb(x, seconds=1.2, wet=0.25, seed=3, bright=6000):
    ir_t = _t(seconds)
    ir = _band(_rng(seed).standard_normal(len(ir_t)), 150, bright) * np.exp(-ir_t * 6.9 / seconds)
    n = len(x) + len(ir)
    tail = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(ir, n), n)
    dry = np.concatenate([x, np.zeros(len(ir))])
    return dry + wet * _norm(tail, np.max(np.abs(x)))


def _fade(x, d_in=0.004, d_out=0.02):
    a, b = int(d_in * SR), int(d_out * SR)
    x = x.copy()
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def whoosh(dur=0.7, lo=260, peak=1900, end=420, seed=1):
    """Air past the camera: the band rises to a peak at 55% and falls, loudest at the peak."""
    fc = lambda u: lo * (peak / lo) ** (u / .55) if u < .55 else peak * (end / peak) ** ((u - .55) / .45)
    amp = lambda u: (u / .55) ** 2 if u < .55 else ((1 - u) / .45) ** 1.6
    return _fade(_sweep_noise(dur, fc, lambda u: 0.75, amp, seed))


def riser(dur=1.4, seed=5):
    """Noise and a sine both climbing to a hard stop, for the beat before an impact."""
    t = _t(dur)
    noise = _sweep_noise(dur, lambda u: 300 * (5200 / 300) ** u, lambda u: 0.6, lambda u: u ** 2.2, seed)
    tone = np.sin(TAU * np.cumsum(110 * 4 ** (t / dur)) / SR) * (t / dur) ** 2
    return _fade(_norm(noise + 0.35 * tone), 0.01, 0.012)


def boom(dur=2.2, seed=7):
    """The cinematic impact: a pitch-dropping sub, a struck transient and a long tail."""
    t = _t(dur)
    f = 36 + 74 * np.exp(-t * 7)
    body = np.tanh(2.4 * np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t * 2.2)) / np.tanh(2.4)
    hit = _band(_rng(seed).standard_normal(len(t)), 60, 2800) * np.exp(-t * 38)
    return _norm(_reverb(_norm(body + 0.55 * _norm(hit)), 1.8, 0.35, seed, 3000))[:int((dur + .8) * SR)]


def hit(seed=9, low=58, snap=0.5):
    """A short slam for words landing: a thump and a bright snap."""
    t = _t(0.5)
    body = np.sin(TAU * np.cumsum(low + 150 * np.exp(-t * 30)) / SR) * np.exp(-t * 11)
    crack = _band(_rng(seed).standard_normal(len(t)), 1800, 9000) * np.exp(-t * 70)
    return _norm(_reverb(_norm(body + snap * _norm(crack)), 0.7, 0.18, seed))


def tick(freq=2600, seed=11):
    t = _t(0.05)
    click = _band(_rng(seed).standard_normal(len(t)), 1500, 7000) * np.exp(-t * 700)
    return _norm(_fade(0.6 * np.sin(TAU * freq * t) * np.exp(-t * 260) + click, 0.0005, 0.005))


def pop(f0=520, f1=900):
    t = _t(0.12)
    f = f0 + (f1 - f0) * (1 - np.exp(-t * 60))
    return _norm(_fade(np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t * 42), 0.001, 0.01))


def note(freq=880.0, dur=0.9):
    """A soft mallet note: fundamental plus the marimba's 3.9x partial, a little room."""
    t = _t(dur)
    x = np.sin(TAU * freq * t) * np.exp(-t * 5) + 0.25 * np.sin(TAU * freq * 3.93 * t) * np.exp(-t * 22)
    return _norm(_reverb(_fade(x, 0.002, 0.05), 0.8, 0.2))


def chime():
    """Success: two bell notes a fifth apart."""
    a, b = note(1318.5, 1.1), note(1975.5, 1.2)
    out = np.zeros(len(b) + int(0.09 * SR))
    out[:len(a)] += a
    out[int(0.09 * SR):] += b
    return _norm(out)


def error():
    """Failure: two low, rounded tones stepping down."""
    t = _t(0.16)
    tone = lambda f: _band(np.tanh(3 * np.sin(TAU * f * t)), 80, 1400) * np.exp(-t * 14)
    out = np.zeros(int(0.42 * SR))
    out[:len(t)] += tone(233)
    out[int(0.15 * SR):int(0.15 * SR) + len(t)] += tone(185)
    return _norm(_reverb(_fade(out), 0.5, 0.15))


def shimmer(dur=1.4, seed=13):
    t = _t(dur)
    air = _band(_rng(seed).standard_normal(len(t)), 5000, 12000)
    rng = _rng(seed + 1)
    bells = sum(np.sin(TAU * f * t + rng.uniform(0, TAU)) for f in rng.uniform(2400, 5200, 7))
    env = np.sin(np.pi * t / dur) ** 2
    return _norm((0.5 * _norm(air) + 0.5 * _norm(bells)) * env)


def scan(dur=1.6, seed=17):
    """A searching tone for the spectrum ray: a filtered band gliding up."""
    return _fade(_sweep_noise(dur, lambda u: 500 * 5 ** u, lambda u: 0.35, lambda u: 0.4 + 0.6 * np.sin(np.pi * u), seed), 0.05, 0.08)


def reveal(dur=0.55, seed=19):
    """The cold open: a reversed swell that opens the frame."""
    t = _t(dur)
    swell = _band(_rng(seed).standard_normal(len(t)), 80, 2400) * (t / dur) ** 3
    sub = np.sin(TAU * 42 * t) * (t / dur) ** 2
    return _fade(_norm(swell + 0.8 * sub), 0.01, 0.006)


KINDS = {"whoosh": whoosh, "riser": riser, "boom": boom, "hit": hit, "tick": tick, "pop": pop, "note": note,
         "chime": chime, "error": error, "shimmer": shimmer, "scan": scan, "reveal": reveal}


@lru_cache(maxsize=None)
def make(kind, params):
    return KINDS[kind](**dict(params))


def track(cues, total, path):
    """Pan and sum every cue into one stereo 16-bit WAV of `total` seconds."""
    out = np.zeros((int((total + 3) * SR), 2))
    for c in cues:
        params = tuple(sorted((k, v) for k, v in c.items() if k not in ("t", "kind", "gain", "pan")))
        x = make(c["kind"], params) * c.get("gain", 1.0)
        pan = c.get("pan", 0.0)
        left, right = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        i = int(c["t"] * SR)
        n = min(len(x), len(out) - i)
        if i < 0 or n <= 0:
            continue
        out[i:i + n, 0] += x[:n] * left * 1.414
        out[i:i + n, 1] += x[:n] * right * 1.414
    out = out[:int(total * SR)]
    peak = np.max(np.abs(out))
    assert peak < 1.0, f"sfx track clips at {peak:.2f}: lower the gains"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((out * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    # audition every effect, one after another, into audio/sfx-demo.wav
    from pathlib import Path
    demo = [{"t": 0.8 * i + (1.5 if i > 2 else 0), "kind": k, "gain": 0.5} for i, k in enumerate(KINDS)]
    track(demo, 14, Path(__file__).parent / "audio/sfx-demo.wav")
    print("audio/sfx-demo.wav")
