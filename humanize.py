# humanize.py
# ============================================================================
# Humanize Synth -> "Real-ish Piano" (less robotic, more acoustic behavior)
#
# ADDITIONS (Omnisphere-ish piano key vibe):
#   5) Key Noise (attack-only "thock/click" layer, band-limited + shaped)
#   6) Soundboard Resonance (short, damped resonator cluster ~120–450 Hz)
#   7) Pedal / Sympathetic Wash (very small, lowpassed early-reflection bloom)
#   8) Stereo Keybed Micro-Variation (tiny per-channel timing skew, mono-safe)
#
# Notes:
#   - Still NumPy-only, GUI schema, stateless per buffer.
#   - These are "psychoacoustic" layers. They won't replace a real sampled piano,
#     but they push synthesizer behavior toward believable piano cues fast.
# ============================================================================

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple

from pipeline import BaseBlock, BLOCKS, AudioBuffer, ensure_stereo

_TWOPI = 2.0 * np.pi


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _softsat(x: np.ndarray, drive: float) -> np.ndarray:
    d = float(np.clip(drive, 0.0, 12.0))
    if d <= 1e-9:
        return x.astype(np.float32, copy=False)
    k = 0.55 + (d ** 1.05)
    y = x * (1.0 + k) / (1.0 + k * np.abs(x))
    return y.astype(np.float32)

def _onepole_env(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    sr = int(max(1, sr))
    a = float(max(0.01, attack_ms))
    r = float(max(0.01, release_ms))
    a_c = np.exp(-1.0 / (sr * (a / 1000.0)))
    r_c = np.exp(-1.0 / (sr * (r / 1000.0)))

    env = np.empty_like(x, dtype=np.float32)
    y = 0.0
    for i in range(x.shape[0]):
        v = float(abs(x[i]))
        if v > y:
            y = a_c * y + (1.0 - a_c) * v
        else:
            y = r_c * y + (1.0 - r_c) * v
        env[i] = y
    return env

def _biquad_process_mono(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x, dtype=np.float32)
    x1 = x2 = 0.0
    y1 = y2 = 0.0
    for i in range(x.shape[0]):
        x0 = float(x[i])
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
    return y

def _biquad_process_stereo(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    yL = _biquad_process_mono(x[:, 0], b0, b1, b2, a1, a2)
    yR = _biquad_process_mono(x[:, 1], b0, b1, b2, a1, a2)
    return np.stack([yL, yR], axis=1).astype(np.float32)

def _rbj_peaking(sr: int, f0: float, q: float, gain_db: float) -> tuple[float, float, float, float, float]:
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    q = float(np.clip(q, 0.05, 24.0))
    A = float(10.0 ** (gain_db / 40.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)
    alpha = s / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * c
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * c
    a2 = 1.0 - alpha / A

    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0
    return float(b0), float(b1), float(b2), float(a1), float(a2)

def _rbj_highshelf(sr: int, f0: float, slope: float, gain_db: float) -> tuple[float, float, float, float, float]:
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    slope = float(np.clip(slope, 0.1, 5.0))
    A = float(10.0 ** (gain_db / 40.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)

    alpha = s / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) + (A - 1.0) * c + beta)
    b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * c)
    b2 = A * ((A + 1.0) + (A - 1.0) * c - beta)
    a0 = (A + 1.0) - (A - 1.0) * c + beta
    a1 = 2.0 * ((A - 1.0) - (A + 1.0) * c)
    a2 = (A + 1.0) - (A - 1.0) * c - beta

    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0
    return float(b0), float(b1), float(b2), float(a1), float(a2)

def _rbj_lowshelf(sr: int, f0: float, slope: float, gain_db: float) -> tuple[float, float, float, float, float]:
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    slope = float(np.clip(slope, 0.1, 5.0))
    A = float(10.0 ** (gain_db / 40.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)

    alpha = s / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    beta = 2.0 * np.sqrt(A) * alpha

    b0 = A * ((A + 1.0) - (A - 1.0) * c + beta)
    b1 = 2.0 * A * ((A - 1.0) - (A + 1.0) * c)
    b2 = A * ((A + 1.0) - (A - 1.0) * c - beta)
    a0 = (A + 1.0) + (A - 1.0) * c + beta
    a1 = -2.0 * ((A - 1.0) + (A + 1.0) * c)
    a2 = (A + 1.0) + (A - 1.0) * c - beta

    b0 /= a0; b1 /= a0; b2 /= a0
    a1 /= a0; a2 /= a0
    return float(b0), float(b1), float(b2), float(a1), float(a2)

def _fractional_delay_mono(x: np.ndarray, delay_samps: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    y = np.empty(n, dtype=np.float32)
    for i in range(n):
        d = float(delay_samps[i])
        idx = i - d
        if idx <= 0.0:
            y[i] = float(x[0]); continue
        if idx >= n - 1:
            y[i] = float(x[-1]); continue
        j = int(idx)
        frac = idx - j
        y[i] = float((1.0 - frac) * x[j] + frac * x[j + 1])
    return y

def _stereo_mid_side(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1])
    return m.astype(np.float32), s.astype(np.float32)

def _stereo_from_mid_side(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    l = (m + s).astype(np.float32)
    r = (m - s).astype(np.float32)
    return np.stack([l, r], axis=1).astype(np.float32)

def _smoothstep(x: np.ndarray) -> np.ndarray:
    return (x * x * (3.0 - 2.0 * x)).astype(np.float32)

def _lin_interp(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sample x at fractional indices idx (mono)."""
    n = x.shape[0]
    idx = np.clip(idx, 0.0, n - 1.00001)
    j = idx.astype(np.int32)
    frac = idx - j
    j2 = np.clip(j + 1, 0, n - 1)
    return ((1.0 - frac) * x[j] + frac * x[j2]).astype(np.float32)


# ----------------------------------------------------------------------------
# New: 5) Key Noise (attack-only layer)
# ----------------------------------------------------------------------------

class PianoKeyNoise(BaseBlock):
    """
    Adds a very small key/hammer mechanical noise layer:
      - derived from fast transient energy (env_fast - env_slow)
      - band-limited around 1k-5k (click/thock region)
      - only around attacks; won't smear sustain
    """
    KIND = "fx"
    PARAMS = {
        "amount": {"type": "float", "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
        "center_hz": {"type": "float", "default": 2400.0, "min": 800.0, "max": 6500.0, "step": 50.0},
        "q": {"type": "float", "default": 1.2, "min": 0.4, "max": 6.0, "step": 0.05},
        "attack_ms": {"type": "float", "default": 1.2, "min": 0.3, "max": 20.0, "step": 0.1},
        "release_ms": {"type": "float", "default": 30.0, "min": 5.0, "max": 200.0, "step": 1.0},
        "sat": {"type": "float", "default": 0.10, "min": 0.0, "max": 6.0, "step": 0.1},
        "mix": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        n = x.shape[0]

        amount = _clip01(params.get("amount", 0.18))
        center = float(params.get("center_hz", 2400.0))
        q = float(params.get("q", 1.2))
        a_ms = float(params.get("attack_ms", 1.2))
        r_ms = float(params.get("release_ms", 30.0))
        sat = float(params.get("sat", 0.10))
        mix = _clip01(params.get("mix", 0.30))
        seed = int(params.get("seed", 0))

        if mix <= 1e-6 or amount <= 1e-6 or n < 4:
            return AudioBuffer(x, sr), {}

        # transient detector
        m, _ = _stereo_mid_side(x)
        env_fast = _onepole_env(m, sr, a_ms, max(2.0, a_ms * 2.0))
        env_slow = _onepole_env(m, sr, 20.0, max(40.0, r_ms))

        trans = np.clip((env_fast - env_slow) / (np.max(env_fast) + 1e-6), 0.0, 1.0).astype(np.float32)
        trans = _smoothstep(trans)

        # noise source (stateless)
        rng = np.random.RandomState(seed & 0xFFFFFFFF)
        noise = rng.normal(0.0, 1.0, size=n).astype(np.float32)

        # band-limit by peaking EQ boost then top cut (cheap but effective)
        b0, b1, b2, a1, a2 = _rbj_peaking(sr, center, q, 6.0)
        k = _biquad_process_mono(noise, b0, b1, b2, a1, a2)
        b0, b1, b2, a1, a2 = _rbj_highshelf(sr, 6500.0, 0.9, -10.0)
        k = _biquad_process_mono(k, b0, b1, b2, a1, a2)

        # gate it to attacks
        k *= (amount * trans)

        # make it more "thock" than hiss
        k = _softsat(k, sat)

        # mix equally into L/R (mono mechanical noise is realistic)
        y = x + mix * np.stack([k, k], axis=1).astype(np.float32)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), {}


# ----------------------------------------------------------------------------
# New: 6) Soundboard Resonance (short, damped resonator cluster)
# ----------------------------------------------------------------------------

class PianoSoundboardResonance(BaseBlock):
    """
    Adds short, damped resonance in low-mids (like a piano body/soundboard):
      - uses a few short comb-ish taps (very short delays)
      - lowpassed/damped by shelving
      - driven by signal (no external IR)
    """
    KIND = "fx"
    PARAMS = {
        "amount": {"type": "float", "default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01},
        "base_hz": {"type": "float", "default": 170.0, "min": 90.0, "max": 420.0, "step": 5.0},
        "spread": {"type": "float", "default": 0.22, "min": 0.0, "max": 0.8, "step": 0.01},
        "damp": {"type": "float", "default": 0.72, "min": 0.0, "max": 0.95, "step": 0.01},
        "decay": {"type": "float", "default": 0.55, "min": 0.1, "max": 0.95, "step": 0.01},
        "mix": {"type": "float", "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        n = x.shape[0]

        amount = _clip01(params.get("amount", 0.22))
        base_hz = float(params.get("base_hz", 170.0))
        spread = float(np.clip(float(params.get("spread", 0.22)), 0.0, 0.95))
        damp = float(np.clip(float(params.get("damp", 0.72)), 0.0, 0.95))
        decay = float(np.clip(float(params.get("decay", 0.55)), 0.05, 0.98))
        mix = _clip01(params.get("mix", 0.18))
        seed = int(params.get("seed", 0))

        if mix <= 1e-6 or amount <= 1e-6 or n < 8:
            return AudioBuffer(x, sr), {}

        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        # build 3 resonant delays from base frequency
        # delay samples roughly sr / f
        freqs = np.array([1.0, 1.0 + 0.55 * spread, 1.0 + 1.05 * spread], dtype=np.float32) * base_hz
        freqs *= (1.0 + rng.uniform(-0.03, 0.03, size=freqs.shape)).astype(np.float32)

        delays = np.clip((sr / np.clip(freqs, 30.0, 0.49 * sr)), 8.0, float(n - 1)).astype(np.float32)

        # create a short feedback-ish resonance using a few taps (stateless, so no feedback loop)
        # wet = x + decay * delayed(x) + decay^2 * delayed2(x) ...
        wet = x.copy()
        for d in delays:
            di = int(round(float(d)))
            if di < n:
                wet[di:] += (x[:-di] * (amount * decay)).astype(np.float32)
            di2 = int(round(float(d) * 2.0))
            if di2 < n:
                wet[di2:] += (x[:-di2] * (amount * (decay ** 2))).astype(np.float32)

        # damp highs on wet to keep it woody
        b0, b1, b2, a1, a2 = _rbj_highshelf(sr, 4200.0, 0.9, -18.0 * damp)
        wet = _biquad_process_stereo(wet, b0, b1, b2, a1, a2)

        out = (1.0 - mix) * x + mix * wet
        return AudioBuffer(out.astype(np.float32, copy=False), sr), {}


# ----------------------------------------------------------------------------
# New: 7) Pedal / Sympathetic Wash (tiny bloom, lowpassed, level-following)
# ----------------------------------------------------------------------------

class PianoPedalBloom(BaseBlock):
    """
    Adds a subtle "pedal down" sympathetic bloom:
      - extremely short, lowpassed smear that follows level
      - NOT a reverb; it's a tiny wash under the sustain
    """
    KIND = "fx"
    PARAMS = {
        "amount": {"type": "float", "default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01},
        "lp_hz": {"type": "float", "default": 5200.0, "min": 1200.0, "max": 12000.0, "step": 100.0},
        "follow": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
        "attack_ms": {"type": "float", "default": 18.0, "min": 2.0, "max": 200.0, "step": 1.0},
        "release_ms": {"type": "float", "default": 220.0, "min": 10.0, "max": 2000.0, "step": 5.0},
        "mix": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        n = x.shape[0]

        amount = _clip01(params.get("amount", 0.22))
        lp_hz = float(params.get("lp_hz", 5200.0))
        follow = _clip01(params.get("follow", 0.55))
        a_ms = float(params.get("attack_ms", 18.0))
        r_ms = float(params.get("release_ms", 220.0))
        mix = _clip01(params.get("mix", 0.10))
        seed = int(params.get("seed", 0))

        if mix <= 1e-6 or amount <= 1e-6 or n < 8:
            return AudioBuffer(x, sr), {}

        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        m, _ = _stereo_mid_side(x)
        env = _onepole_env(m, sr, a_ms, r_ms)
        lvl = np.clip(env / (np.max(env) + 1e-6), 0.0, 1.0).astype(np.float32)
        lvl = _smoothstep(lvl)

        # very short smear (a few ms)
        d1 = max(1, int(round((2.5 + rng.uniform(-0.6, 0.6)) * 0.001 * sr)))
        d2 = max(1, int(round((5.5 + rng.uniform(-0.8, 0.8)) * 0.001 * sr)))

        wet = x.copy()
        if d1 < n:
            wet[d1:] += x[:-d1] * (amount * 0.22)
        if d2 < n:
            wet[d2:] += x[:-d2] * (amount * 0.14)

        # lowpass-ish by shelving down highs above lp_hz
        b0, b1, b2, a1, a2 = _rbj_highshelf(sr, lp_hz, 0.9, -18.0)
        wet = _biquad_process_stereo(wet, b0, b1, b2, a1, a2)

        # follow level (more bloom when sustaining)
        wet = x + (wet - x) * (follow * lvl[:, None])

        out = (1.0 - mix) * x + mix * wet
        return AudioBuffer(out.astype(np.float32, copy=False), sr), {}


# ----------------------------------------------------------------------------
# New: 8) Stereo Keybed Micro-Variation (tiny timing skew, mono-safe)
# ----------------------------------------------------------------------------

class PianoKeybedMicro(BaseBlock):
    """
    A tiny L/R timing skew to mimic mic placement + keybed irregularity.
    This is NOT chorus: delays are sub-millisecond to ~2ms and constant per buffer.
    """
    KIND = "fx"
    PARAMS = {
        "max_ms": {"type": "float", "default": 1.6, "min": 0.0, "max": 6.0, "step": 0.1},
        "amount": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
        "mix": {"type": "float", "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        n = x.shape[0]

        max_ms = float(np.clip(float(params.get("max_ms", 1.6)), 0.0, 20.0))
        amt = _clip01(params.get("amount", 0.55))
        mix = _clip01(params.get("mix", 0.20))
        seed = int(params.get("seed", 0))

        if mix <= 1e-6 or amt <= 1e-6 or max_ms <= 1e-6 or n < 4:
            return AudioBuffer(x, sr), {}

        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        max_s = (max_ms * 0.001 * sr) * amt
        # constant per buffer, tiny
        dL = float(np.clip(rng.uniform(0.0, max_s), 0.0, max_s))
        dR = float(np.clip(rng.uniform(0.0, max_s), 0.0, max_s))

        idx = np.arange(n, dtype=np.float32)
        y = np.empty_like(x, dtype=np.float32)
        y[:, 0] = _lin_interp(x[:, 0], idx - dL)
        y[:, 1] = _lin_interp(x[:, 1], idx - dR)

        out = (1.0 - mix) * x + mix * y
        return AudioBuffer(out.astype(np.float32, copy=False), sr), {}


# ----------------------------------------------------------------------------
# Registration (new piano-ish layers)
# ----------------------------------------------------------------------------

BLOCKS.register("piano_key_noise", PianoKeyNoise)
BLOCKS.register("piano_soundboard", PianoSoundboardResonance)
BLOCKS.register("piano_pedal_bloom", PianoPedalBloom)
BLOCKS.register("piano_keybed_micro", PianoKeybedMicro)
