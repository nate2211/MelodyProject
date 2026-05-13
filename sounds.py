from __future__ import annotations

from typing import Any, Dict, Tuple, Optional, List

import numpy as np

from pipeline import BaseBlock, BLOCKS, AudioBuffer, ensure_stereo

try:
    from melodyproject_native import NATIVE_DSP, native_status
except Exception:
    NATIVE_DSP = None  # type: ignore

    def native_status() -> Dict[str, Any]:
        return {"available": False, "load_error": "melodyproject_native not importable"}


# ============================================================================
# sounds.py
#
# Fixed for pipeline block params.
#
# Main fix:
# - Every execute(payload, *, params) merges incoming pipeline params with PARAMS
#   defaults before rendering.
# - Instruments accept both:
#       wave / waveform
#       amp / gain / master_gain
#       velocity / vel from payload
# - Wave controls are rendered in Python so changing wave params actually changes
#   the audible sound.
#
# Blocks:
#   Instruments:
#       synth_keys
#       lead_synth
#       piano_keys
#       piano_key
#       guitar_pluck
#       bell_fm
#       brass_synth
#       flute_synth
#       clarinet_synth
#       string_pad
#
#   FX:
#       gain
#       delay
#       lowpass
#       highpass
#       bandpass
#       softclip
#       sound_polish
# ============================================================================

_TWOPI = 2.0 * np.pi


# ============================================================================
# Native helpers
# ============================================================================

def _native_available() -> bool:
    return bool(NATIVE_DSP is not None and getattr(NATIVE_DSP, "available", False))


def _native_has(name: str) -> bool:
    return bool(NATIVE_DSP is not None and hasattr(NATIVE_DSP, name))


def _meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "native": _native_available(),
        "native_status": native_status(),
    }
    if extra:
        out.update(extra)
    return out


# ============================================================================
# Param helpers
# ============================================================================

def _defaults_from_schema(schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {k: v.get("default") for k, v in (schema or {}).items()}


def _merge_params(schema: Dict[str, Dict[str, Any]], params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Critical pipeline fix.

    The GUI/pipeline may pass only changed params inside BlockInstance.params.
    This merges those changed params with PARAMS defaults so every sound block
    always receives a complete usable param set.
    """
    out = _defaults_from_schema(schema)
    if params:
        out.update(dict(params))

    # Accept old/simple synth param names too.
    if "waveform" in out and "wave" not in out:
        out["wave"] = out["waveform"]
    if "wave" in out and "waveform" not in out:
        out["waveform"] = out["wave"]

    if "gain" in out and "amp" not in out:
        out["amp"] = out["gain"]
    if "master_gain" in out and "amp" not in out:
        out["amp"] = out["master_gain"]
    if "amp" in out and "gain" not in out:
        out["gain"] = out["amp"]

    return out


def _voice_payload(payload: Any, params: Dict[str, Any]) -> Tuple[float, float, int, float]:
    """
    Accepts the payload format produced by pipeline._render_python_instrument_track.
    """
    if not isinstance(payload, dict):
        sr = int(params.get("sr", 48000))
        return 440.0, 0.25, sr, 1.0

    freq = float(payload.get("freq", params.get("freq", 440.0)))
    dur = float(payload.get("duration", params.get("duration", 0.25)))
    sr = int(payload.get("sr", params.get("sr", 48000)))
    vel = float(payload.get("vel", payload.get("velocity", params.get("velocity", 1.0))))

    return freq, dur, sr, vel


def _frames(dur: float, sr: int) -> int:
    return max(1, int(round(float(dur) * int(sr))))


def _clamp(x: float, a: float, b: float) -> float:
    return float(np.clip(float(x), float(a), float(b)))


def _as_audio_buffer(payload: Any, sr: int) -> AudioBuffer:
    if isinstance(payload, AudioBuffer):
        return AudioBuffer(_sanitize(payload.data), int(payload.sr))
    return AudioBuffer(_sanitize(payload), int(sr))


def _sanitize(x: Any, ceiling: float = 0.995) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -float(ceiling), float(ceiling)).astype(np.float32, copy=False)


# ============================================================================
# Wave helpers
# ============================================================================

_WAVE_ALIASES = {
    "sin": "sine",
    "sine": "sine",

    "tri": "triangle",
    "triangle": "triangle",

    "sq": "square",
    "sqr": "square",
    "square": "square",
    "pulse": "square",
    "pwm": "square",

    "saw": "saw",
    "sawtooth": "saw",
    "saw-tooth": "saw",
    "tooth": "saw",

    "organ": "organ",
    "drawbar": "organ",

    "harm": "harmonic",
    "harmonic": "harmonic",

    "bright": "bright",
    "nasal": "nasal",

    "bass": "bass",
    "sub": "bass",

    "noise": "noise",
    "white": "noise",
    "white_noise": "noise",
}

_WAVE_CHOICES = [
    "sine",
    "triangle",
    "square",
    "pulse",
    "saw",
    "sawtooth",
    "organ",
    "harmonic",
    "bright",
    "nasal",
    "bass",
    "noise",
]


def _norm_wave(w: Any, default: str = "sine") -> str:
    s = str(w or "").strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return _WAVE_ALIASES.get(s, default)


def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(t, dtype=np.float32)
    dt = float(max(1.0e-8, dt))

    m = t < dt
    if np.any(m):
        x = (t[m] / dt).astype(np.float32)
        out[m] = x + x - x * x - 1.0

    m = t > (1.0 - dt)
    if np.any(m):
        x = ((t[m] - 1.0) / dt).astype(np.float32)
        out[m] = x * x + x + x + 1.0

    return out


def _saw_blep(phase01: np.ndarray, dt: float) -> np.ndarray:
    y = (2.0 * phase01 - 1.0).astype(np.float32)
    y -= _poly_blep(phase01, dt)
    return y.astype(np.float32, copy=False)


def _square_blep(phase01: np.ndarray, dt: float, pwm: float) -> np.ndarray:
    pwm = float(np.clip(pwm, 0.01, 0.99))
    y = np.where(phase01 < pwm, 1.0, -1.0).astype(np.float32)
    y += _poly_blep(phase01, dt)
    t2 = (phase01 - pwm) % 1.0
    y -= _poly_blep(t2, dt)
    return y.astype(np.float32, copy=False)


def _tri_from_square(square: np.ndarray) -> np.ndarray:
    y = np.cumsum(square).astype(np.float32)
    y -= np.mean(y)
    m = float(np.max(np.abs(y))) + 1.0e-8
    return (y / m).astype(np.float32, copy=False)


def _phase_distort(phase01: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, -0.95, 0.95))
    if abs(a) < 1.0e-8:
        return phase01.astype(np.float32, copy=False)

    p = phase01.astype(np.float32, copy=False)
    bend = 0.5 + 0.45 * a
    bend = float(np.clip(bend, 0.05, 0.95))

    out = np.empty_like(p, dtype=np.float32)
    m = p < bend
    out[m] = (p[m] / bend) * 0.5
    out[~m] = 0.5 + ((p[~m] - bend) / (1.0 - bend)) * 0.5
    return out.astype(np.float32, copy=False)


def _asym_phase(phase01: np.ndarray, asymmetry: float) -> np.ndarray:
    a = float(np.clip(asymmetry, -1.0, 1.0))
    if abs(a) < 1.0e-8:
        return phase01.astype(np.float32, copy=False)

    p = phase01.astype(np.float32, copy=False)
    warped = p + 0.12 * a * np.sin(_TWOPI * p)
    return (warped % 1.0).astype(np.float32, copy=False)


def _osc_wave_from_phase(
    wave: str,
    phase01: np.ndarray,
    dt: float,
    pwm: float,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    wave = _norm_wave(wave, default="sine")
    phase01 = np.asarray(phase01, dtype=np.float32) % 1.0
    dt = float(max(1.0e-8, dt))

    if wave == "sine":
        return np.sin(_TWOPI * phase01).astype(np.float32)

    if wave == "saw":
        return _saw_blep(phase01, dt)

    if wave == "square":
        return _square_blep(phase01, dt, pwm=pwm)

    if wave == "triangle":
        sq = _square_blep(phase01, dt, pwm=0.5)
        return _tri_from_square(sq)

    if wave == "organ":
        y = (
            1.00 * np.sin(_TWOPI * phase01)
            + 0.55 * np.sin(_TWOPI * 2.0 * phase01)
            + 0.30 * np.sin(_TWOPI * 3.0 * phase01)
            + 0.18 * np.sin(_TWOPI * 4.0 * phase01)
        ).astype(np.float32)
        return (y / 1.85).astype(np.float32, copy=False)

    if wave == "harmonic":
        y = (
            1.00 * np.sin(_TWOPI * phase01)
            + 0.33 * np.sin(_TWOPI * 2.0 * phase01 + 0.11)
            + 0.22 * np.sin(_TWOPI * 3.0 * phase01 + 0.23)
            + 0.12 * np.sin(_TWOPI * 5.0 * phase01 + 0.37)
        ).astype(np.float32)
        return (y / 1.55).astype(np.float32, copy=False)

    if wave == "bright":
        y = (
            _saw_blep(phase01, dt)
            + 0.45 * _square_blep((phase01 + 0.08) % 1.0, dt, pwm=0.42)
            + 0.16 * np.sin(_TWOPI * 7.0 * phase01)
        ).astype(np.float32)
        return (y / 1.45).astype(np.float32, copy=False)

    if wave == "nasal":
        y = (
            0.45 * np.sin(_TWOPI * phase01)
            + 1.00 * np.sin(_TWOPI * 2.0 * phase01 + 0.05)
            + 0.45 * np.sin(_TWOPI * 3.0 * phase01 + 0.10)
        ).astype(np.float32)
        return (y / 1.45).astype(np.float32, copy=False)

    if wave == "bass":
        y = (
            1.00 * np.sin(_TWOPI * phase01)
            + 0.24 * _square_blep(phase01, dt, pwm=0.50)
            + 0.10 * np.sin(_TWOPI * 0.5 * phase01)
        ).astype(np.float32)
        return np.tanh(y * 0.95).astype(np.float32)

    if wave == "noise":
        if rng is None:
            rng = np.random.RandomState(0)
        return rng.uniform(-1.0, 1.0, size=phase01.shape[0]).astype(np.float32)

    return np.sin(_TWOPI * phase01).astype(np.float32)


def _waveshape(x: np.ndarray, drive: float, fold: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    drive = float(np.clip(drive, 0.05, 24.0))
    fold = float(np.clip(fold, 0.0, 1.0))

    y = np.tanh(x * (0.30 + drive ** 1.15)).astype(np.float32)

    if fold > 1.0e-8:
        k = 1.0 + 10.0 * (fold ** 0.9)
        z = y * k
        z = ((z + 1.0) % 4.0) - 2.0
        z = 2.0 - np.abs(z)
        y = (z - 1.0).astype(np.float32)

    return y.astype(np.float32, copy=False)


def _tilt_eq(x: np.ndarray, tilt: float) -> np.ndarray:
    tilt = float(np.clip(tilt, -1.0, 1.0))
    if abs(tilt) < 1.0e-8:
        return x.astype(np.float32, copy=False)

    dx = np.empty_like(x, dtype=np.float32)
    dx[0] = x[0]
    dx[1:] = x[1:] - x[:-1]

    y = np.tanh(x + 1.15 * tilt * dx).astype(np.float32)
    return y.astype(np.float32, copy=False)


def _bitcrush_mono(x: np.ndarray, bit_depth: float) -> np.ndarray:
    bits = float(np.clip(bit_depth, 2.0, 24.0))
    if bits >= 23.5:
        return x.astype(np.float32, copy=False)

    levels = float(2 ** int(round(bits)))
    return (np.round(np.clip(x, -1.0, 1.0) * (levels * 0.5)) / (levels * 0.5)).astype(np.float32)


def _sample_hold_mono(x: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1.0e-8:
        return x.astype(np.float32, copy=False)

    step = int(1 + round(amount * 96.0))
    if step <= 1:
        return x.astype(np.float32, copy=False)

    idx = (np.arange(x.shape[0]) // step) * step
    idx = np.clip(idx, 0, x.shape[0] - 1)
    return x[idx].astype(np.float32, copy=False)


def osc_advanced(
    wave: str,
    t: np.ndarray,
    freq: float,
    sr: int,
    *,
    unison: int = 1,
    detune_cents: float = 0.0,
    spread: float = 0.5,
    pwm: float = 0.5,
    sync: float = 0.0,
    sync_ratio: float = 2.0,
    fm_ratio: float = 0.0,
    fm_index: float = 0.0,
    pm_amount: float = 0.0,
    pd_amount: float = 0.0,
    drive: float = 1.0,
    fold: float = 0.0,
    tilt: float = 0.0,
    seed: int = 0,
    wave_alt: str = "sine",
    wave_blend: float = 0.0,
    morph: float = 0.0,
    asymmetry: float = 0.0,
    harmonic_2: float = 0.0,
    harmonic_3: float = 0.0,
    harmonic_4: float = 0.0,
    harmonic_5: float = 0.0,
    ring_ratio: float = 0.0,
    ring_mix: float = 0.0,
    bit_depth: float = 24.0,
    sample_hold: float = 0.0,
) -> np.ndarray:
    wave = _norm_wave(wave, default="sine")
    wave_alt = _norm_wave(wave_alt, default="sine")
    wave_blend = float(np.clip(wave_blend, 0.0, 1.0))
    morph = float(np.clip(morph, -1.0, 1.0))
    asymmetry = float(np.clip(asymmetry, -1.0, 1.0))
    harmonic_2 = float(np.clip(harmonic_2, -1.0, 1.0))
    harmonic_3 = float(np.clip(harmonic_3, -1.0, 1.0))
    harmonic_4 = float(np.clip(harmonic_4, -1.0, 1.0))
    harmonic_5 = float(np.clip(harmonic_5, -1.0, 1.0))
    ring_ratio = float(np.clip(ring_ratio, 0.0, 24.0))
    ring_mix = float(np.clip(ring_mix, 0.0, 1.0))
    bit_depth = float(np.clip(bit_depth, 2.0, 24.0))
    sample_hold = float(np.clip(sample_hold, 0.0, 1.0))

    freq = float(max(0.01, freq))
    sr = int(max(1, sr))
    n = int(t.shape[0])

    unison = int(np.clip(unison, 1, 16))
    detune_cents = float(np.clip(detune_cents, 0.0, 120.0))
    spread = float(np.clip(spread, 0.0, 1.0))
    pwm = float(np.clip(pwm, 0.01, 0.99))
    sync = float(np.clip(sync, 0.0, 1.0))
    sync_ratio = float(max(1.0, sync_ratio))
    fm_ratio = float(np.clip(fm_ratio, 0.0, 24.0))
    fm_index = float(np.clip(fm_index, 0.0, 60.0))
    pm_amount = float(np.clip(pm_amount, 0.0, 0.85))
    pd_amount = float(np.clip(pd_amount, -0.95, 0.95))

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    if unison == 1:
        detunes = np.asarray([0.0], dtype=np.float32)
    else:
        idx = np.arange(unison, dtype=np.float32)
        det = idx - (unison - 1) / 2.0
        det /= np.max(np.abs(det)) + 1.0e-8
        detunes = np.sign(det) * (np.abs(det) ** 1.2) * detune_cents * spread

    phase_offsets = rng.uniform(0.0, 1.0, size=unison).astype(np.float32) * (0.35 * spread)
    out = np.zeros(n, dtype=np.float32)

    for voice_i in range(unison):
        cents = float(detunes[voice_i])
        voice_freq = freq * (2.0 ** (cents / 1200.0))
        dt = voice_freq / float(sr)

        phase = (voice_freq * t).astype(np.float32)

        if sync > 0.0:
            base = phase % 1.0
            synced = (phase * sync_ratio) % 1.0
            phase01 = (1.0 - sync) * base + sync * synced
        else:
            phase01 = phase % 1.0

        phase01 = (phase01 + phase_offsets[voice_i]) % 1.0

        if abs(pd_amount) > 1.0e-8:
            phase01 = _phase_distort(phase01, pd_amount)

        if fm_ratio > 0.0 and fm_index > 0.0:
            fm = np.sin(_TWOPI * (voice_freq * fm_ratio) * t).astype(np.float32)
            phase01 = (phase01 + ((fm_index ** 1.03) * 0.055) * fm) % 1.0

        if pm_amount > 0.0:
            pm = np.sin(_TWOPI * (voice_freq * 0.5) * t).astype(np.float32)
            phase01 = (phase01 + (((pm_amount ** 0.9) * 1.20) / _TWOPI) * pm) % 1.0

        if abs(asymmetry) > 1.0e-8:
            phase01 = _asym_phase(phase01, asymmetry)

        y = _osc_wave_from_phase(wave, phase01, dt, pwm=pwm, rng=rng)

        if wave_blend > 1.0e-8:
            y_alt = _osc_wave_from_phase(wave_alt, phase01, dt, pwm=pwm, rng=rng)
            y = ((1.0 - wave_blend) * y + wave_blend * y_alt).astype(np.float32)

        if abs(morph) > 1.0e-8:
            if morph > 0.0:
                y = ((1.0 - morph) * y + morph * np.tanh(y * (1.0 + 5.0 * morph))).astype(np.float32)
            else:
                m = abs(morph)
                y = ((1.0 - m) * y + m * np.sign(y) * np.sqrt(np.abs(y) + 1.0e-8)).astype(np.float32)

        out += y

    out /= float(unison)

    if any(abs(v) > 1.0e-8 for v in (harmonic_2, harmonic_3, harmonic_4, harmonic_5)):
        phase_base = (freq * t) % 1.0
        extra = (
            harmonic_2 * np.sin(_TWOPI * 2.0 * phase_base + 0.07)
            + harmonic_3 * np.sin(_TWOPI * 3.0 * phase_base + 0.13)
            + harmonic_4 * np.sin(_TWOPI * 4.0 * phase_base + 0.19)
            + harmonic_5 * np.sin(_TWOPI * 5.0 * phase_base + 0.29)
        ).astype(np.float32)
        out = (out + 0.35 * extra).astype(np.float32)

    if ring_mix > 1.0e-8 and ring_ratio > 1.0e-8:
        ring = np.sin(_TWOPI * (freq * ring_ratio) * t).astype(np.float32)
        out = ((1.0 - ring_mix) * out + ring_mix * (out * ring)).astype(np.float32)

    out = _waveshape(out, drive=drive, fold=fold)
    out = _tilt_eq(out, tilt=tilt)
    out = _bitcrush_mono(out, bit_depth)
    out = _sample_hold_mono(out, sample_hold)

    return out.astype(np.float32, copy=False)


# ============================================================================
# Stereo / envelopes / FX helpers
# ============================================================================

def _pan_stereo(x: np.ndarray, pan: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]

    pan = float(np.clip(pan, -1.0, 1.0))
    left = np.sqrt(0.5 * (1.0 - pan))
    right = np.sqrt(0.5 * (1.0 + pan))

    return np.stack([x * left, x * right], axis=1).astype(np.float32, copy=False)


def _adsr_env(
    n: int,
    sr: int,
    a: float,
    d: float,
    s: float,
    r: float,
    curve: float = 0.55,
) -> np.ndarray:
    n = int(max(1, n))
    sr = int(max(1, sr))

    a = max(0.0, float(a))
    d = max(0.0, float(d))
    s = float(np.clip(s, 0.0, 1.0))
    r = max(0.0, float(r))
    curve = float(np.clip(curve, 0.15, 2.5))

    env = np.ones(n, dtype=np.float32)

    a_n = int(round(a * sr))
    d_n = int(round(d * sr))
    r_n = int(round(r * sr))

    a_n = max(0, min(a_n, n))
    d_n = max(0, min(d_n, n - a_n))
    r_n = max(0, min(r_n, n))

    if a_n > 1:
        x = np.linspace(0.0, 1.0, a_n, dtype=np.float32)
        env[:a_n] = x ** (1.0 / curve)
    elif a_n == 1:
        env[0] = 1.0

    idx = a_n

    if d_n > 1:
        x = np.linspace(0.0, 1.0, d_n, dtype=np.float32)
        env[idx:idx + d_n] = (1.0 - s) * ((1.0 - x) ** curve) + s
    elif d_n == 1 and idx < n:
        env[idx] = s

    idx += d_n

    if idx < n:
        env[idx:] = s

    if r_n > 1:
        x = np.linspace(0.0, 1.0, r_n, dtype=np.float32)
        start_idx = max(0, n - r_n)
        release_start = float(env[start_idx]) if start_idx < n else float(env[-1])
        if release_start <= 1.0e-8:
            release_start = float(np.max(env))
        env[-r_n:] = release_start * (1.0 - x) ** (1.0 / curve)
    elif r_n == 1:
        env[-1] = 0.0

    return np.clip(env, 0.0, 1.0).astype(np.float32, copy=False)


def _exp_decay(n: int, sr: int, t60: float) -> np.ndarray:
    n = int(max(1, n))
    sr = int(max(1, sr))
    t60 = float(max(1.0e-4, t60))
    t = np.arange(n, dtype=np.float32) / float(sr)
    return np.exp(np.log(0.001) * (t / t60)).astype(np.float32)


def _soft_limiter_stereo(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("soft_clip_normalize"):
            return NATIVE_DSP.soft_clip_normalize(
                y,
                ceiling=float(ceiling),
                peak=float(ceiling),
                only_if_over=True,
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    ceiling = float(max(0.05, ceiling))
    y = np.tanh(y / ceiling) * ceiling

    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx > ceiling and mx > 1.0e-9:
        y *= ceiling / mx

    return y.astype(np.float32, copy=False)


def _one_pole_lowpass_stereo(x: np.ndarray, sr: int, cutoff_hz: float, wet: float = 1.0) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    sr = int(max(1, sr))
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    wet = float(np.clip(wet, 0.0, 1.0))

    if wet <= 1.0e-8:
        return x

    rc = 1.0 / (_TWOPI * cutoff_hz)
    dt = 1.0 / float(sr)
    alpha = dt / (rc + dt)

    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]

    for i in range(1, x.shape[0]):
        y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])

    return ((1.0 - wet) * x + wet * y).astype(np.float32, copy=False)


def _one_pole_highpass_stereo(x: np.ndarray, sr: int, cutoff_hz: float, wet: float = 1.0) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    lp = _one_pole_lowpass_stereo(x, sr, cutoff_hz, wet=1.0)
    hp = x - lp
    wet = float(np.clip(wet, 0.0, 1.0))
    return ((1.0 - wet) * x + wet * hp).astype(np.float32, copy=False)


def _delay_stereo(
    x: np.ndarray,
    sr: int,
    delay_ms: float,
    feedback: float,
    wet: float,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("delay"):
            return NATIVE_DSP.delay(
                x,
                sample_rate=int(sr),
                delay_ms=float(delay_ms),
                feedback=float(feedback),
                wet=float(wet),
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    delay_n = max(1, int(round(float(delay_ms) * 0.001 * int(sr))))
    feedback = float(np.clip(feedback, 0.0, 0.98))
    wet = float(wet)

    y = x.copy()

    for i in range(delay_n, y.shape[0]):
        y[i] += y[i - delay_n] * feedback

    return (x + y * wet).astype(np.float32, copy=False)


def _chorus_stereo(
    x: np.ndarray,
    sr: int,
    *,
    rate_hz: float,
    depth_ms: float,
    mix: float,
    seed: int,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    mix = float(np.clip(mix, 0.0, 1.0))

    if mix <= 1.0e-8:
        return x

    sr = int(max(1, sr))
    n = x.shape[0]
    t = np.arange(n, dtype=np.float32) / float(sr)

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    rate = float(np.clip(rate_hz, 0.02, 8.0))
    depth = int(round(float(np.clip(depth_ms, 0.1, 30.0)) * 0.001 * sr))
    base = int(round(0.006 * sr))
    max_delay = max(4, base + depth + 4)

    phase_l = float(rng.uniform(0.0, _TWOPI))
    phase_r = float(rng.uniform(0.0, _TWOPI))

    lfo_l = np.sin(_TWOPI * rate * t + phase_l)
    lfo_r = np.sin(_TWOPI * rate * t + phase_r)

    def _delay_channel(inp: np.ndarray, lfo: np.ndarray) -> np.ndarray:
        out = np.zeros_like(inp, dtype=np.float32)
        for i in range(n):
            d = base + int(round((0.5 + 0.5 * float(lfo[i])) * depth))
            d = int(np.clip(d, 1, max_delay - 1))
            j = i - d
            out[i] = inp[j] if j >= 0 else 0.0
        return out

    wet_l = _delay_channel(x[:, 0], lfo_l)
    wet_r = _delay_channel(x[:, 1], lfo_r)
    wet = np.stack([wet_l, wet_r], axis=1).astype(np.float32)

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


def _simple_reverb_stereo(
    x: np.ndarray,
    sr: int,
    *,
    mix: float,
    room: float,
    predelay_ms: float = 12.0,
    damp_hz: float = 7000.0,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    mix = float(np.clip(mix, 0.0, 1.0))

    if mix <= 1.0e-8:
        return x

    sr = int(max(1, sr))
    room = float(np.clip(room, 0.0, 1.0))
    predelay = int(round(float(np.clip(predelay_ms, 0.0, 100.0)) * 0.001 * sr))

    n = x.shape[0]
    wet_in = np.zeros_like(x, dtype=np.float32)

    if predelay > 0 and predelay < n:
        wet_in[predelay:] = x[:-predelay]
    else:
        wet_in = x.copy()

    delays = [
        int(sr * (0.029 + 0.020 * room)),
        int(sr * (0.037 + 0.025 * room)),
        int(sr * (0.041 + 0.030 * room)),
        int(sr * (0.053 + 0.033 * room)),
    ]
    delays = [max(8, d) for d in delays]

    feedback = 0.58 + 0.32 * room
    wet = np.zeros_like(wet_in, dtype=np.float32)

    for ch in (0, 1):
        acc = np.zeros(n, dtype=np.float32)

        for d in delays:
            buf = np.zeros(d, dtype=np.float32)
            idx = 0
            out = np.zeros(n, dtype=np.float32)

            for i in range(n):
                delayed = buf[idx]
                out[i] = delayed
                buf[idx] = wet_in[i, ch] + delayed * feedback
                idx = (idx + 1) % d

            acc += out

        wet[:, ch] = acc / float(len(delays))

    wet = _one_pole_lowpass_stereo(wet, sr, damp_hz, wet=1.0)
    wet = np.tanh(wet * 1.2).astype(np.float32)

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


def _body_resonance_stereo(
    x: np.ndarray,
    sr: int,
    amount: float,
    modes: List[Tuple[float, float]],
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    amount = float(np.clip(amount, 0.0, 1.0))

    if amount <= 1.0e-8:
        return x

    acc = np.zeros_like(x, dtype=np.float32)

    for cutoff, gain in modes:
        acc += float(gain) * _one_pole_lowpass_stereo(x, sr, cutoff, wet=1.0)

    return (x + amount * acc).astype(np.float32, copy=False)


def _filtered_noise_mono(n: int, sr: int, seed: int, lowpass_hz: float) -> np.ndarray:
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    nz = rng.normal(0.0, 1.0, size=max(1, int(n))).astype(np.float32)
    st = _pan_stereo(nz, 0.0)
    st = _one_pole_lowpass_stereo(st, sr, lowpass_hz, wet=1.0)
    return st[:, 0].astype(np.float32, copy=False)


# ============================================================================
# Shared synth params
# ============================================================================

_SYNTH_PARAM_INFO: Dict[str, Dict[str, Any]] = {
    "wave": {"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
    "waveform": {"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
    "wave_alt": {"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
    "wave_blend": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},

    "amp": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
    "gain": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
    "master_gain": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},

    "attack": {"type": "float", "default": 0.010, "min": 0.0, "max": 3.0, "step": 0.001},
    "decay": {"type": "float", "default": 0.120, "min": 0.0, "max": 6.0, "step": 0.001},
    "sustain": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
    "release": {"type": "float", "default": 0.180, "min": 0.0, "max": 8.0, "step": 0.001},
    "env_curve": {"type": "float", "default": 0.55, "min": 0.15, "max": 2.5, "step": 0.01},

    "unison": {"type": "int", "default": 1, "min": 1, "max": 16, "step": 1},
    "detune_cents": {"type": "float", "default": 0.0, "min": 0.0, "max": 120.0, "step": 0.5},
    "spread": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "pwm": {"type": "float", "default": 0.50, "min": 0.01, "max": 0.99, "step": 0.01},
    "sync": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "sync_ratio": {"type": "float", "default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05},

    "fm_ratio": {"type": "float", "default": 0.0, "min": 0.0, "max": 24.0, "step": 0.05},
    "fm_index": {"type": "float", "default": 0.0, "min": 0.0, "max": 60.0, "step": 0.1},
    "pm_amount": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.85, "step": 0.01},
    "pd_amount": {"type": "float", "default": 0.0, "min": -0.95, "max": 0.95, "step": 0.01},

    "drive": {"type": "float", "default": 1.0, "min": 0.05, "max": 24.0, "step": 0.05},
    "fold": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "tilt": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "morph": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "asymmetry": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},

    "harmonic_2": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "harmonic_3": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "harmonic_4": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "harmonic_5": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},

    "ring_ratio": {"type": "float", "default": 0.0, "min": 0.0, "max": 24.0, "step": 0.05},
    "ring_mix": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "bit_depth": {"type": "float", "default": 24.0, "min": 2.0, "max": 24.0, "step": 1.0},
    "sample_hold": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},

    "cutoff_hz": {"type": "float", "default": 16000.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    "res": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "tone": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},

    "chorus_mix": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "chorus_rate": {"type": "float", "default": 0.35, "min": 0.02, "max": 8.0, "step": 0.05},
    "chorus_depth_ms": {"type": "float", "default": 7.0, "min": 0.1, "max": 30.0, "step": 0.1},

    "reverb_mix": {"type": "float", "default": 0.0, "min": 0.0, "max": 0.85, "step": 0.01},
    "reverb_room": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
    "reverb_predelay_ms": {"type": "float", "default": 12.0, "min": 0.0, "max": 100.0, "step": 1.0},
    "reverb_damp_hz": {"type": "float", "default": 7000.0, "min": 500.0, "max": 18000.0, "step": 100.0},

    "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
}


_POST_SYNTH_PARAM_INFO = {
    "post_drive": {"type": "float", "default": 1.0, "min": 0.05, "max": 12.0, "step": 0.05},
    "post_fold": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "post_tilt": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "post_ring_ratio": {"type": "float", "default": 0.0, "min": 0.0, "max": 24.0, "step": 0.05},
    "post_ring_mix": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "post_bit_depth": {"type": "float", "default": 24.0, "min": 2.0, "max": 24.0, "step": 1.0},
    "post_sample_hold": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "tremolo_mix": {"type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "tremolo_rate": {"type": "float", "default": 5.0, "min": 0.02, "max": 30.0, "step": 0.05},
}


def _synth_schema(**overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    schema = {k: dict(v) for k, v in _SYNTH_PARAM_INFO.items()}

    for k, v in _POST_SYNTH_PARAM_INFO.items():
        schema.setdefault(k, dict(v))

    for k, v in overrides.items():
        if k in schema and isinstance(v, dict):
            schema[k].update(v)
        else:
            schema[k] = dict(v)

    return schema


def _apply_extra_post_controls_stereo(
    y: np.ndarray,
    sr: int,
    params: Dict[str, Any],
    *,
    freq: float = 440.0,
    seed: int = 0,
) -> np.ndarray:
    y = ensure_stereo(y).astype(np.float32, copy=False)
    n = y.shape[0]

    if n <= 0:
        return y

    post_drive = float(params.get("post_drive", 1.0))
    post_fold = float(params.get("post_fold", 0.0))
    post_tilt = float(params.get("post_tilt", 0.0))
    post_ring_mix = float(np.clip(params.get("post_ring_mix", 0.0), 0.0, 1.0))
    post_ring_ratio = float(max(0.0, params.get("post_ring_ratio", 0.0)))
    tremolo_mix = float(np.clip(params.get("tremolo_mix", 0.0), 0.0, 1.0))
    tremolo_rate = float(np.clip(params.get("tremolo_rate", 5.0), 0.02, 30.0))
    post_bits = float(params.get("post_bit_depth", 24.0))
    post_hold = float(params.get("post_sample_hold", 0.0))

    if abs(post_drive - 1.0) > 1.0e-8 or post_fold > 1.0e-8:
        y[:, 0] = _waveshape(y[:, 0], post_drive, post_fold)
        y[:, 1] = _waveshape(y[:, 1], post_drive, post_fold)

    if abs(post_tilt) > 1.0e-8:
        y[:, 0] = _tilt_eq(y[:, 0], post_tilt)
        y[:, 1] = _tilt_eq(y[:, 1], post_tilt)

    if post_ring_mix > 1.0e-8 and post_ring_ratio > 1.0e-8:
        t = np.arange(n, dtype=np.float32) / float(max(1, sr))
        ring = np.sin(_TWOPI * float(freq) * post_ring_ratio * t).astype(np.float32)
        y = ((1.0 - post_ring_mix) * y + post_ring_mix * (y * ring[:, None])).astype(np.float32)

    if tremolo_mix > 1.0e-8:
        rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
        phase = float(rng.uniform(0.0, _TWOPI))
        t = np.arange(n, dtype=np.float32) / float(max(1, sr))
        trem = 0.5 + 0.5 * np.sin(_TWOPI * tremolo_rate * t + phase).astype(np.float32)
        gain = (1.0 - tremolo_mix) + tremolo_mix * trem
        y = (y * gain[:, None]).astype(np.float32)

    if post_bits < 23.5:
        y[:, 0] = _bitcrush_mono(y[:, 0], post_bits)
        y[:, 1] = _bitcrush_mono(y[:, 1], post_bits)

    if post_hold > 1.0e-8:
        y[:, 0] = _sample_hold_mono(y[:, 0], post_hold)
        y[:, 1] = _sample_hold_mono(y[:, 1], post_hold)

    return y.astype(np.float32, copy=False)


def _render_synth_voice(
    payload: Any,
    raw_params: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]],
    *,
    default_wave: str,
    default_amp: float,
    character: str,
) -> Tuple[AudioBuffer, Dict[str, Any]]:
    params = _merge_params(schema, raw_params)

    freq, dur, sr, vel = _voice_payload(payload, params)
    n = _frames(dur, sr)
    t = np.arange(n, dtype=np.float32) / float(sr)

    seed = int(params.get("seed", 0))
    wave = params.get("wave", params.get("waveform", default_wave))
    amp = float(params.get("amp", params.get("gain", params.get("master_gain", default_amp))))

    env = _adsr_env(
        n,
        sr,
        a=float(params.get("attack", 0.010)),
        d=float(params.get("decay", 0.120)),
        s=float(params.get("sustain", 0.75)),
        r=float(params.get("release", 0.180)),
        curve=float(params.get("env_curve", 0.55)),
    )

    vel = float(np.clip(vel, 0.0, 2.0))
    vel_gain = 0.35 + 0.85 * min(1.0, vel)
    vel_bright = 0.65 + 0.70 * min(1.0, vel)

    x = osc_advanced(
        wave=str(wave),
        t=t,
        freq=freq,
        sr=sr,
        unison=int(params.get("unison", 1)),
        detune_cents=float(params.get("detune_cents", 0.0)),
        spread=float(params.get("spread", 0.5)),
        pwm=float(params.get("pwm", 0.5)),
        sync=float(params.get("sync", 0.0)),
        sync_ratio=float(params.get("sync_ratio", 2.0)),
        fm_ratio=float(params.get("fm_ratio", 0.0)),
        fm_index=float(params.get("fm_index", 0.0)),
        pm_amount=float(params.get("pm_amount", 0.0)),
        pd_amount=float(params.get("pd_amount", 0.0)),
        drive=float(params.get("drive", 1.0)),
        fold=float(params.get("fold", 0.0)),
        tilt=float(params.get("tilt", 0.0)),
        seed=seed,
        wave_alt=str(params.get("wave_alt", "sine")),
        wave_blend=float(params.get("wave_blend", 0.0)),
        morph=float(params.get("morph", 0.0)),
        asymmetry=float(params.get("asymmetry", 0.0)),
        harmonic_2=float(params.get("harmonic_2", 0.0)) * vel_bright,
        harmonic_3=float(params.get("harmonic_3", 0.0)) * vel_bright,
        harmonic_4=float(params.get("harmonic_4", 0.0)) * vel_bright,
        harmonic_5=float(params.get("harmonic_5", 0.0)) * vel_bright,
        ring_ratio=float(params.get("ring_ratio", 0.0)),
        ring_mix=float(params.get("ring_mix", 0.0)) * vel_bright,
        bit_depth=float(params.get("bit_depth", 24.0)),
        sample_hold=float(params.get("sample_hold", 0.0)),
    )

    rng = np.random.RandomState(seed & 0xFFFFFFFF)

    if character == "piano":
        hammer = rng.normal(0.0, 1.0, n).astype(np.float32)
        hammer *= _exp_decay(n, sr, 0.035)
        x = 0.88 * x * _exp_decay(n, sr, float(params.get("body_decay", 2.6))) + 0.045 * hammer

    elif character == "guitar":
        x = x * _exp_decay(n, sr, float(params.get("decay", 2.2)))

    elif character == "bell":
        ratio = float(params.get("fm_ratio", 3.0))
        index = float(params.get("fm_index", 8.0))
        mod = np.sin(_TWOPI * freq * ratio * t).astype(np.float32)
        bell = np.sin(_TWOPI * freq * t + index * mod).astype(np.float32)
        bell *= _exp_decay(n, sr, float(params.get("decay", 3.5)))
        shimmer = float(np.clip(params.get("shimmer", 0.25), 0.0, 1.0))
        x = (1.0 - shimmer) * x * _exp_decay(n, sr, 2.2) + shimmer * bell

    elif character == "brass":
        swell = 1.0 - np.exp(-t / max(0.002, float(params.get("attack", 0.045))))
        x = np.tanh((x * swell) * (1.4 + float(params.get("brightness", 0.45)))).astype(np.float32)

    elif character == "flute":
        breath = _filtered_noise_mono(n, sr, seed + 17, lowpass_hz=5000.0)
        breath_amount = float(np.clip(params.get("breath", 0.09), 0.0, 1.0))
        x = 0.92 * x + breath_amount * 0.08 * breath

    elif character == "clarinet":
        reed = np.sin(_TWOPI * freq * 3.0 * t).astype(np.float32)
        x = 0.70 * x + 0.30 * reed

    elif character == "string":
        slow = 0.5 + 0.5 * np.sin(_TWOPI * 0.35 * t + seed).astype(np.float32)
        x = x * (0.85 + 0.15 * slow)

    x = x * env * amp * vel_gain
    y = _pan_stereo(x, float(params.get("pan", 0.0)))

    cutoff = float(params.get("cutoff_hz", 16000.0))
    tone = float(params.get("tone", 0.0))

    if tone < 0.0:
        cutoff *= 1.0 + tone * 0.85
    elif tone > 0.0:
        y = np.tanh(y * (1.0 + tone * 1.8)).astype(np.float32)

    y = _one_pole_lowpass_stereo(y, sr, cutoff, wet=1.0)

    body = float(params.get("body", 0.0))
    if body > 1.0e-8:
        y = _body_resonance_stereo(
            y,
            sr,
            amount=body,
            modes=[(freq * 1.0, 0.45), (freq * 2.0, 0.22), (freq * 3.0, 0.12)],
        )

    chorus_mix = float(params.get("chorus_mix", 0.0))
    if chorus_mix > 1.0e-8:
        y = _chorus_stereo(
            y,
            sr,
            rate_hz=float(params.get("chorus_rate", 0.35)),
            depth_ms=float(params.get("chorus_depth_ms", 7.0)),
            mix=chorus_mix,
            seed=seed + 101,
        )

    reverb_mix = float(params.get("reverb_mix", 0.0))
    if reverb_mix > 1.0e-8:
        y = _simple_reverb_stereo(
            y,
            sr,
            mix=reverb_mix,
            room=float(params.get("reverb_room", 0.55)),
            predelay_ms=float(params.get("reverb_predelay_ms", 12.0)),
            damp_hz=float(params.get("reverb_damp_hz", 7000.0)),
        )

    y = _apply_extra_post_controls_stereo(y, sr, params, freq=freq, seed=seed + 607)
    y = _soft_limiter_stereo(y, ceiling=0.98)

    return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({
        "engine": f"numpy_{character}",
        "wave": _norm_wave(wave, default_wave),
        "params_received": True,
    })


# ============================================================================
# Instruments
# ============================================================================

class SynthKeys(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "saw", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "saw", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.008, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.120, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.180, "min": 0.0, "max": 8.0, "step": 0.001},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="saw",
            default_amp=0.30,
            character="synth",
        )


class LeadSynth(SynthKeys):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "square", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "square", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.32, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.32, "min": 0.0, "max": 1.5, "step": 0.01},
        unison={"type": "int", "default": 3, "min": 1, "max": 16, "step": 1},
        detune_cents={"type": "float", "default": 7.0, "min": 0.0, "max": 120.0, "step": 0.5},
        spread={"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
        attack={"type": "float", "default": 0.003, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.08, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.16, "min": 0.0, "max": 8.0, "step": 0.001},
        cutoff_hz={"type": "float", "default": 9000.0, "min": 20.0, "max": 22000.0, "step": 50.0},
        res={"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        chorus_mix={"type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
        reverb_mix={"type": "float", "default": 0.12, "min": 0.0, "max": 0.85, "step": 0.01},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="square",
            default_amp=0.32,
            character="synth",
        )


class PianoKeys(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "harmonic", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "harmonic", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.38, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.38, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.002, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.20, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.75, "min": 0.0, "max": 8.0, "step": 0.001},
        body={"type": "float", "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01},
        body_decay={"type": "float", "default": 2.6, "min": 0.05, "max": 12.0, "step": 0.05},
        cutoff_hz={"type": "float", "default": 11000.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="harmonic",
            default_amp=0.38,
            character="piano",
        )


class GuitarPluck(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.5, "step": 0.01},
        "gain": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.5, "step": 0.01},
        "decay": {"type": "float", "default": 2.2, "min": 0.05, "max": 12.0, "step": 0.05},
        "brightness": {"type": "float", "default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01},
        "pick": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01},
        "noise": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        "chorus_mix": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.12, "min": 0.0, "max": 0.85, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        freq, dur, sr, vel = _voice_payload(payload, params)
        n = _frames(dur, sr)

        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        amp = float(params.get("amp", params.get("gain", 0.42)))
        decay = float(params.get("decay", 2.2))
        brightness = float(np.clip(params.get("brightness", 0.58), 0.0, 1.0))
        pick = float(np.clip(params.get("pick", 0.45), 0.0, 1.0))
        body = float(np.clip(params.get("body", 0.38), 0.0, 1.0))
        noise = float(np.clip(params.get("noise", 0.10), 0.0, 1.0))
        pan = float(params.get("pan", 0.0))

        delay = max(2, int(round(sr / max(20.0, freq))))
        ring = rng.uniform(-1.0, 1.0, size=delay).astype(np.float32)
        ring += pick * np.sin(_TWOPI * np.arange(delay, dtype=np.float32) / float(delay)).astype(np.float32)
        ring *= 0.65 + 0.35 * brightness

        x = np.zeros(n, dtype=np.float32)
        pos = 0
        fb = np.exp(np.log(0.001) / (max(0.05, decay) * sr / float(delay)))

        for i in range(n):
            nxt = (pos + 1) % delay
            y0 = ring[pos]
            avg = 0.5 * (ring[pos] + ring[nxt])
            ring[pos] = (brightness * y0 + (1.0 - brightness) * avg) * fb
            x[i] = y0
            pos = nxt

        if noise > 1.0e-8:
            hit_n = min(n, max(16, int(round(0.02 * sr))))
            hit = rng.normal(0.0, 1.0, size=hit_n).astype(np.float32)
            x[:hit_n] += noise * 0.12 * hit * _exp_decay(hit_n, sr, 0.025)

        vel_gain = 0.35 + 0.85 * float(np.clip(vel, 0.0, 1.0))
        x = np.tanh(x * (1.0 + pick * 1.5)).astype(np.float32)
        x *= amp * vel_gain

        y = _pan_stereo(x, pan)
        y = _body_resonance_stereo(
            y,
            sr,
            amount=body,
            modes=[(freq, 0.50), (freq * 2.0, 0.22), (freq * 3.0, 0.12)],
        )

        if float(params.get("chorus_mix", 0.05)) > 1.0e-8:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.24,
                depth_ms=4.5,
                mix=float(params.get("chorus_mix", 0.05)),
                seed=seed + 11,
            )

        if float(params.get("reverb_mix", 0.12)) > 1.0e-8:
            y = _simple_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.12)),
                room=float(params.get("reverb_room", 0.45)),
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_guitar_pluck", "params_received": True})


class BellFM(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.35, "min": 0.0, "max": 1.5, "step": 0.01},
        decay={"type": "float", "default": 3.5, "min": 0.05, "max": 12.0, "step": 0.05},
        fm_ratio={"type": "float", "default": 3.0, "min": 0.0, "max": 24.0, "step": 0.05},
        fm_index={"type": "float", "default": 8.0, "min": 0.0, "max": 60.0, "step": 0.1},
        shimmer={"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        body={"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
        reverb_mix={"type": "float", "default": 0.30, "min": 0.0, "max": 0.85, "step": 0.01},
        reverb_room={"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="sine",
            default_amp=0.35,
            character="bell",
        )


class BrassSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "saw", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "saw", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.36, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.36, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.045, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.20, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.78, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.24, "min": 0.0, "max": 8.0, "step": 0.001},
        brightness={"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        unison={"type": "int", "default": 4, "min": 1, "max": 16, "step": 1},
        detune_cents={"type": "float", "default": 7.0, "min": 0.0, "max": 120.0, "step": 0.5},
        drive={"type": "float", "default": 1.8, "min": 0.05, "max": 24.0, "step": 0.05},
        cutoff_hz={"type": "float", "default": 7500.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="saw",
            default_amp=0.36,
            character="brass",
        )


class FluteSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "sine", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.055, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.12, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.84, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.28, "min": 0.0, "max": 8.0, "step": 0.001},
        breath={"type": "float", "default": 0.09, "min": 0.0, "max": 1.0, "step": 0.01},
        harmonic_2={"type": "float", "default": 0.16, "min": -1.0, "max": 1.0, "step": 0.01},
        cutoff_hz={"type": "float", "default": 6000.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="sine",
            default_amp=0.30,
            character="flute",
        )


class ClarinetSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "nasal", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "nasal", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.32, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.32, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.035, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.12, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.80, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.22, "min": 0.0, "max": 8.0, "step": 0.001},
        cutoff_hz={"type": "float", "default": 4200.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="nasal",
            default_amp=0.32,
            character="clarinet",
        )


class StringPad(BaseBlock):
    KIND = "instrument"
    PARAMS = _synth_schema(
        wave={"type": "choice", "default": "organ", "choices": _WAVE_CHOICES},
        waveform={"type": "choice", "default": "organ", "choices": _WAVE_CHOICES},
        amp={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        gain={"type": "float", "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.01},
        attack={"type": "float", "default": 0.35, "min": 0.0, "max": 3.0, "step": 0.001},
        decay={"type": "float", "default": 0.25, "min": 0.0, "max": 6.0, "step": 0.001},
        sustain={"type": "float", "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
        release={"type": "float", "default": 0.75, "min": 0.0, "max": 8.0, "step": 0.001},
        unison={"type": "int", "default": 6, "min": 1, "max": 16, "step": 1},
        detune_cents={"type": "float", "default": 12.0, "min": 0.0, "max": 120.0, "step": 0.5},
        chorus_mix={"type": "float", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01},
        reverb_mix={"type": "float", "default": 0.22, "min": 0.0, "max": 0.85, "step": 0.01},
        cutoff_hz={"type": "float", "default": 8500.0, "min": 20.0, "max": 22000.0, "step": 50.0},
    )

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        return _render_synth_voice(
            payload,
            params,
            self.PARAMS,
            default_wave="organ",
            default_amp=0.30,
            character="string",
        )


# ============================================================================
# FX blocks
# ============================================================================

class Gain(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = ensure_stereo(buf.data) * float(params.get("gain", 1.0))
        return AudioBuffer(_sanitize(y), int(buf.sr)), _meta({"engine": "numpy_gain"})


class Delay(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "delay_ms": {"type": "float", "default": 250.0, "min": 1.0, "max": 5000.0, "step": 1.0},
        "feedback": {"type": "float", "default": 0.25, "min": 0.0, "max": 0.98, "step": 0.01},
        "wet": {"type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = _delay_stereo(
            buf.data,
            int(buf.sr),
            delay_ms=float(params.get("delay_ms", 250.0)),
            feedback=float(params.get("feedback", 0.25)),
            wet=float(params.get("wet", 0.35)),
        )
        return AudioBuffer(_soft_limiter_stereo(y, 0.98), int(buf.sr)), _meta({"engine": "numpy_delay"})


class OnePoleLP(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 6000.0, "min": 10.0, "max": 22000.0, "step": 10.0},
        "cutoff": {"type": "float", "default": 6000.0, "min": 10.0, "max": 22000.0, "step": 10.0},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        cutoff = float(params.get("cutoff_hz", params.get("cutoff", 6000.0)))
        y = _one_pole_lowpass_stereo(buf.data, int(buf.sr), cutoff, wet=float(params.get("wet", 1.0)))
        return AudioBuffer(_sanitize(y), int(buf.sr)), _meta({"engine": "numpy_lowpass"})


class Highpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 200.0, "min": 10.0, "max": 18000.0, "step": 10.0},
        "cutoff": {"type": "float", "default": 200.0, "min": 10.0, "max": 18000.0, "step": 10.0},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        cutoff = float(params.get("cutoff_hz", params.get("cutoff", 200.0)))
        y = _one_pole_highpass_stereo(buf.data, int(buf.sr), cutoff, wet=float(params.get("wet", 1.0)))
        return AudioBuffer(_sanitize(y), int(buf.sr)), _meta({"engine": "numpy_highpass"})


class Bandpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "low_hz": {"type": "float", "default": 200.0, "min": 10.0, "max": 18000.0, "step": 10.0},
        "high_hz": {"type": "float", "default": 6000.0, "min": 20.0, "max": 22000.0, "step": 10.0},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        x = ensure_stereo(buf.data)

        low = float(params.get("low_hz", 200.0))
        high = float(params.get("high_hz", 6000.0))
        if high <= low:
            high = low + 100.0

        lp_high = _one_pole_lowpass_stereo(x, int(buf.sr), high, wet=1.0)
        lp_low = _one_pole_lowpass_stereo(x, int(buf.sr), low, wet=1.0)
        bp = lp_high - lp_low

        wet = float(np.clip(params.get("wet", 1.0), 0.0, 1.0))
        y = (1.0 - wet) * x + wet * bp

        return AudioBuffer(_sanitize(y), int(buf.sr)), _meta({"engine": "numpy_bandpass"})


class SoftClip(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "drive": {"type": "float", "default": 1.2, "min": 0.05, "max": 24.0, "step": 0.05},
        "ceiling": {"type": "float", "default": 0.98, "min": 0.05, "max": 1.0, "step": 0.01},
        "normalize": {"type": "bool", "default": False},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))

        drive = float(params.get("drive", 1.2))
        ceiling = float(params.get("ceiling", 0.98))
        normalize = bool(params.get("normalize", False))

        y = np.tanh(ensure_stereo(buf.data) * max(0.0, drive)).astype(np.float32)

        if normalize:
            mx = float(np.max(np.abs(y))) if y.size else 0.0
            if mx > 1.0e-9:
                y *= ceiling / mx
        else:
            y = _soft_limiter_stereo(y, ceiling=ceiling)

        return AudioBuffer(y.astype(np.float32, copy=False), int(buf.sr)), _meta({"engine": "numpy_softclip"})


class SoundPolish(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "input_gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
        "drive": {"type": "float", "default": 0.65, "min": 0.0, "max": 8.0, "step": 0.05},
        "warmth": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        "brightness": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        "lowpass_hz": {"type": "float", "default": 18000.0, "min": 1000.0, "max": 22000.0, "step": 100.0},
        "width_mix": {"type": "float", "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01},
        "width_ms": {"type": "float", "default": 5.5, "min": 0.1, "max": 30.0, "step": 0.1},
        "output_gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
        "ceiling": {"type": "float", "default": 0.98, "min": 0.05, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        params = _merge_params(self.PARAMS, params)
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        sr = int(buf.sr)

        seed = int(params.get("seed", 0))
        y = ensure_stereo(buf.data).astype(np.float32, copy=False)
        y *= float(params.get("input_gain", 1.0))

        drive = float(params.get("drive", 0.65))
        if drive > 1.0e-8:
            y = np.tanh(y * (1.0 + drive)).astype(np.float32)

        warmth = float(np.clip(params.get("warmth", 0.28), 0.0, 1.0))
        brightness = float(np.clip(params.get("brightness", 0.10), 0.0, 1.0))

        if warmth > 1.0e-8:
            warm = _one_pole_lowpass_stereo(y, sr, 3500.0, wet=1.0)
            y = ((1.0 - warmth * 0.45) * y + (warmth * 0.45) * warm).astype(np.float32)

        if brightness > 1.0e-8:
            bright = _one_pole_highpass_stereo(y, sr, 2600.0, wet=1.0)
            y = (y + brightness * 0.18 * bright).astype(np.float32)

        lowpass_hz = float(params.get("lowpass_hz", 18000.0))
        y = _one_pole_lowpass_stereo(y, sr, cutoff_hz=float(np.clip(lowpass_hz, 1000.0, sr * 0.45)), wet=1.0)

        width_mix = float(params.get("width_mix", 0.08))
        if width_mix > 1.0e-8:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.11,
                depth_ms=float(params.get("width_ms", 5.5)),
                mix=width_mix,
                seed=seed + 401,
            )

        y *= float(params.get("output_gain", 1.0))
        y = _soft_limiter_stereo(y, ceiling=float(params.get("ceiling", 0.98)))

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_sound_polish"})


# ============================================================================
# Register blocks
# ============================================================================

BLOCKS.register("synth_keys", SynthKeys)
BLOCKS.register("synth", SynthKeys)
BLOCKS.register("native_synth", SynthKeys)

BLOCKS.register("piano_keys", PianoKeys)
BLOCKS.register("piano_key", PianoKeys)
BLOCKS.register("piano", PianoKeys)

BLOCKS.register("guitar_pluck", GuitarPluck)
BLOCKS.register("pluck", GuitarPluck)

BLOCKS.register("bell_fm", BellFM)
BLOCKS.register("bell", BellFM)

BLOCKS.register("lead_synth", LeadSynth)

BLOCKS.register("brass_synth", BrassSynth)
BLOCKS.register("brass", BrassSynth)

BLOCKS.register("flute_synth", FluteSynth)
BLOCKS.register("flute", FluteSynth)

BLOCKS.register("clarinet_synth", ClarinetSynth)
BLOCKS.register("clarinet", ClarinetSynth)

BLOCKS.register("string_pad", StringPad)
BLOCKS.register("strings", StringPad)
BLOCKS.register("pad", StringPad)

BLOCKS.register("gain", Gain)
BLOCKS.register("volume", Gain)

BLOCKS.register("delay", Delay)

BLOCKS.register("lowpass", OnePoleLP)
BLOCKS.register("filter_lowpass", OnePoleLP)

BLOCKS.register("highpass", Highpass)
BLOCKS.register("filter_highpass", Highpass)

BLOCKS.register("bandpass", Bandpass)
BLOCKS.register("filter_bandpass", Bandpass)

BLOCKS.register("softclip", SoftClip)
BLOCKS.register("clip", SoftClip)

BLOCKS.register("sound_polish", SoundPolish)
BLOCKS.register("polish", SoundPolish)