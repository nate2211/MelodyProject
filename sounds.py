from __future__ import annotations

import numpy as np
from typing import Any, Dict, Tuple, Optional, List

from pipeline import BaseBlock, BLOCKS, AudioBuffer, ensure_stereo

from melodyproject_native import (
    NATIVE_DSP,
    native_status,
    waveform_id,
    MP_WAVE_SQUARE,
    MP_WAVE_SAW,
    MP_WAVE_TRIANGLE,
)

# ============================================================================
# sounds.py
#
# Native-backed sound blocks with NumPy fallbacks.
#
# Includes:
#   Instruments:
#       synth_keys
#       lead_synth
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
    return bool(getattr(NATIVE_DSP, "available", False))


def _native_sounds_available() -> bool:
    return bool(getattr(NATIVE_DSP, "native_sounds_available", False))


def _native_has(name: str) -> bool:
    return bool(NATIVE_DSP is not None and hasattr(NATIVE_DSP, name))


def _meta(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {
        "native": _native_available(),
        "native_sounds_available": _native_sounds_available(),
        "native_status": native_status(),
    }
    if extra:
        out.update(extra)
    return out


def _voice_payload(payload: Any) -> Tuple[float, float, int, float]:
    freq = float(payload["freq"])
    dur = float(payload["duration"])
    sr = int(payload.get("sr", 48000))
    vel = float(payload.get("vel", payload.get("velocity", 1.0)))
    return freq, dur, sr, vel


def _frames(dur: float, sr: int) -> int:
    return max(1, int(round(float(dur) * int(sr))))


def _clamp(x: float, a: float, b: float) -> float:
    return float(np.clip(float(x), float(a), float(b)))


def _soft_native_or_numpy(
    x: np.ndarray,
    ceiling: float = 0.98,
    peak: float = 0.98,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("soft_clip_normalize"):
            return NATIVE_DSP.soft_clip_normalize(
                x,
                ceiling=float(ceiling),
                peak=float(peak),
                only_if_over=True,
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    ceiling = float(max(0.1, ceiling))
    peak = float(max(0.1, peak))

    y = np.tanh(x / ceiling) * ceiling
    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx > 1e-9 and mx > peak:
        y = y * (peak / mx)

    return y.astype(np.float32, copy=False)


def _soft_limiter_stereo(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    return _soft_native_or_numpy(x, ceiling=ceiling, peak=ceiling)


def _native_basic_note(
    freq: float,
    n: int,
    sr: int,
    *,
    wave: Any = "sine",
    vel: float = 1.0,
    amp: float = 0.25,
    attack: float = 0.004,
    release: float = 0.050,
    pan: float = 0.0,
) -> np.ndarray:
    try:
        midi = int(round(69.0 + 12.0 * np.log2(max(1e-9, float(freq)) / 440.0)))
        return NATIVE_DSP.render_synth_notes(
            np.asarray([midi], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([n], dtype=np.int32),
            np.asarray([vel], dtype=np.float32),
            total_frames=n,
            sample_rate=sr,
            waveform=waveform_id(wave),
            master_gain=float(amp),
            attack_seconds=float(attack),
            release_seconds=float(release),
            pan=float(pan),
        ).astype(np.float32, copy=False)
    except Exception:
        t = np.arange(n, dtype=np.float32) / float(sr)
        phase = _TWOPI * float(freq) * t
        wid = waveform_id(wave)

        if wid == MP_WAVE_SQUARE:
            mono = np.where(np.sin(phase) >= 0.0, 1.0, -1.0).astype(np.float32)
        elif wid == MP_WAVE_SAW:
            p = (float(freq) * t) % 1.0
            mono = (2.0 * p - 1.0).astype(np.float32)
        elif wid == MP_WAVE_TRIANGLE:
            p = (float(freq) * t) % 1.0
            mono = (2.0 * np.abs(2.0 * p - 1.0) - 1.0).astype(np.float32)
        else:
            mono = np.sin(phase).astype(np.float32)

        env = np.ones(n, dtype=np.float32)
        aN = max(0, min(n, int(round(float(attack) * sr))))
        rN = max(0, min(n, int(round(float(release) * sr))))

        if aN > 1:
            env[:aN] *= np.linspace(0.0, 1.0, aN, dtype=np.float32)
        if rN > 1:
            env[-rN:] *= np.linspace(1.0, 0.0, rN, dtype=np.float32)

        mono *= env * float(amp) * float(np.clip(vel, 0.0, 2.0))
        return _pan_stereo(mono, pan)


# ============================================================================
# Pitch / wave helpers
# ============================================================================

def _midi_from_freq(freq: float) -> float:
    freq = float(max(1e-6, freq))
    return 69.0 + 12.0 * np.log2(freq / 440.0)


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
}


def _norm_wave(w: Any, default: str = "sine") -> str:
    s = str(w or "").strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return _WAVE_ALIASES.get(s, default)


# ============================================================================
# PolyBLEP oscillator helpers
# ============================================================================

def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(t, dtype=np.float32)
    dt = float(max(1e-8, dt))

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
    m = float(np.max(np.abs(y))) + 1e-8
    return (y / m).astype(np.float32, copy=False)


def _phase_distort(phase01: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, -0.95, 0.95))
    if abs(a) < 1e-6:
        return phase01.astype(np.float32, copy=False)

    p = phase01.astype(np.float32, copy=False)
    bend = 0.5 + 0.45 * a
    bend = float(np.clip(bend, 0.05, 0.95))

    out = np.empty_like(p, dtype=np.float32)
    m = p < bend
    out[m] = (p[m] / bend) * 0.5
    out[~m] = 0.5 + ((p[~m] - bend) / (1.0 - bend)) * 0.5
    return out.astype(np.float32, copy=False)


def _waveshape(x: np.ndarray, drive: float, fold: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    try:
        st = ensure_stereo(x)
        if _native_has("waveshape"):
            y = NATIVE_DSP.waveshape(
                st,
                drive=float(drive),
                fold=float(fold),
                tilt=0.0,
            )
            if x.ndim == 1:
                return y[:, 0].astype(np.float32, copy=False)
            return y.astype(np.float32, copy=False)
    except Exception:
        pass

    drive = float(np.clip(drive, 0.05, 20.0))
    fold = float(np.clip(fold, 0.0, 1.0))

    d = 0.30 + (drive ** 1.15)
    y = np.tanh(x * d).astype(np.float32)

    if fold > 1e-8:
        k = 1.0 + 10.0 * (fold ** 0.9)
        z = y * k
        z = ((z + 1.0) % 4.0) - 2.0
        z = 2.0 - np.abs(z)
        y = (z - 1.0).astype(np.float32)

    return y.astype(np.float32, copy=False)


def _tilt_eq(x: np.ndarray, tilt: float) -> np.ndarray:
    tilt = float(np.clip(tilt, -1.0, 1.0))
    if abs(tilt) < 1e-6:
        return x.astype(np.float32, copy=False)

    dx = np.empty_like(x, dtype=np.float32)
    dx[0] = x[0]
    dx[1:] = x[1:] - x[:-1]

    y = np.tanh(x + 1.15 * tilt * dx).astype(np.float32)
    return y.astype(np.float32, copy=False)


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
) -> np.ndarray:
    wave = _norm_wave(wave, default="sine")
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
        det /= np.max(np.abs(det)) + 1e-8
        detunes = np.sign(det) * (np.abs(det) ** 1.2) * detune_cents * spread

    ph_off = rng.uniform(0.0, 1.0, size=unison).astype(np.float32) * (0.35 * spread)
    out = np.zeros(n, dtype=np.float32)

    for v in range(unison):
        cents = float(detunes[v])
        f = freq * (2.0 ** (cents / 1200.0))
        dt = f / float(sr)

        phase = (f * t).astype(np.float32)

        if sync > 0.0:
            base = phase % 1.0
            synced = (phase * sync_ratio) % 1.0
            phase01 = (1.0 - sync) * base + sync * synced
        else:
            phase01 = phase % 1.0

        phase01 = (phase01 + ph_off[v]) % 1.0

        if abs(pd_amount) > 1e-8:
            phase01 = _phase_distort(phase01, pd_amount)

        if fm_ratio > 0.0 and fm_index > 0.0:
            voice_fm_mod = np.sin(_TWOPI * (f * fm_ratio) * t).astype(np.float32)
            phase01 = (phase01 + ((fm_index ** 1.03) * 0.055) * voice_fm_mod) % 1.0

        if pm_amount > 0.0:
            voice_pm_mod = np.sin(_TWOPI * (f * 0.5) * t).astype(np.float32)
            phase01 = (phase01 + (((pm_amount ** 0.9) * 1.20) / _TWOPI) * voice_pm_mod) % 1.0

        if wave == "sine":
            y = np.sin(_TWOPI * phase01).astype(np.float32)
        elif wave == "saw":
            y = _saw_blep(phase01, dt)
        elif wave == "square":
            y = _square_blep(phase01, dt, pwm=pwm)
        elif wave == "triangle":
            sq = _square_blep(phase01, dt, pwm=0.5)
            y = _tri_from_square(sq)
        else:
            y = np.sin(_TWOPI * phase01).astype(np.float32)

        out += y

    out /= float(unison)
    out = _waveshape(out, drive=drive, fold=fold)
    out = _tilt_eq(out, tilt=tilt)

    return out.astype(np.float32, copy=False)


# ============================================================================
# Stereo / envelope helpers
# ============================================================================

def _pan_stereo(x: np.ndarray, pan: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, 0]

    pan = float(np.clip(pan, -1.0, 1.0))
    l = np.sqrt(0.5 * (1.0 - pan))
    r = np.sqrt(0.5 * (1.0 + pan))

    return np.stack([x * l, x * r], axis=1).astype(np.float32, copy=False)


def _microshift_stereo(
    x: np.ndarray,
    sr: int,
    amount_ms: float,
    mix: float,
    seed: int,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("microshift"):
            return NATIVE_DSP.microshift(
                x,
                sample_rate=int(sr),
                amount_ms=float(amount_ms),
                mix=float(mix),
                seed=int(seed),
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-6:
        return x

    sr = int(max(1, sr))
    n = x.shape[0]

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    amt = float(np.clip(amount_ms, 0.0, 25.0))
    base = int(round((amt / 1000.0) * sr))
    base = max(0, min(base, int(0.03 * sr)))

    if base <= 0:
        return x

    t = np.arange(n, dtype=np.float32) / float(sr)
    rate = float(rng.uniform(0.08, 0.22))
    depth = int(max(1, round(0.25 * base)))

    ph_l = float(rng.uniform(0.0, 2.0 * np.pi))
    ph_r = float(rng.uniform(0.0, 2.0 * np.pi))

    lfo_l = np.sin(_TWOPI * rate * t + ph_l).astype(np.float32)
    lfo_r = np.sin(_TWOPI * rate * t + ph_r).astype(np.float32)

    def _delay(inp: np.ndarray, lfo: np.ndarray) -> np.ndarray:
        out = np.empty_like(inp, dtype=np.float32)
        for i in range(n):
            d = base + int(round(depth * (0.5 * (1.0 + float(lfo[i])))))
            j = i - d
            out[i] = float(inp[j]) if j >= 0 else 0.0
        return out

    wet_l = _delay(x[:, 0], lfo_l)
    wet_r = _delay(x[:, 1], lfo_r)
    wet = np.stack([wet_l, wet_r], axis=1).astype(np.float32)
    wet = np.tanh(wet * 1.15).astype(np.float32)

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


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

    try:
        if _native_has("apply_adsr"):
            tmp = np.ones((n, 2), dtype=np.float32)
            tmp = NATIVE_DSP.apply_adsr(
                tmp,
                sample_rate=sr,
                attack=a,
                decay=d,
                sustain=s,
                release=r,
                curve=curve,
            )
            return tmp[:, 0].astype(np.float32, copy=False)
    except Exception:
        pass

    env = np.ones(n, dtype=np.float32)

    aN = int(round(a * sr))
    dN = int(round(d * sr))
    rN = int(round(r * sr))

    aN = max(0, min(aN, n))
    dN = max(0, min(dN, n - aN))
    rN = max(0, min(rN, n))

    if aN > 1:
        x = np.linspace(0.0, 1.0, aN, dtype=np.float32)
        env[:aN] = x ** (1.0 / curve)
    elif aN == 1:
        env[0] = 1.0

    idx = aN

    if dN > 1:
        x = np.linspace(0.0, 1.0, dN, dtype=np.float32)
        dec = (1.0 - s) * ((1.0 - x) ** curve) + s
        env[idx:idx + dN] = dec
    elif dN == 1 and idx < n:
        env[idx] = s

    idx += dN

    if idx < n:
        env[idx:] = s

    if rN > 1:
        x = np.linspace(0.0, 1.0, rN, dtype=np.float32)
        start_idx = max(0, n - rN - 1)
        release_start_val = float(env[start_idx])
        rel = release_start_val * (1.0 - x) ** (1.0 / curve)
        env[-rN:] = rel
    elif rN == 1:
        env[-1] = 0.0

    return np.clip(env, 0.0, 1.0).astype(np.float32, copy=False)


def _exp_decay(n: int, sr: int, t60: float) -> np.ndarray:
    n = int(max(1, n))
    sr = int(max(1, sr))
    t60 = float(max(1e-4, t60))

    t = np.arange(n, dtype=np.float32) / float(sr)
    return np.exp(np.log(0.001) * (t / t60)).astype(np.float32)


def _mono_vibrato_phase(
    freq: float,
    t: np.ndarray,
    sr: int,
    depth_cents: float,
    rate_hz: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    ph = float(rng.uniform(0.0, 2.0 * np.pi))

    depth_cents = float(np.clip(depth_cents, 0.0, 80.0))
    rate_hz = float(np.clip(rate_hz, 0.05, 12.0))

    vib = np.sin(_TWOPI * rate_hz * t + ph).astype(np.float32)
    inst_freq = float(freq) * (2.0 ** ((depth_cents * vib) / 1200.0))

    phase = np.cumsum(inst_freq.astype(np.float32)) * (_TWOPI / float(sr))
    return phase.astype(np.float32, copy=False)


def _filtered_noise_mono(
    n: int,
    sr: int,
    seed: int,
    lowpass_hz: float,
    highpass_hz: float = 20.0,
) -> np.ndarray:
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    nz = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    st = _pan_stereo(nz, 0.0)

    if highpass_hz > 20.0:
        b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, float(highpass_hz), 0.707)
        st = _biquad_process_stereo(st, b0, b1, b2, a1, a2)

    st = _svf_lowpass_stereo(st, sr, cutoff_hz=float(lowpass_hz), res=0.0)
    return st[:, 0].astype(np.float32, copy=False)


# ============================================================================
# Filters / body / reverb / chorus
# ============================================================================

def _svf_lowpass_stereo(
    x: np.ndarray,
    sr: int,
    cutoff_hz: float,
    res: float,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("svf_lowpass"):
            return NATIVE_DSP.svf_lowpass(
                x,
                sample_rate=int(sr),
                cutoff_hz=float(cutoff_hz),
                resonance=float(res),
                wet=1.0,
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    sr = int(max(1, sr))
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, 0.49 * sr))
    res = float(np.clip(res, 0.0, 1.0))

    g = np.tan(np.pi * cutoff_hz / float(sr))
    g = float(np.clip(g, 0.0, 1.5))

    r = 1.15 - 1.02 * res
    r = float(np.clip(r, 0.08, 1.5))

    den = 1.0 + g * (g + r)

    y = np.empty_like(x, dtype=np.float32)

    ic1_l = 0.0
    ic2_l = 0.0
    ic1_r = 0.0
    ic2_r = 0.0

    for i in range(x.shape[0]):
        in_l = float(x[i, 0])
        in_r = float(x[i, 1])

        v0 = in_l - r * ic2_l
        v1 = (g * v0 + ic1_l) / den
        v2 = ic2_l + g * v1
        ic1_l = 2.0 * v1 - ic1_l
        ic2_l = 2.0 * v2 - ic2_l
        y[i, 0] = v2

        v0 = in_r - r * ic2_r
        v1 = (g * v0 + ic1_r) / den
        v2 = ic2_r + g * v1
        ic1_r = 2.0 * v1 - ic1_r
        ic2_r = 2.0 * v2 - ic2_r
        y[i, 1] = v2

    return y.astype(np.float32, copy=False)


def _keytracked_cutoff(base_cutoff: float, freq: float, keytrack: float) -> float:
    base_cutoff = float(base_cutoff)
    keytrack = float(np.clip(keytrack, 0.0, 1.0))
    return base_cutoff * (float(freq) / 440.0) ** (0.9 * keytrack)


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

    try:
        if _native_has("chorus"):
            return NATIVE_DSP.chorus(
                x,
                sample_rate=int(sr),
                rate_hz=float(rate_hz),
                depth_ms=float(depth_ms),
                mix=float(mix),
                seed=int(seed),
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-6:
        return x

    sr = int(max(1, sr))
    rate_hz = float(np.clip(rate_hz, 0.02, 5.0))
    depth_ms = float(np.clip(depth_ms, 0.2, 30.0))

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    n = x.shape[0]
    t = np.arange(n, dtype=np.float32) / float(sr)

    base_s = int(round(0.009 * sr))
    depth_s = max(1, int(round((depth_ms / 1000.0) * sr)))

    rate_l = rate_hz * (1.0 + rng.uniform(-0.05, 0.05))
    rate_r = rate_hz * (1.0 + rng.uniform(-0.05, 0.05))

    ph_l = float(rng.uniform(0.0, 2.0 * np.pi))
    ph_r = float(rng.uniform(0.0, 2.0 * np.pi)) + np.pi / 2.0

    lfo_l = np.sin(_TWOPI * rate_l * t + ph_l).astype(np.float32)
    lfo_r = np.sin(_TWOPI * rate_r * t + ph_r).astype(np.float32)

    def _delay_chan(inp: np.ndarray, lfo: np.ndarray) -> np.ndarray:
        out = np.empty_like(inp, dtype=np.float32)
        max_delay_samples = base_s + depth_s + 2
        delay_line = np.zeros(max_delay_samples, dtype=np.float32)
        write_idx = 0

        for i in range(n):
            current_delay = base_s + (0.5 * (1.0 + float(lfo[i]))) * depth_s
            di = int(current_delay)
            frac = current_delay - di

            read_idx_0 = (write_idx - di + max_delay_samples) % max_delay_samples
            read_idx_1 = (write_idx - di - 1 + max_delay_samples) % max_delay_samples

            s0 = delay_line[read_idx_0]
            s1 = delay_line[read_idx_1]

            out[i] = (1.0 - frac) * s0 + frac * s1

            delay_line[write_idx] = float(inp[i])
            write_idx = (write_idx + 1) % max_delay_samples

        return out

    wet_l = _delay_chan(x[:, 0], lfo_l)
    wet_r = _delay_chan(x[:, 1], lfo_r)

    wet = np.stack([wet_l, wet_r], axis=1).astype(np.float32)
    wet = np.tanh(wet * 1.10).astype(np.float32)

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


def _schroeder_reverb_stereo(
    x: np.ndarray,
    sr: int,
    *,
    mix: float,
    room: float,
    predelay_ms: float = 12.0,
    damp_hz: float = 7000.0,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)

    try:
        if _native_has("reverb"):
            return NATIVE_DSP.reverb(
                x,
                sample_rate=int(sr),
                mix=float(mix),
                room=float(room),
                predelay_ms=float(predelay_ms),
                damp_hz=float(damp_hz),
            ).astype(np.float32, copy=False)
    except Exception:
        pass

    sr = int(max(1, sr))
    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-6:
        return x

    room = float(np.clip(room, 0.0, 1.0))
    predelay_ms = float(np.clip(predelay_ms, 0.0, 80.0))
    damp_hz = float(np.clip(damp_hz, 1200.0, 14000.0))

    n = x.shape[0]
    pd = int(round((predelay_ms / 1000.0) * sr))

    if pd > 0:
        wet_in = np.zeros_like(x, dtype=np.float32)
        if pd < n:
            wet_in[pd:] = x[:-pd]
    else:
        wet_in = x

    base = np.asarray([0.0297, 0.0371, 0.0411, 0.0437, 0.0461, 0.0503], dtype=np.float32)
    comb_delays = (base * (0.62 + 0.95 * room) * sr).astype(int)
    comb_delays = np.clip(comb_delays, 64, int(0.12 * sr))

    fb = 0.74 + 0.18 * room

    def _comb(inp: np.ndarray, d: int) -> np.ndarray:
        y = np.zeros_like(inp, dtype=np.float32)
        buf = np.zeros(d, dtype=np.float32)
        idx = 0

        for i in range(inp.shape[0]):
            v = float(inp[i]) + fb * float(buf[idx])
            y[i] = float(buf[idx])
            buf[idx] = v
            idx = (idx + 1) % d

        return y

    wet = np.zeros_like(wet_in, dtype=np.float32)

    for ch in (0, 1):
        inp = wet_in[:, ch]
        comb_sum = np.zeros_like(inp, dtype=np.float32)

        for d in comb_delays:
            comb_sum += _comb(inp, int(d))

        wet[:, ch] = comb_sum * (1.0 / float(len(comb_delays)))

    wet = _svf_lowpass_stereo(wet, sr, cutoff_hz=damp_hz, res=0.0)
    wet = np.tanh(wet * 1.05).astype(np.float32)

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


def _body_resonance_stereo(
    x: np.ndarray,
    sr: int,
    amount: float,
    modes: List[Tuple[float, float]],
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    amount = float(np.clip(amount, 0.0, 1.0))

    if amount <= 1e-6:
        return x

    acc = np.zeros_like(x, dtype=np.float32)

    for f, g in modes:
        y = _svf_lowpass_stereo(x, sr, cutoff_hz=float(f), res=0.9)
        acc += float(g) * y

    out = x + amount * acc
    return out.astype(np.float32, copy=False)


# ============================================================================
# Biquad helpers
# ============================================================================

def _biquad_highpass_coeff(
    sr: int,
    fc: float,
    q: float,
) -> Tuple[float, float, float, float, float]:
    sr = int(max(1, sr))
    fc = float(np.clip(fc, 20.0, sr * 0.45))
    q = float(np.clip(q, 0.1, 24.0))

    w0 = 2.0 * np.pi * fc / float(sr)
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / (2.0 * q)

    b0 = (1.0 + cosw) / 2.0
    b1 = -(1.0 + cosw)
    b2 = (1.0 + cosw) / 2.0

    a0 = 1.0 + alpha
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha

    return (
        float(b0 / a0),
        float(b1 / a0),
        float(b2 / a0),
        float(a1 / a0),
        float(a2 / a0),
    )


def _biquad_bandpass_coeff(
    sr: int,
    f0: float,
    q: float,
) -> Tuple[float, float, float, float, float]:
    sr = int(max(1, sr))
    f0 = float(np.clip(f0, 20.0, sr * 0.45))
    q = float(np.clip(q, 0.1, 24.0))

    w0 = 2.0 * np.pi * f0 / float(sr)
    cosw = np.cos(w0)
    sinw = np.sin(w0)
    alpha = sinw / (2.0 * q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha

    a0 = 1.0 + alpha
    a1 = -2.0 * cosw
    a2 = 1.0 - alpha

    return (
        float(b0 / a0),
        float(b1 / a0),
        float(b2 / a0),
        float(a1 / a0),
        float(a2 / a0),
    )


def _biquad_process_stereo(
    x: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    y = np.zeros_like(x, dtype=np.float32)

    for ch in (0, 1):
        x1 = 0.0
        x2 = 0.0
        y1 = 0.0
        y2 = 0.0

        for i in range(x.shape[0]):
            x0 = float(x[i, ch])
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

            y[i, ch] = y0

            x2 = x1
            x1 = x0
            y2 = y1
            y1 = y0

    return y.astype(np.float32, copy=False)


# ============================================================================
# Instruments
# ============================================================================

class SynthKeys(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "wave": {
            "type": "choice",
            "default": "saw",
            "choices": ["sine", "triangle", "square", "pulse", "saw", "sawtooth"],
        },

        "amp": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},

        "attack": {"type": "float", "default": 0.012, "min": 0.0, "max": 3.0, "step": 0.005},
        "decay": {"type": "float", "default": 0.150, "min": 0.0, "max": 6.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.78, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 0.45, "min": 0.0, "max": 8.0, "step": 0.01},
        "env_curve": {"type": "float", "default": 0.65, "min": 0.15, "max": 2.5, "step": 0.05},

        "unison": {"type": "int", "default": 9, "min": 1, "max": 16, "step": 1},
        "detune_cents": {"type": "float", "default": 18.0, "min": 0.0, "max": 80.0, "step": 0.5},
        "spread": {"type": "float", "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01},

        "pwm": {"type": "float", "default": 0.48, "min": 0.01, "max": 0.99, "step": 0.01},

        "sync": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},
        "sync_ratio": {"type": "float", "default": 3.2, "min": 1.0, "max": 12.0, "step": 0.25},
        "fm_ratio": {"type": "float", "default": 1.6, "min": 0.0, "max": 12.0, "step": 0.25},
        "fm_index": {"type": "float", "default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1},
        "pm_amount": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.35, "step": 0.005},
        "pd_amount": {"type": "float", "default": 0.15, "min": -0.95, "max": 0.95, "step": 0.01},

        "drive": {"type": "float", "default": 1.80, "min": 0.25, "max": 10.0, "step": 0.05},
        "fold": {"type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
        "tilt": {"type": "float", "default": 0.15, "min": -1.0, "max": 1.0, "step": 0.01},

        "sub": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        "noise": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        "sparkle": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},

        "cutoff_hz": {"type": "float", "default": 6500.0, "min": 80.0, "max": 20000.0, "step": 50.0},
        "res": {"type": "float", "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01},
        "keytrack": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},

        "width_mix": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
        "width_ms": {"type": "float", "default": 8.0, "min": 0.0, "max": 25.0, "step": 0.5},

        "chorus_mix": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
        "chorus_rate": {"type": "float", "default": 0.20, "min": 0.02, "max": 4.0, "step": 0.02},
        "chorus_depth_ms": {"type": "float", "default": 10.0, "min": 0.5, "max": 25.0, "step": 0.5},

        "reverb_mix": {"type": "float", "default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01},
        "reverb_predelay_ms": {"type": "float", "default": 18.0, "min": 0.0, "max": 80.0, "step": 1.0},
        "reverb_damp_hz": {"type": "float", "default": 6500.0, "min": 1500.0, "max": 14000.0, "step": 100.0},

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)

        try:
            if _native_has("render_sound_synth_keys"):
                y = NATIVE_DSP.render_sound_synth_keys(
                    freq=freq,
                    frames=n,
                    sample_rate=sr,
                    velocity=vel,
                    **params,
                )
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_synth_keys"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        t = np.arange(n, dtype=np.float32) / float(sr)
        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        wave_ui = params.get("wave", "saw")
        amp = float(params.get("amp", 0.35))
        pan = float(params.get("pan", 0.0))

        env = _adsr_env(
            n,
            sr,
            a=float(params.get("attack", 0.012)),
            d=float(params.get("decay", 0.150)),
            s=float(params.get("sustain", 0.78)),
            r=float(params.get("release", 0.45)),
            curve=float(params.get("env_curve", 0.65)),
        )

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.45 + 0.85 * (vel ** 0.9)
        vel_bright = 0.65 + 0.75 * (vel ** 0.9)

        f0 = freq * float(1.0 + 0.0025 * rng.uniform(-1.0, 1.0))

        x_main = osc_advanced(
            wave=wave_ui,
            t=t,
            freq=f0,
            sr=sr,
            unison=int(params.get("unison", 9)),
            detune_cents=float(params.get("detune_cents", 18.0)),
            spread=float(params.get("spread", 0.90)),
            pwm=float(params.get("pwm", 0.48)),
            sync=float(params.get("sync", 0.15)),
            sync_ratio=float(params.get("sync_ratio", 3.2)),
            fm_ratio=float(params.get("fm_ratio", 1.6)),
            fm_index=float(params.get("fm_index", 5.5)) * vel_bright,
            pm_amount=float(params.get("pm_amount", 0.08)) * vel_bright,
            pd_amount=float(params.get("pd_amount", 0.15)),
            drive=float(params.get("drive", 1.80)),
            fold=float(params.get("fold", 0.12)),
            tilt=float(params.get("tilt", 0.15)),
            seed=seed,
        )

        sub_amt = float(params.get("sub", 0.28))
        if sub_amt > 1e-6:
            sub = np.sin(_TWOPI * (0.5 * freq) * t).astype(np.float32)
            sub = np.tanh(sub * 1.35).astype(np.float32)
        else:
            sub = 0.0

        noise_amt = float(params.get("noise", 0.10))
        if noise_amt > 1e-6:
            nz = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            nz = np.tanh(nz * 0.45).astype(np.float32)
            nz_st = _pan_stereo(nz, 0.0)
            nz_st = _svf_lowpass_stereo(nz_st, sr, cutoff_hz=8000.0 * vel_bright, res=0.0)
            nz = nz_st[:, 0]
        else:
            nz = 0.0

        sparkle_amt = float(params.get("sparkle", 0.15))
        if sparkle_amt > 1e-6:
            sparkle = (
                0.16 * np.sin(_TWOPI * freq * 2.0 * t)
                + 0.08 * np.sin(_TWOPI * freq * 3.01 * t)
            ).astype(np.float32)
        else:
            sparkle = 0.0

        x = (
            x_main
            + sub_amt * 0.55 * sub
            + noise_amt * 0.06 * nz
            + sparkle_amt * sparkle
        ).astype(np.float32)

        x *= env * amp * vel_gain

        y = _pan_stereo(x, pan)

        cutoff = _keytracked_cutoff(
            float(params.get("cutoff_hz", 6500.0)),
            freq,
            float(params.get("keytrack", 0.45)),
        )
        cutoff = float(np.clip(cutoff, 40.0, sr * 0.45))

        y = _svf_lowpass_stereo(y, sr, cutoff_hz=cutoff, res=float(params.get("res", 0.40)))

        if float(params.get("width_mix", 0.35)) > 1e-6:
            y = _microshift_stereo(
                y,
                sr,
                amount_ms=float(params.get("width_ms", 8.0)),
                mix=float(params.get("width_mix", 0.35)),
                seed=seed + 17,
            )

        if float(params.get("chorus_mix", 0.35)) > 1e-6:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=float(params.get("chorus_rate", 0.20)),
                depth_ms=float(params.get("chorus_depth_ms", 10.0)),
                mix=float(params.get("chorus_mix", 0.35)),
                seed=seed + 31,
            )

        if float(params.get("reverb_mix", 0.25)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.25)),
                room=float(params.get("reverb_room", 0.60)),
                predelay_ms=float(params.get("reverb_predelay_ms", 18.0)),
                damp_hz=float(params.get("reverb_damp_hz", 6500.0)),
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)

        extra = {"engine": "numpy_synth_keys"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class LeadSynth(SynthKeys):
    KIND = "instrument"
    PARAMS = dict(SynthKeys.PARAMS)
    PARAMS.update({
        "wave": {
            "type": "choice",
            "default": "square",
            "choices": ["sine", "triangle", "square", "pulse", "saw", "sawtooth"],
        },
        "amp": {"type": "float", "default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01},
        "unison": {"type": "int", "default": 3, "min": 1, "max": 16, "step": 1},
        "detune_cents": {"type": "float", "default": 7.0, "min": 0.0, "max": 80.0, "step": 0.5},
        "spread": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
        "attack": {"type": "float", "default": 0.003, "min": 0.0, "max": 3.0, "step": 0.005},
        "decay": {"type": "float", "default": 0.08, "min": 0.0, "max": 6.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 0.16, "min": 0.0, "max": 8.0, "step": 0.01},
        "cutoff_hz": {"type": "float", "default": 9000.0, "min": 80.0, "max": 20000.0, "step": 50.0},
        "res": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        "chorus_mix": {"type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.12, "min": 0.0, "max": 0.8, "step": 0.01},
    })


class GuitarPluck(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},
        "decay": {"type": "float", "default": 2.2, "min": 0.05, "max": 12.0, "step": 0.05},
        "brightness": {"type": "float", "default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01},
        "pick": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01},
        "noise": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},
        "chorus_mix": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.6, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.18, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},
        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)

        try:
            if _native_has("render_sound_guitar_pluck"):
                y = NATIVE_DSP.render_sound_guitar_pluck(
                    freq=freq,
                    frames=n,
                    sample_rate=sr,
                    velocity=vel,
                    **params,
                )
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_guitar_pluck"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        amp = float(params.get("amp", 0.42))
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

        if noise > 1e-6:
            hit_n = min(n, max(16, int(round(0.02 * sr))))
            hit = rng.normal(0.0, 1.0, size=hit_n).astype(np.float32)
            x[:hit_n] += noise * 0.12 * hit * _exp_decay(hit_n, sr, 0.025)

        x = np.tanh(x * (1.0 + pick * 1.5)).astype(np.float32)
        x *= amp * (0.35 + 0.85 * float(np.clip(vel, 0.0, 1.0)))

        y = _pan_stereo(x, pan)

        y = _body_resonance_stereo(
            y,
            sr,
            amount=body,
            modes=[
                (180.0, 0.45),
                (360.0, 0.32),
                (720.0, 0.20),
                (1450.0, 0.12),
            ],
        )

        if float(params.get("chorus_mix", 0.08)) > 1e-6:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.25,
                depth_ms=8.0,
                mix=float(params.get("chorus_mix", 0.08)),
                seed=seed + 73,
            )

        if float(params.get("reverb_mix", 0.18)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.18)),
                room=float(params.get("reverb_room", 0.55)),
                predelay_ms=18.0,
                damp_hz=6500.0,
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)

        extra = {"engine": "numpy_guitar_pluck"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class BellFM(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
        "brightness": {"type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01},
        "inharm": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
        "decay": {"type": "float", "default": 3.5, "min": 0.05, "max": 12.0, "step": 0.05},
        "fm_ratio": {"type": "float", "default": 3.0, "min": 0.0, "max": 12.0, "step": 0.05},
        "fm_index": {"type": "float", "default": 8.0, "min": 0.0, "max": 24.0, "step": 0.1},
        "strike": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        "shimmer": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
        "chorus_mix": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.30, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},
        "reverb_predelay_ms": {"type": "float", "default": 22.0, "min": 0.0, "max": 80.0, "step": 1.0},
        "reverb_damp_hz": {"type": "float", "default": 6000.0, "min": 1500.0, "max": 14000.0, "step": 100.0},
        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)

        try:
            if _native_has("render_sound_bell_fm"):
                y = NATIVE_DSP.render_sound_bell_fm(
                    freq=freq,
                    frames=n,
                    sample_rate=sr,
                    velocity=vel,
                    **params,
                )
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_bell_fm"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        t = np.arange(n, dtype=np.float32) / float(sr)

        amp = float(params.get("amp", 0.35))
        brightness = float(np.clip(params.get("brightness", 0.65), 0.0, 1.0))
        inharm = float(np.clip(params.get("inharm", 0.75), 0.0, 1.0))
        decay = float(params.get("decay", 3.5))
        fm_ratio = float(params.get("fm_ratio", 3.0))
        fm_index = float(params.get("fm_index", 8.0))
        strike = float(np.clip(params.get("strike", 0.28), 0.0, 1.0))
        shimmer = float(np.clip(params.get("shimmer", 0.25), 0.0, 1.0))
        body = float(np.clip(params.get("body", 0.35), 0.0, 1.0))
        pan = float(params.get("pan", 0.0))

        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.50 + 0.80 * (vel ** 0.9)
        vel_bright = 0.60 + 0.80 * (vel ** 0.9)

        base_ratios = np.asarray([1.0, 2.02, 2.78, 3.95, 5.48, 6.90, 8.10], dtype=np.float32)
        harm_ratios = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        ratios = (1.0 - inharm) * harm_ratios + inharm * base_ratios

        gains = np.asarray([1.00, 0.60, 0.40, 0.25, 0.16, 0.10, 0.06], dtype=np.float32)
        gains = gains * (0.80 + 0.80 * brightness)
        gains[0] *= 1.0 - 0.20 * brightness

        base_t60 = max(0.05, decay)
        t60s = base_t60 * (0.90 - 0.15 * brightness) / (
            0.80 + 0.30 * np.arange(len(ratios), dtype=np.float32)
        )
        t60s = np.clip(t60s, 0.10, 20.0)

        x = np.zeros(n, dtype=np.float32)

        fm_on = fm_ratio > 1e-6 and fm_index > 1e-6
        fm_index_eff = fm_index * (0.45 + 0.80 * brightness) * vel_bright

        for i, (r, g) in enumerate(zip(ratios, gains)):
            cents = float(rng.uniform(-5.0, 5.0))
            f = (freq * float(r)) * (2.0 ** (cents / 1200.0))
            ph = float(rng.uniform(0.0, 2.0 * np.pi))

            if fm_on:
                mod = np.sin(_TWOPI * (f * fm_ratio) * t + ph * 0.45).astype(np.float32)
                part = np.sin(_TWOPI * f * t + ph + (0.10 * fm_index_eff * float(g)) * mod).astype(np.float32)
            else:
                part = np.sin(_TWOPI * f * t + ph).astype(np.float32)

            env = _exp_decay(n, sr, t60=float(t60s[i]))
            x += (float(g) * part * env).astype(np.float32)

        x *= amp * vel_gain

        if strike > 1e-6:
            hit_n = int(max(16, round(0.015 * sr)))
            hit_n = min(hit_n, n)

            hit = rng.normal(0.0, 1.0, size=hit_n).astype(np.float32)
            hit *= _exp_decay(hit_n, sr, t60=0.018)

            x[:hit_n] += strike * 0.35 * hit

        if shimmer > 1e-6:
            env = _exp_decay(n, sr, t60=decay * 0.65)
            x += shimmer * 0.12 * env * np.sin(_TWOPI * freq * 4.0 * t).astype(np.float32)
            x += shimmer * 0.07 * env * np.sin(_TWOPI * freq * 6.7 * t).astype(np.float32)

        y = _pan_stereo(x, pan)

        y = _body_resonance_stereo(
            y,
            sr,
            amount=body,
            modes=[
                (180.0, 0.45),
                (360.0, 0.32),
                (720.0, 0.20),
                (1450.0, 0.12),
            ],
        )

        if float(params.get("chorus_mix", 0.15)) > 1e-6:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.18,
                depth_ms=7.0,
                mix=float(params.get("chorus_mix", 0.15)),
                seed=seed + 191,
            )

        if float(params.get("reverb_mix", 0.30)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.30)),
                room=float(params.get("reverb_room", 0.70)),
                predelay_ms=float(params.get("reverb_predelay_ms", 22.0)),
                damp_hz=float(params.get("reverb_damp_hz", 6000.0)),
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)

        extra = {"engine": "numpy_bell_fm"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class BrassSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01},

        "attack": {"type": "float", "default": 0.035, "min": 0.0, "max": 2.0, "step": 0.005},
        "decay": {"type": "float", "default": 0.18, "min": 0.0, "max": 4.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.82, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 0.24, "min": 0.0, "max": 5.0, "step": 0.01},

        "brightness": {"type": "float", "default": 0.62, "min": 0.0, "max": 1.0, "step": 0.01},
        "buzz": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        "growl": {"type": "float", "default": 0.16, "min": 0.0, "max": 1.0, "step": 0.01},
        "air": {"type": "float", "default": 0.035, "min": 0.0, "max": 0.5, "step": 0.005},
        "body": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},

        "vibrato_rate": {"type": "float", "default": 5.2, "min": 0.1, "max": 10.0, "step": 0.1},
        "vibrato_depth_cents": {"type": "float", "default": 8.0, "min": 0.0, "max": 60.0, "step": 0.5},

        "cutoff_hz": {"type": "float", "default": 6800.0, "min": 400.0, "max": 18000.0, "step": 50.0},
        "res": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},

        "chorus_mix": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.7, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.18, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)
        t = np.arange(n, dtype=np.float32) / float(sr)

        seed = int(params.get("seed", 0))
        amp = float(params.get("amp", 0.38))
        brightness = float(np.clip(params.get("brightness", 0.62), 0.0, 1.0))
        buzz = float(np.clip(params.get("buzz", 0.45), 0.0, 1.0))
        growl = float(np.clip(params.get("growl", 0.16), 0.0, 1.0))
        air = float(np.clip(params.get("air", 0.035), 0.0, 0.5))
        pan = float(params.get("pan", 0.0))

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.45 + 0.90 * (vel ** 0.85)
        vel_bright = 0.70 + 0.75 * (vel ** 0.85)

        env = _adsr_env(
            n,
            sr,
            a=float(params.get("attack", 0.035)),
            d=float(params.get("decay", 0.18)),
            s=float(params.get("sustain", 0.82)),
            r=float(params.get("release", 0.24)),
            curve=0.75,
        )

        phase = _mono_vibrato_phase(
            freq,
            t,
            sr,
            depth_cents=float(params.get("vibrato_depth_cents", 8.0)),
            rate_hz=float(params.get("vibrato_rate", 5.2)),
            seed=seed + 11,
        )

        growl_mod = np.sin(_TWOPI * 38.0 * t + 0.7).astype(np.float32)
        phase_g = phase + growl * 0.10 * growl_mod

        x = (
            1.00 * np.sin(phase_g)
            + (0.62 + 0.40 * brightness) * np.sin(2.0 * phase_g + 0.02)
            + (0.34 + 0.38 * brightness) * np.sin(3.0 * phase_g + 0.04)
            + (0.18 + 0.32 * brightness) * np.sin(4.0 * phase_g + 0.07)
            + (0.08 + 0.20 * brightness) * np.sin(5.0 * phase_g + 0.11)
        ).astype(np.float32)

        x /= 2.20
        x = _waveshape(x, drive=1.0 + 2.5 * buzz * vel_bright, fold=0.02 + 0.08 * buzz)

        if air > 1e-6:
            breath = _filtered_noise_mono(n, sr, seed + 101, lowpass_hz=9000.0, highpass_hz=900.0)
            x += air * breath * env

        x *= env * amp * vel_gain

        y = _pan_stereo(x, pan)

        cutoff = float(params.get("cutoff_hz", 6800.0)) * (0.65 + 0.80 * vel_bright)
        y = _svf_lowpass_stereo(
            y,
            sr,
            cutoff_hz=float(np.clip(cutoff, 400.0, sr * 0.45)),
            res=float(params.get("res", 0.35)),
        )

        y = _body_resonance_stereo(
            y,
            sr,
            amount=float(params.get("body", 0.42)),
            modes=[
                (220.0, 0.35),
                (480.0, 0.24),
                (980.0, 0.18),
                (2200.0, 0.12),
            ],
        )

        if float(params.get("chorus_mix", 0.08)) > 1e-6:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.18,
                depth_ms=4.5,
                mix=float(params.get("chorus_mix", 0.08)),
                seed=seed + 31,
            )

        if float(params.get("reverb_mix", 0.18)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.18)),
                room=float(params.get("reverb_room", 0.55)),
                predelay_ms=18.0,
                damp_hz=6200.0,
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_brass_synth"})


class FluteSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.32, "min": 0.0, "max": 1.0, "step": 0.01},

        "attack": {"type": "float", "default": 0.055, "min": 0.0, "max": 2.0, "step": 0.005},
        "decay": {"type": "float", "default": 0.12, "min": 0.0, "max": 3.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.88, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 0.30, "min": 0.0, "max": 5.0, "step": 0.01},

        "breath": {"type": "float", "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
        "air": {"type": "float", "default": 0.24, "min": 0.0, "max": 1.0, "step": 0.01},
        "edge": {"type": "float", "default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},

        "vibrato_rate": {"type": "float", "default": 5.6, "min": 0.1, "max": 10.0, "step": 0.1},
        "vibrato_depth_cents": {"type": "float", "default": 10.0, "min": 0.0, "max": 80.0, "step": 0.5},

        "cutoff_hz": {"type": "float", "default": 9200.0, "min": 1000.0, "max": 18000.0, "step": 50.0},
        "reverb_mix": {"type": "float", "default": 0.24, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.62, "min": 0.0, "max": 1.0, "step": 0.01},

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)
        t = np.arange(n, dtype=np.float32) / float(sr)

        seed = int(params.get("seed", 0))
        amp = float(params.get("amp", 0.32))
        breath = float(np.clip(params.get("breath", 0.18), 0.0, 1.0))
        air = float(np.clip(params.get("air", 0.24), 0.0, 1.0))
        edge = float(np.clip(params.get("edge", 0.18), 0.0, 1.0))
        pan = float(params.get("pan", 0.0))

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.45 + 0.85 * vel
        vel_air = 0.50 + 0.85 * vel

        env = _adsr_env(
            n,
            sr,
            a=float(params.get("attack", 0.055)),
            d=float(params.get("decay", 0.12)),
            s=float(params.get("sustain", 0.88)),
            r=float(params.get("release", 0.30)),
            curve=0.85,
        )

        phase = _mono_vibrato_phase(
            freq,
            t,
            sr,
            depth_cents=float(params.get("vibrato_depth_cents", 10.0)),
            rate_hz=float(params.get("vibrato_rate", 5.6)),
            seed=seed + 17,
        )

        tone = (
            1.00 * np.sin(phase)
            + 0.16 * edge * np.sin(2.0 * phase + 0.4)
            + 0.07 * edge * np.sin(3.0 * phase + 0.9)
        ).astype(np.float32)

        breath_noise = _filtered_noise_mono(n, sr, seed + 111, lowpass_hz=10500.0, highpass_hz=1700.0)
        slow_breath = 0.65 + 0.35 * np.sin(_TWOPI * 1.6 * t + 0.25).astype(np.float32)

        x = tone + (0.10 * breath + 0.12 * air * vel_air) * breath_noise * slow_breath
        x = _waveshape(x, drive=0.85 + 0.65 * edge, fold=0.0)
        x *= env * amp * vel_gain

        y = _pan_stereo(x, pan)

        b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, 180.0, 0.707)
        y = _biquad_process_stereo(y, b0, b1, b2, a1, a2)

        y = _svf_lowpass_stereo(
            y,
            sr,
            cutoff_hz=float(params.get("cutoff_hz", 9200.0)),
            res=0.08,
        )

        y = _body_resonance_stereo(
            y,
            sr,
            amount=float(params.get("body", 0.28)),
            modes=[
                (520.0, 0.20),
                (980.0, 0.22),
                (2100.0, 0.15),
                (4200.0, 0.08),
            ],
        )

        if float(params.get("reverb_mix", 0.24)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.24)),
                room=float(params.get("reverb_room", 0.62)),
                predelay_ms=24.0,
                damp_hz=7600.0,
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_flute_synth"})


class ClarinetSynth(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.34, "min": 0.0, "max": 1.0, "step": 0.01},

        "attack": {"type": "float", "default": 0.025, "min": 0.0, "max": 2.0, "step": 0.005},
        "decay": {"type": "float", "default": 0.14, "min": 0.0, "max": 3.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.84, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 0.22, "min": 0.0, "max": 5.0, "step": 0.01},

        "reed": {"type": "float", "default": 0.38, "min": 0.0, "max": 1.0, "step": 0.01},
        "breath": {"type": "float", "default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01},
        "brightness": {"type": "float", "default": 0.44, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.46, "min": 0.0, "max": 1.0, "step": 0.01},

        "vibrato_rate": {"type": "float", "default": 4.7, "min": 0.1, "max": 10.0, "step": 0.1},
        "vibrato_depth_cents": {"type": "float", "default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5},

        "cutoff_hz": {"type": "float", "default": 5200.0, "min": 400.0, "max": 14000.0, "step": 50.0},
        "reverb_mix": {"type": "float", "default": 0.18, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01},

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)
        t = np.arange(n, dtype=np.float32) / float(sr)

        seed = int(params.get("seed", 0))
        amp = float(params.get("amp", 0.34))
        reed = float(np.clip(params.get("reed", 0.38), 0.0, 1.0))
        breath = float(np.clip(params.get("breath", 0.08), 0.0, 1.0))
        brightness = float(np.clip(params.get("brightness", 0.44), 0.0, 1.0))
        pan = float(params.get("pan", 0.0))

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.45 + 0.85 * vel

        env = _adsr_env(
            n,
            sr,
            a=float(params.get("attack", 0.025)),
            d=float(params.get("decay", 0.14)),
            s=float(params.get("sustain", 0.84)),
            r=float(params.get("release", 0.22)),
            curve=0.70,
        )

        phase = _mono_vibrato_phase(
            freq,
            t,
            sr,
            depth_cents=float(params.get("vibrato_depth_cents", 4.0)),
            rate_hz=float(params.get("vibrato_rate", 4.7)),
            seed=seed + 19,
        )

        x = (
            1.00 * np.sin(phase)
            + (0.58 + 0.35 * brightness) * np.sin(3.0 * phase + 0.05)
            + (0.28 + 0.30 * brightness) * np.sin(5.0 * phase + 0.12)
            + (0.11 + 0.20 * brightness) * np.sin(7.0 * phase + 0.19)
        ).astype(np.float32)

        x /= 1.95
        x = _waveshape(x, drive=0.9 + 1.65 * reed, fold=0.01 + 0.05 * reed)

        if breath > 1e-6:
            nz = _filtered_noise_mono(n, sr, seed + 141, lowpass_hz=6200.0, highpass_hz=700.0)
            x += breath * 0.16 * nz * env

        x *= env * amp * vel_gain

        y = _pan_stereo(x, pan)

        y = _svf_lowpass_stereo(
            y,
            sr,
            cutoff_hz=float(params.get("cutoff_hz", 5200.0)) * (0.75 + 0.65 * vel),
            res=0.22,
        )

        y = _body_resonance_stereo(
            y,
            sr,
            amount=float(params.get("body", 0.46)),
            modes=[
                (180.0, 0.20),
                (620.0, 0.32),
                (1180.0, 0.25),
                (2350.0, 0.13),
            ],
        )

        if float(params.get("reverb_mix", 0.18)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.18)),
                room=float(params.get("reverb_room", 0.52)),
                predelay_ms=18.0,
                damp_hz=6000.0,
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_clarinet_synth"})


class StringPad(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01},

        "attack": {"type": "float", "default": 0.55, "min": 0.0, "max": 5.0, "step": 0.01},
        "decay": {"type": "float", "default": 0.35, "min": 0.0, "max": 5.0, "step": 0.01},
        "sustain": {"type": "float", "default": 0.86, "min": 0.0, "max": 1.0, "step": 0.01},
        "release": {"type": "float", "default": 1.20, "min": 0.0, "max": 8.0, "step": 0.01},

        "brightness": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},
        "motion": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},
        "body": {"type": "float", "default": 0.34, "min": 0.0, "max": 1.0, "step": 0.01},

        "chorus_mix": {"type": "float", "default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01},
        "reverb_mix": {"type": "float", "default": 0.30, "min": 0.0, "max": 0.8, "step": 0.01},
        "reverb_room": {"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq, dur, sr, vel = _voice_payload(payload)
        n = _frames(dur, sr)
        t = np.arange(n, dtype=np.float32) / float(sr)

        seed = int(params.get("seed", 0))
        amp = float(params.get("amp", 0.30))
        brightness = float(np.clip(params.get("brightness", 0.42), 0.0, 1.0))
        motion = float(np.clip(params.get("motion", 0.42), 0.0, 1.0))
        pan = float(params.get("pan", 0.0))

        env = _adsr_env(
            n,
            sr,
            a=float(params.get("attack", 0.55)),
            d=float(params.get("decay", 0.35)),
            s=float(params.get("sustain", 0.86)),
            r=float(params.get("release", 1.20)),
            curve=1.05,
        )

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.45 + 0.75 * vel

        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        x = np.zeros(n, dtype=np.float32)
        detunes = [-9.0, -4.0, 0.0, 4.0, 9.0]

        for i, cents in enumerate(detunes):
            f = float(freq) * (2.0 ** (float(cents) / 1200.0))

            phase = _mono_vibrato_phase(
                f,
                t,
                sr,
                depth_cents=2.0 + 6.0 * motion,
                rate_hz=0.25 + 0.08 * i + rng.uniform(0.0, 0.05),
                seed=seed + i * 37,
            )

            sawish = (
                0.75 * np.sin(phase)
                + 0.25 * np.sin(2.0 * phase)
                + 0.12 * np.sin(3.0 * phase)
            ).astype(np.float32)

            x += sawish

        x /= float(len(detunes))
        x = _waveshape(x, drive=0.70 + 0.60 * brightness, fold=0.0)
        x *= env * amp * vel_gain

        y = _pan_stereo(x, pan)

        cutoff = 2300.0 + 8500.0 * brightness
        y = _svf_lowpass_stereo(y, sr, cutoff_hz=cutoff, res=0.10)

        y = _body_resonance_stereo(
            y,
            sr,
            amount=float(params.get("body", 0.34)),
            modes=[
                (220.0, 0.22),
                (440.0, 0.18),
                (880.0, 0.13),
                (1760.0, 0.08),
            ],
        )

        if float(params.get("chorus_mix", 0.42)) > 1e-6:
            y = _chorus_stereo(
                y,
                sr,
                rate_hz=0.16 + 0.20 * motion,
                depth_ms=12.0 + 10.0 * motion,
                mix=float(params.get("chorus_mix", 0.42)),
                seed=seed + 227,
            )

        if float(params.get("reverb_mix", 0.30)) > 1e-6:
            y = _schroeder_reverb_stereo(
                y,
                sr,
                mix=float(params.get("reverb_mix", 0.30)),
                room=float(params.get("reverb_room", 0.70)),
                predelay_ms=28.0,
                damp_hz=6500.0,
            )

        y = _soft_limiter_stereo(y, ceiling=0.98)
        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_string_pad"})


# ============================================================================
# FX
# ============================================================================

class Gain(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        g = float(params.get("gain", 1.0))

        try:
            if _native_has("gain"):
                y = NATIVE_DSP.gain(x, g)
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_gain"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        y = (x * g).astype(np.float32)

        extra = {"engine": "numpy_gain"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class Delay(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "delay_ms": {"type": "float", "default": 240.0, "min": 1.0, "max": 5000.0, "step": 1.0},
        "feedback": {"type": "float", "default": 0.28, "min": 0.0, "max": 0.98, "step": 0.01},
        "wet": {"type": "float", "default": 0.28, "min": 0.0, "max": 2.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        delay_ms = float(params.get("delay_ms", 240.0))
        feedback = float(np.clip(params.get("feedback", 0.28), 0.0, 0.98))
        wet = float(params.get("wet", 0.28))

        try:
            if _native_has("delay"):
                y = NATIVE_DSP.delay(
                    x,
                    sample_rate=sr,
                    delay_ms=delay_ms,
                    feedback=feedback,
                    wet=wet,
                )
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_delay"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        d = max(1, int(round(delay_ms * sr / 1000.0)))
        y = np.copy(x).astype(np.float32)

        if d < x.shape[0]:
            for i in range(d, x.shape[0]):
                y[i] += wet * x[i - d] + feedback * y[i - d]

        y = _soft_limiter_stereo(y, ceiling=0.98)

        extra = {"engine": "numpy_delay"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class OnePoleLP(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 5000.0, "min": 20.0, "max": 20000.0, "step": 20.0},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        cutoff_hz = float(params.get("cutoff_hz", 5000.0))
        wet = float(np.clip(params.get("wet", 1.0), 0.0, 1.0))

        y = _svf_lowpass_stereo(x, sr, cutoff_hz=cutoff_hz, res=0.0)
        out = ((1.0 - wet) * x + wet * y).astype(np.float32)

        return AudioBuffer(out.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_lowpass"})


class Highpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 120.0, "min": 20.0, "max": 12000.0, "step": 10.0},
        "q": {"type": "float", "default": 0.707, "min": 0.1, "max": 24.0, "step": 0.01},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        cutoff_hz = float(params.get("cutoff_hz", 120.0))
        q = float(params.get("q", 0.707))
        wet = float(np.clip(params.get("wet", 1.0), 0.0, 1.0))

        b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, cutoff_hz, q)
        y = _biquad_process_stereo(x, b0, b1, b2, a1, a2)

        out = ((1.0 - wet) * x + wet * y).astype(np.float32)

        return AudioBuffer(out.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_highpass"})


class Bandpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "freq_hz": {"type": "float", "default": 1200.0, "min": 20.0, "max": 18000.0, "step": 20.0},
        "q": {"type": "float", "default": 1.0, "min": 0.1, "max": 24.0, "step": 0.01},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        freq_hz = float(params.get("freq_hz", 1200.0))
        q = float(params.get("q", 1.0))
        wet = float(np.clip(params.get("wet", 1.0), 0.0, 1.0))

        b0, b1, b2, a1, a2 = _biquad_bandpass_coeff(sr, freq_hz, q)
        y = _biquad_process_stereo(x, b0, b1, b2, a1, a2)

        out = ((1.0 - wet) * x + wet * y).astype(np.float32)

        return AudioBuffer(out.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_bandpass"})


class SoftClip(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "drive": {"type": "float", "default": 1.0, "min": 0.0, "max": 12.0, "step": 0.05},
        "ceiling": {"type": "float", "default": 0.98, "min": 0.1, "max": 1.0, "step": 0.01},
        "normalize": {"type": "bool", "default": True},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        drive = float(params.get("drive", 1.0))
        ceiling = float(params.get("ceiling", 0.98))
        normalize = bool(params.get("normalize", True))

        try:
            if _native_has("soft_clip_normalize"):
                y = NATIVE_DSP.soft_clip_normalize(
                    x * max(0.0, drive),
                    ceiling=ceiling,
                    peak=ceiling,
                    only_if_over=not normalize,
                )
                return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "native_softclip"})
        except Exception as exc:
            native_error = str(exc)
        else:
            native_error = None

        y = np.tanh(x * max(0.0, drive)).astype(np.float32)

        if normalize:
            mx = float(np.max(np.abs(y))) if y.size else 0.0
            if mx > 1e-9:
                y *= ceiling / mx
        else:
            y = _soft_limiter_stereo(y, ceiling=ceiling)

        extra = {"engine": "numpy_softclip"}
        if native_error:
            extra["native_error"] = native_error

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta(extra)


class SoundPolish(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "input_gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
        "drive": {"type": "float", "default": 0.65, "min": 0.0, "max": 8.0, "step": 0.05},
        "warmth": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},
        "air": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},

        "highpass_hz": {"type": "float", "default": 28.0, "min": 10.0, "max": 1000.0, "step": 5.0},
        "lowpass_hz": {"type": "float", "default": 18000.0, "min": 1000.0, "max": 22000.0, "step": 50.0},

        "width_mix": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.7, "step": 0.01},
        "width_ms": {"type": "float", "default": 5.5, "min": 0.0, "max": 25.0, "step": 0.5},

        "output_gain": {"type": "float", "default": 0.95, "min": 0.0, "max": 2.0, "step": 0.01},
        "ceiling": {"type": "float", "default": 0.98, "min": 0.1, "max": 1.0, "step": 0.01},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        input_gain = float(params.get("input_gain", 1.0))
        drive = float(params.get("drive", 0.65))
        warmth = float(np.clip(params.get("warmth", 0.28), 0.0, 1.0))
        air = float(np.clip(params.get("air", 0.10), 0.0, 1.0))
        highpass_hz = float(params.get("highpass_hz", 28.0))
        lowpass_hz = float(params.get("lowpass_hz", 18000.0))
        output_gain = float(params.get("output_gain", 0.95))
        ceiling = float(params.get("ceiling", 0.98))
        seed = int(params.get("seed", 0))

        y = (x * input_gain).astype(np.float32)

        if highpass_hz > 10.0:
            b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, highpass_hz, 0.707)
            y = _biquad_process_stereo(y, b0, b1, b2, a1, a2)

        if drive > 1e-6:
            y = _waveshape(y, drive=0.65 + drive, fold=0.0)

        if warmth > 1e-6:
            warm = _svf_lowpass_stereo(
                y,
                sr,
                cutoff_hz=1200.0 + 2600.0 * (1.0 - warmth),
                res=0.12,
            )
            y = ((1.0 - 0.22 * warmth) * y + (0.22 * warmth) * warm).astype(np.float32)

        if air > 1e-6:
            hp = y.copy()
            b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, 5200.0, 0.707)
            hp = _biquad_process_stereo(hp, b0, b1, b2, a1, a2)
            y = (y + air * 0.22 * hp).astype(np.float32)

        y = _svf_lowpass_stereo(
            y,
            sr,
            cutoff_hz=float(np.clip(lowpass_hz, 1000.0, sr * 0.45)),
            res=0.0,
        )

        if float(params.get("width_mix", 0.08)) > 1e-6:
            y = _microshift_stereo(
                y,
                sr,
                amount_ms=float(params.get("width_ms", 5.5)),
                mix=float(params.get("width_mix", 0.08)),
                seed=seed + 401,
            )

        y = (y * output_gain).astype(np.float32)
        y = _soft_limiter_stereo(y, ceiling=ceiling)

        return AudioBuffer(y.astype(np.float32, copy=False), sr), _meta({"engine": "numpy_sound_polish"})


# ============================================================================
# Register blocks
# ============================================================================

BLOCKS.register("synth_keys", SynthKeys)
BLOCKS.register("guitar_pluck", GuitarPluck)
BLOCKS.register("bell_fm", BellFM)
BLOCKS.register("lead_synth", LeadSynth)

BLOCKS.register("brass_synth", BrassSynth)
BLOCKS.register("flute_synth", FluteSynth)
BLOCKS.register("clarinet_synth", ClarinetSynth)
BLOCKS.register("string_pad", StringPad)

BLOCKS.register("gain", Gain)
BLOCKS.register("delay", Delay)
BLOCKS.register("lowpass", OnePoleLP)
BLOCKS.register("highpass", Highpass)
BLOCKS.register("bandpass", Bandpass)
BLOCKS.register("softclip", SoftClip)
BLOCKS.register("sound_polish", SoundPolish)