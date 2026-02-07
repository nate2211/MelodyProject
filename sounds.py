import numpy as np
from typing import Any, Dict, Tuple, Optional, List

from pipeline import BaseBlock, BLOCKS, AudioBuffer, ensure_stereo

# ============================================================================
# sounds.py  (Production-ready, "Omni-ish" polish, NumPy-only)
#
# Goals:
#   - Cleaner oscillators (PolyBLEP) + better unison voicing
#   - Musical envelopes (curved ADSR) + velocity support
#   - Keytracked resonant filter + optional filter envelope
#   - Stereo movement (chorus / microshift) + lush but controlled reverb
#   - Pluck with better exciter + body + air + controlled sustain tail
#   - Bell with modal/inharmonic partials + strike + shimmer
#
# Constraints:
#   - NumPy only
#   - Stateless per buffer (no persistent filter states across calls)
#   - Works with your BaseBlock / AudioBuffer pipeline
# ============================================================================

_TWOPI = 2.0 * np.pi


# ----------------------------------------------------------------------------
# Note / pitch helpers
# ----------------------------------------------------------------------------
def _midi_from_freq(freq: float) -> float:
    freq = float(max(1e-6, freq))
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def _clamp(x: float, a: float, b: float) -> float:
    return float(np.clip(x, a, b))


# ----------------------------------------------------------------------------
# Wave normalization
# ----------------------------------------------------------------------------
_WAVE_ALIASES = {
    "sin": "sine", "sine": "sine",
    "tri": "triangle", "triangle": "triangle",
    "sq": "square", "sqr": "square", "square": "square",
    "pulse": "square", "pwm": "square",
    "saw": "saw", "sawtooth": "saw", "saw-tooth": "saw", "tooth": "saw",
}


def _norm_wave(w: Any, default: str = "sine") -> str:
    s = str(w or "").strip().lower()
    s = s.replace(" ", "").replace("_", "").replace("-", "")
    return _WAVE_ALIASES.get(s, default)


# ----------------------------------------------------------------------------
# PolyBLEP helpers (band-limited edges)
# ----------------------------------------------------------------------------
def _poly_blep(t: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(t, dtype=np.float32)

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
    return y


def _square_blep(phase01: np.ndarray, dt: float, pwm: float) -> np.ndarray:
    pwm = float(np.clip(pwm, 0.01, 0.99))
    y = np.where(phase01 < pwm, 1.0, -1.0).astype(np.float32)
    y += _poly_blep(phase01, dt)
    t2 = (phase01 - pwm) % 1.0
    y -= _poly_blep(t2, dt)
    return y


def _tri_from_square(square: np.ndarray) -> np.ndarray:
    # leaky-integrate to avoid runaway DC and keep it stable per-buffer
    y = np.cumsum(square).astype(np.float32)
    y -= np.mean(y)
    m = np.max(np.abs(y)) + 1e-8
    return (y / m).astype(np.float32)


# ----------------------------------------------------------------------------
# Shaping / color
# ----------------------------------------------------------------------------
def _waveshape(x: np.ndarray, drive: float, fold: float) -> np.ndarray:
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

    return y


def _tilt_eq(x: np.ndarray, tilt: float) -> np.ndarray:
    tilt = float(np.clip(tilt, -1.0, 1.0))
    if abs(tilt) < 1e-6:
        return x.astype(np.float32, copy=False)

    dx = np.empty_like(x, dtype=np.float32)
    dx[0] = x[0]
    dx[1:] = x[1:] - x[:-1]
    y = np.tanh(x + 1.15 * tilt * dx).astype(np.float32)
    return y


def _soft_limiter_stereo(x: np.ndarray, ceiling: float = 0.98) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    c = float(max(0.1, ceiling))
    # gentle ceiling: tanh scaled so ~c at large inputs
    y = np.tanh(x / c) * c
    return y.astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Stereo helpers
# ----------------------------------------------------------------------------
def _pan_stereo(x: np.ndarray, pan: float) -> np.ndarray:
    pan = float(np.clip(pan, -1.0, 1.0))
    l = np.sqrt(0.5 * (1.0 - pan))
    r = np.sqrt(0.5 * (1.0 + pan))
    return np.stack([x * l, x * r], axis=1).astype(np.float32, copy=False)


def _microshift_stereo(x: np.ndarray, sr: int, amount_ms: float, mix: float, seed: int) -> np.ndarray:
    """
    Tiny L/R offset + tiny modulation = perceived width without obvious chorus.
    """
    x = ensure_stereo(x).astype(np.float32, copy=False)
    mix = _clamp(float(mix), 0.0, 1.0)
    if mix <= 1e-6:
        return x

    sr = int(sr)
    n = x.shape[0]
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    amt = _clamp(float(amount_ms), 0.0, 25.0)
    base = int(round((amt / 1000.0) * sr))
    base = max(0, min(base, int(0.03 * sr)))

    # gentle, slow wander
    t = (np.arange(n, dtype=np.float32) / float(sr))
    rate = float(rng.uniform(0.08, 0.22))
    depth = int(max(1, round(0.25 * base))) if base > 0 else 1
    phL = float(rng.uniform(0.0, 2 * np.pi))
    phR = float(rng.uniform(0.0, 2 * np.pi))

    lfoL = np.sin(_TWOPI * rate * t + phL).astype(np.float32)
    lfoR = np.sin(_TWOPI * rate * t + phR).astype(np.float32)

    def _delay(inp: np.ndarray, lfo: np.ndarray) -> np.ndarray:
        out = np.empty_like(inp, dtype=np.float32)
        for i in range(n):
            d = base + int(round(depth * (0.5 * (1.0 + float(lfo[i])))))
            j = i - d
            out[i] = float(inp[j]) if j >= 0 else 0.0
        return out

    wetL = _delay(x[:, 0], lfoL)
    wetR = _delay(x[:, 1], lfoR)
    wet = np.stack([wetL, wetR], axis=1).astype(np.float32)
    wet = np.tanh(wet * 1.15).astype(np.float32)
    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Envelopes
# ----------------------------------------------------------------------------
def _adsr_env(n: int, sr: int, a: float, d: float, s: float, r: float, curve: float = 0.55) -> np.ndarray:
    """
    Curved ADSR:
      curve < 1 => snappier; curve > 1 => softer
    """
    n = int(max(1, n))
    sr = int(max(1, sr))
    a = max(0.0, float(a))
    d = max(0.0, float(d))
    s = float(np.clip(s, 0.0, 1.0))
    r = max(0.0, float(r))
    curve = float(np.clip(curve, 0.15, 2.5))

    aN = int(round(a * sr))
    dN = int(round(d * sr))
    rN = int(round(r * sr))
    aN = max(0, min(aN, n))
    dN = max(0, min(dN, n - aN))
    rN = max(0, min(rN, n))

    env = np.empty(n, dtype=np.float32)

    # attack
    if aN > 1:
        x = np.linspace(0.0, 1.0, aN, dtype=np.float32)
        env[:aN] = x ** (1.0 / curve)
    elif aN == 1:
        env[0] = 1.0
    else:  # aN == 0
        env[:0] = []  # no attack phase
        pass

    # decay to sustain
    idx = aN
    if dN > 1:
        x = np.linspace(0.0, 1.0, dN, dtype=np.float32)
        # Ensure decay starts from 1.0 (after attack peak) and goes to sustain level
        dec_start_val = 1.0 if aN > 0 else 0.0  # If no attack, decay starts from 0 to s
        if aN == 1 and dN > 0:  # If attack is 1 sample, ensure decay starts from 1.0
            dec_start_val = 1.0
        else:  # If aN > 1, the end of attack is 1.0
            dec_start_val = 1.0

        dec = (dec_start_val - s) * ((1.0 - x) ** curve) + s
        env[idx:idx + dN] = dec
    elif dN == 1:
        env[idx] = s
    idx += dN

    # sustain
    susN = max(0, n - idx)
    if susN > 0:
        env[idx:] = s

    # release (always applied at end of buffer; works for one-shot notes)
    if rN > 1:
        x = np.linspace(0.0, 1.0, rN, dtype=np.float32)
        # Release starts from the current value at n-rN
        release_start_val = env[max(0, n - rN - 1)] if n - rN - 1 >= 0 else s  # Fallback to sustain if too short
        rel = release_start_val * (1.0 - x) ** (1.0 / curve)
        env[-rN:] = rel
    elif rN == 1:
        env[-1] = 0.0  # Ensure it ends at 0

    # safety
    return np.clip(env, 0.0, 1.0).astype(np.float32)


def _exp_decay(n: int, sr: int, t60: float) -> np.ndarray:
    n = int(max(1, n))
    sr = int(max(1, sr))
    t60 = float(max(1e-4, t60))
    t = np.arange(n, dtype=np.float32) / float(sr)
    # 60 dB down => factor 0.001
    return np.exp(np.log(0.001) * (t / t60)).astype(np.float32)


# ----------------------------------------------------------------------------
# Drift / movement
# ----------------------------------------------------------------------------
def _analog_drift(t: np.ndarray, depth: float, rate_hz: float, seed: int) -> np.ndarray:
    depth = float(np.clip(depth, 0.0, 0.02))
    rate_hz = float(np.clip(rate_hz, 0.01, 2.0))
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    ph = float(rng.uniform(0.0, 2.0 * np.pi))
    lfo = np.sin(_TWOPI * rate_hz * t + ph).astype(np.float32)
    return (1.0 + depth * lfo).astype(np.float32)


# ----------------------------------------------------------------------------
# Filter: stable SVF lowpass (per-buffer state)
# ----------------------------------------------------------------------------
def _svf_lowpass_stereo(x: np.ndarray, sr: int, cutoff_hz: float, res: float) -> np.ndarray:
    """
    Chamberlin/TPT-ish SVF.
    res: 0..1 (more = more resonance)
    """
    x = ensure_stereo(x).astype(np.float32, copy=False)
    sr = int(sr)

    cutoff_hz = float(np.clip(cutoff_hz, 10.0, 0.49 * sr))
    res = float(np.clip(res, 0.0, 1.0))

    g = np.tan(np.pi * cutoff_hz / float(sr))
    g = float(np.clip(g, 0.0, 1.5))

    # Resonance mapping (avoid self-osc runaway)
    R = 1.15 - 1.02 * res  # ~1.15..0.13
    R = float(np.clip(R, 0.08, 1.5))

    y = np.empty_like(x, dtype=np.float32)

    ic1L = 0.0
    ic2L = 0.0
    ic1R = 0.0
    ic2R = 0.0
    den = (1.0 + g * (g + R))

    for i in range(x.shape[0]):
        inL = float(x[i, 0])
        inR = float(x[i, 1])

        # Left
        v0 = inL - R * ic2L
        v1 = (g * v0 + ic1L) / den
        v2 = ic2L + g * v1
        ic1L = 2.0 * v1 - ic1L
        ic2L = 2.0 * v2 - ic2L
        y[i, 0] = v2

        # Right
        v0 = inR - R * ic2R
        v1 = (g * v0 + ic1R) / den
        v2 = ic2R + g * v1
        ic1R = 2.0 * v1 - ic1R
        ic2R = 2.0 * v2 - ic2R
        y[i, 1] = v2

    return y.astype(np.float32, copy=False)


def _keytracked_cutoff(base_cutoff: float, freq: float, keytrack: float) -> float:
    base_cutoff = float(base_cutoff)
    keytrack = float(np.clip(keytrack, 0.0, 1.0))
    # Slightly adjust the exponent for a more musical keytracking curve
    return base_cutoff * (freq / 440.0) ** (0.9 * keytrack)


# ----------------------------------------------------------------------------
# Oscillator: PolyBLEP + unison + simple sync + FM/PM + PD + shaping
# ----------------------------------------------------------------------------
def _phase_distort(phase01: np.ndarray, amount: float) -> np.ndarray:
    a = float(np.clip(amount, -0.95, 0.95))
    if abs(a) < 1e-6:
        return phase01.astype(np.float32, copy=False)

    p = phase01.astype(np.float32, copy=False)
    bend = 0.5 + 0.45 * a
    out = np.empty_like(p, dtype=np.float32)
    m = p < bend
    out[m] = (p[m] / bend) * 0.5
    out[~m] = 0.5 + ((p[~m] - bend) / (1.0 - bend)) * 0.5
    return out


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
    sr = int(sr)
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

    # Unison detune distribution (slightly "center heavy" for nicer stacks)
    if unison == 1:
        detunes = np.array([0.0], dtype=np.float32)
        pans = np.array([0.0], dtype=np.float32)  # Not used directly here, but good for consistency
    else:
        idx = np.arange(unison, dtype=np.float32)
        det = idx - (unison - 1) / 2.0
        det /= (np.max(np.abs(det)) + 1e-8)
        # Use a slightly exponential spread for detune for more natural voicing
        detunes = np.sign(det) * (np.abs(det) ** 1.2) * detune_cents * spread

        # stereo-ish pan weights (if caller wants to pan voices later)
        pans = det.copy()
        pans /= (np.max(np.abs(pans)) + 1e-8)
        pans *= spread

    # Random phase offset for each unison voice for more organic sound
    ph_off = rng.uniform(0.0, 1.0, size=unison).astype(np.float32) * (0.35 * spread)
    out = np.zeros(n, dtype=np.float32)

    # Pre-calculate FM/PM modulation if active to avoid re-calculating inside loop
    _fm_mod = None
    _fm_depth = 0.0
    if fm_ratio > 0.0 and fm_index > 0.0:
        fm_f = freq * fm_ratio
        _fm_mod = np.sin(_TWOPI * fm_f * t).astype(np.float32)
        _fm_depth = (fm_index ** 1.03) * 0.055

    _pm_mod = None
    _pm_depth = 0.0
    if pm_amount > 0.0:
        # PM carrier frequency can be different, often a sub-multiple or slightly detuned
        pm_carrier_freq = freq * (0.5 + 0.1 * rng.uniform(-1.0, 1.0))
        _pm_mod = np.sin(_TWOPI * pm_carrier_freq * t).astype(np.float32)
        _pm_depth = (pm_amount ** 0.9) * 1.20

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

        if pd_amount != 0.0:
            phase01 = _phase_distort(phase01, pd_amount)

        if _fm_mod is not None:
            # Apply FM modulation with individual voice detune
            voice_fm_mod = np.sin(_TWOPI * (f * fm_ratio) * t).astype(np.float32)  # Recalculate per voice for detuning
            phase01 = (phase01 + _fm_depth * voice_fm_mod) % 1.0

        if _pm_mod is not None:
            # Apply PM modulation with individual voice detune
            voice_pm_mod = np.sin(_TWOPI * (f * (0.5 + 0.1 * rng.uniform(-1.0, 1.0))) * t).astype(np.float32)
            phase01 = (phase01 + (_pm_depth / _TWOPI) * voice_pm_mod) % 1.0

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


# ----------------------------------------------------------------------------
# Chorus (modulated delay) + Reverb (Schroeder-ish), tuned for "mix-ready"
# ----------------------------------------------------------------------------
def _chorus_stereo(x: np.ndarray, sr: int, *, rate_hz: float, depth_ms: float, mix: float, seed: int) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    sr = int(sr)

    mix = float(np.clip(mix, 0.0, 1.0))
    if mix <= 1e-6:
        return x

    rate_hz = float(np.clip(rate_hz, 0.02, 5.0))
    depth_ms = float(np.clip(depth_ms, 0.2, 30.0))

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    n = x.shape[0]
    t = (np.arange(n, dtype=np.float32) / float(sr))

    base_ms = 9.0  # Base delay for chorus
    base_s = int(round((base_ms / 1000.0) * sr))
    depth_s = int(round((depth_ms / 1000.0) * sr))
    depth_s = max(1, depth_s)  # Ensure at least 1 sample depth

    # LFOs for left and right channels, slightly detuned and phase offset
    rateL = rate_hz * (1.0 + rng.uniform(-0.05, 0.05))
    rateR = rate_hz * (1.0 + rng.uniform(-0.05, 0.05))
    phL = float(rng.uniform(0.0, 2 * np.pi))
    phR = float(rng.uniform(0.0, 2 * np.pi)) + np.pi / 2.0  # 90 degree phase shift for stereo width

    lfoL = np.sin(_TWOPI * rateL * t + phL).astype(np.float32)
    lfoR = np.sin(_TWOPI * rateR * t + phR).astype(np.float32)

    def _delay_chan(inp: np.ndarray, lfo: np.ndarray) -> np.ndarray:
        out = np.empty_like(inp, dtype=np.float32)
        # Using a fixed-size buffer for delay line, larger than max delay
        max_delay_samples = base_s + depth_s + 2
        delay_line = np.zeros(max_delay_samples, dtype=np.float32)
        write_idx = 0

        for i in range(n):
            # Current delay in samples, modulated by LFO
            current_delay = base_s + (0.5 * (1.0 + float(lfo[i]))) * depth_s
            di = int(current_delay)
            frac = current_delay - di

            # Read index for interpolation
            read_idx_0 = (write_idx - di + max_delay_samples) % max_delay_samples
            read_idx_1 = (write_idx - di - 1 + max_delay_samples) % max_delay_samples

            s0 = delay_line[read_idx_0]
            s1 = delay_line[read_idx_1]

            # Linear interpolation
            out[i] = (1.0 - frac) * s0 + frac * s1

            # Write current input to delay line
            delay_line[write_idx] = float(inp[i])
            write_idx = (write_idx + 1) % max_delay_samples
        return out

    wetL = _delay_chan(x[:, 0], lfoL)
    wetR = _delay_chan(x[:, 1], lfoR)
    wet = np.stack([wetL, wetR], axis=1).astype(np.float32)

    wet = np.tanh(wet * 1.10).astype(np.float32)  # Gentle saturation on wet signal
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
    sr = int(sr)

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

    # Comb filter delays (slightly adjusted for more diffusion and density)
    base = np.array([0.0297, 0.0371, 0.0411, 0.0437, 0.0461, 0.0503], dtype=np.float32)  # Added two more combs
    comb_delays = (base * (0.62 + 0.95 * room) * sr).astype(int)
    comb_delays = np.clip(comb_delays, 64, int(0.12 * sr))  # Slightly longer max delay

    fb = 0.74 + 0.18 * room  # 0.74..0.92, feedback amount, longer for larger rooms

    def _comb(inp: np.ndarray, d: int) -> np.ndarray:
        y = np.zeros_like(inp, dtype=np.float32)
        buf = np.zeros(d, dtype=np.float32)
        idx = 0
        for i in range(inp.shape[0]):
            v = float(inp[i]) + fb * float(buf[idx])
            y[i] = float(buf[idx])  # Output is the delayed signal
            buf[idx] = v  # Store the sum for next iteration
            idx = (idx + 1) % d
        return y

    def _allpass(inp: np.ndarray, d: int, g: float) -> np.ndarray:
        y = np.zeros_like(inp, dtype=np.float32)
        buf = np.zeros(d, dtype=np.float32)
        idx = 0
        for i in range(inp.shape[0]):
            b = float(buf[idx])
            x0 = float(inp[i])
            y0 = -g * x0 + b
            buf[idx] = x0 + g * y0
            y[i] = y0
            idx = (idx + 1) % d
        return y

    wet = np.zeros_like(wet_in, dtype=np.float32)
    ap1_d = int(max(32, int(0.0050 * sr)))
    ap2_d = int(max(32, int(0.0019 * sr)))
    ap3_d = int(max(32, int(0.0012 * sr)))  # Added a third allpass for more diffusion

    for ch in (0, 1):
        inp = wet_in[:, ch]
        comb_sum = np.zeros_like(inp, dtype=np.float32)
        for d in comb_delays:
            comb_sum += _comb(inp, int(d))
        comb_sum *= (1.0 / float(len(comb_delays)))  # Normalize sum

        ap = _allpass(comb_sum, ap1_d, 0.70)
        ap = _allpass(ap, ap2_d, 0.70)
        ap = _allpass(ap, ap3_d, 0.70)  # Process through third allpass
        wet[:, ch] = ap

    # damping (darken the tail) - applied after allpass filters
    wet = _svf_lowpass_stereo(wet, sr, cutoff_hz=damp_hz, res=0.0)
    wet = np.tanh(wet * 1.05).astype(np.float32)  # Gentle saturation for character

    return ((1.0 - mix) * x + mix * wet).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------
# Body resonances (parallel resonant-ish coloration)
# ----------------------------------------------------------------------------
def _body_resonance_stereo(x: np.ndarray, sr: int, amount: float, modes: List[Tuple[float, float]]) -> np.ndarray:
    x = ensure_stereo(x).astype(np.float32, copy=False)
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return x

    acc = np.zeros_like(x, dtype=np.float32)
    for (f, g) in modes:
        # Use a slightly higher resonance for body modes to make them ring more
        y = _svf_lowpass_stereo(x, sr, cutoff_hz=float(f), res=0.9)
        acc += float(g) * y

    out = x + amount * acc
    return out.astype(np.float32, copy=False)


# ============================================================================
# Instruments
# ============================================================================

class SynthKeys(BaseBlock):
    """
    More "produced" synth keys:
      - main: polyblep osc + unison
      - sub: sine with gentle saturation
      - noise: colored + filtered for air
      - sparkle: upper partials + subtle shimmer
      - ADSR + velocity shaping (if payload provides "vel")
      - keytracked resonant LP + filter envelope
      - microshift width + chorus + controlled reverb
      - final soft limiter
    """
    KIND = "instrument"
    PARAMS = {
        "wave": {"type": "choice", "default": "saw",
                 "choices": ["sine", "triangle", "square", "pulse", "saw", "sawtooth"]},

        "amp": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},  # Slightly higher default amp

        # Amp env
        "attack": {"type": "float", "default": 0.012, "min": 0.0, "max": 3.0, "step": 0.005},  # Slightly slower attack
        "decay": {"type": "float", "default": 0.150, "min": 0.0, "max": 6.0, "step": 0.01},  # Longer decay
        "sustain": {"type": "float", "default": 0.78, "min": 0.0, "max": 1.0, "step": 0.01},  # Higher sustain
        "release": {"type": "float", "default": 0.45, "min": 0.0, "max": 8.0, "step": 0.01},
        # Longer release for lush tails
        "env_curve": {"type": "float", "default": 0.65, "min": 0.15, "max": 2.5, "step": 0.05},  # Softer curve

        # Unison
        "unison": {"type": "int", "default": 9, "min": 1, "max": 16, "step": 1},  # More unison voices
        "detune_cents": {"type": "float", "default": 18.0, "min": 0.0, "max": 80.0, "step": 0.5},  # Wider detune
        "spread": {"type": "float", "default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01},  # Wider spread

        "pwm": {"type": "float", "default": 0.48, "min": 0.01, "max": 0.99, "step": 0.01},  # Subtle PWM

        # Movement / color
        "sync": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},  # Slightly more sync
        "sync_ratio": {"type": "float", "default": 3.2, "min": 1.0, "max": 12.0, "step": 0.25},  # Adjusted sync ratio
        "fm_ratio": {"type": "float", "default": 1.6, "min": 0.0, "max": 12.0, "step": 0.25},  # Adjusted FM ratio
        "fm_index": {"type": "float", "default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1},
        # Higher FM index for more complexity
        "pm_amount": {"type": "float", "default": 0.08, "min": 0.0, "max": 0.35, "step": 0.005},
        # More PM for subtle movement
        "pd_amount": {"type": "float", "default": 0.15, "min": -0.95, "max": 0.95, "step": 0.01},  # More PD for shaping

        "drive": {"type": "float", "default": 1.80, "min": 0.25, "max": 10.0, "step": 0.05},  # More drive for warmth
        "fold": {"type": "float", "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01},  # Subtle folding
        "tilt": {"type": "float", "default": 0.15, "min": -1.0, "max": 1.0, "step": 0.01},  # Slightly brighter tilt

        # Layers
        "sub": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},  # More sub
        "noise": {"type": "float", "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01},  # More air noise
        "sparkle": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01},  # More sparkle

        # Drift
        "drift": {"type": "float", "default": 0.0045, "min": 0.0, "max": 0.02, "step": 0.0005},  # Slightly more drift
        "drift_rate": {"type": "float", "default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01},
        # Slightly faster drift rate

        # Filter
        "cutoff_hz": {"type": "float", "default": 6500.0, "min": 80.0, "max": 20000.0, "step": 50.0},
        # Slightly lower cutoff
        "res": {"type": "float", "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01},  # More resonance
        "keytrack": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},  # More keytracking

        # Filter env
        "fenv_amt": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},
        # More filter envelope amount
        "fenv_attack": {"type": "float", "default": 0.003, "min": 0.0, "max": 2.0, "step": 0.005},
        # Faster filter attack
        "fenv_decay": {"type": "float", "default": 0.180, "min": 0.0, "max": 6.0, "step": 0.01},  # Longer filter decay

        # Stereo
        "width_mix": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},  # Wider microshift mix
        "width_ms": {"type": "float", "default": 8.0, "min": 0.0, "max": 25.0, "step": 0.5},
        # Slightly longer microshift delay

        "chorus_mix": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},  # More chorus mix
        "chorus_rate": {"type": "float", "default": 0.20, "min": 0.02, "max": 4.0, "step": 0.02},  # Slower chorus rate
        "chorus_depth_ms": {"type": "float", "default": 10.0, "min": 0.5, "max": 25.0, "step": 0.5},
        # Deeper chorus depth

        "reverb_mix": {"type": "float", "default": 0.25, "min": 0.0, "max": 0.8, "step": 0.01},  # More reverb mix
        "reverb_room": {"type": "float", "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01},  # Larger room size
        "reverb_predelay_ms": {"type": "float", "default": 18.0, "min": 0.0, "max": 80.0, "step": 1.0},
        # Longer predelay
        "reverb_damp_hz": {"type": "float", "default": 6500.0, "min": 1500.0, "max": 14000.0, "step": 100.0},
        # Slightly darker damping

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq = float(payload["freq"])
        dur = float(payload["duration"])
        sr = int(payload.get("sr", 48000))
        vel = float(payload.get("vel", 1.0))  # optional velocity 0..1

        n = max(1, int(round(dur * sr)))
        t = (np.arange(n, dtype=np.float32) / float(sr))
        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        wave_ui = params.get("wave", "saw")
        amp = float(params.get("amp", 0.35))
        pan = float(params.get("pan", 0.0))

        # amp ADSR (curved)
        env = _adsr_env(
            n, sr,
            a=float(params.get("attack", 0.012)),
            d=float(params.get("decay", 0.150)),
            s=float(params.get("sustain", 0.78)),
            r=float(params.get("release", 0.45)),
            curve=float(params.get("env_curve", 0.65)),
        )

        # gentle velocity to dynamics + brightness (feels more "real")
        vel = float(np.clip(vel, 0.0, 1.0))
        # More pronounced velocity response
        vel_gain = 0.45 + 0.85 * (vel ** 0.9)
        vel_bright = 0.65 + 0.75 * (vel ** 0.9)

        # drift factor
        drift_fac = _analog_drift(
            t,
            depth=float(params.get("drift", 0.0045)),
            rate_hz=float(params.get("drift_rate", 0.25)),
            seed=seed + 11,
        )

        # Main osc
        f0 = freq * float(1.0 + 0.0025 * rng.uniform(-1.0, 1.0))  # Slightly more initial pitch variation
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
            fm_index=float(params.get("fm_index", 5.5)) * vel_bright,  # Velocity affects FM index
            pm_amount=float(params.get("pm_amount", 0.08)) * vel_bright,  # Velocity affects PM amount
            pd_amount=float(params.get("pd_amount", 0.15)),
            drive=float(params.get("drive", 1.80)),
            fold=float(params.get("fold", 0.12)),
            tilt=float(params.get("tilt", 0.15)),
            seed=seed,
        )

        # Sub
        sub_amt = float(params.get("sub", 0.28))
        if sub_amt > 1e-6:
            sub = np.sin(_TWOPI * (0.5 * freq) * t).astype(np.float32)
            sub = np.tanh(sub * 1.35).astype(np.float32)  # Slightly more saturation on sub
        else:
            sub = 0.0

        # Noise (air): light, then lowpassed to remove fizz
        noise_amt = float(params.get("noise", 0.10))
        if noise_amt > 1e-6:
            nz = rng.normal(0.0, 1.0, size=n).astype(np.float32)
            nz = np.tanh(nz * 0.45).astype(np.float32)  # More saturation on noise
            nz_st = _pan_stereo(nz, 0.0)
            nz_st = _svf_lowpass_stereo(nz_st, sr, cutoff_hz=8000.0 * vel_bright,
                                        res=0.0)  # Velocity affects noise brightness
            nz = 0.5 * (nz_st[:, 0] + nz_st[:, 1])
        else:
            nz = 0.0

        # Sparkle partials
        spark = float(params.get("sparkle", 0.15))
        if spark > 1e-6:
            det = float(rng.uniform(-6.0, 6.0))  # Wider detune for sparkle
            f2 = (2.0 * freq) * (2.0 ** (det / 1200.0))
            f3 = (3.0 * freq) * (2.0 ** (-det / 1200.0))
            p2 = np.sin(_TWOPI * f2 * t).astype(np.float32)
            p3 = np.sin(_TWOPI * f3 * t + 1.1).astype(np.float32)
            sp = np.tanh((0.6 * p2 + 0.4 * p3) * 0.95).astype(np.float32)  # More saturation
            sp *= _exp_decay(n, sr, t60=max(0.15, 0.70 * dur))  # Longer sparkle fade
        else:
            sp = 0.0

        x = (x_main + sub_amt * sub + noise_amt * nz + spark * sp).astype(np.float32)
        x *= drift_fac

        # apply amp env + velocity
        x = (x * env * vel_gain).astype(np.float32)

        # to stereo
        st = _pan_stereo(np.tanh(x * 1.25).astype(np.float32), pan)  # More initial saturation

        # filter env (brightness punch)
        cutoff = float(params.get("cutoff_hz", 6500.0))
        keytrack = float(params.get("keytrack", 0.45))
        base_cutoff_eff = _keytracked_cutoff(cutoff, freq, keytrack)

        fenv_amt = float(params.get("fenv_amt", 0.45))
        if fenv_amt > 1e-6:
            fa = float(params.get("fenv_attack", 0.003))
            fd = float(params.get("fenv_decay", 0.180))
            # simple AD curve (no sustain) to mod cutoff
            fenv = _adsr_env(n, sr, a=fa, d=fd, s=0.0, r=0.0, curve=0.65)  # Softer filter env curve
            # Modulate cutoff dynamically at the start for a punch
            # The filter itself is per-buffer, so we'll apply this as an initial "effective" cutoff.
            # For per-sample filter modulation, _svf_lowpass_stereo would need to accept a time-varying cutoff.
            # Here, we'll make the initial cutoff brighter based on the peak of the fenv.
            cutoff_eff = base_cutoff_eff * (1.0 + fenv_amt * 2.5 * fenv[0])  # Initial punch
        else:
            cutoff_eff = base_cutoff_eff

        res = float(params.get("res", 0.40))
        st = _svf_lowpass_stereo(st, sr, cutoff_eff * vel_bright, res)  # Velocity affects filter brightness

        # stereo widening
        st = _microshift_stereo(
            st, sr,
            amount_ms=float(params.get("width_ms", 8.0)),
            mix=float(params.get("width_mix", 0.35)),
            seed=seed + 500,
        )

        # chorus (if desired)
        st = _chorus_stereo(
            st, sr,
            rate_hz=float(params.get("chorus_rate", 0.20)),
            depth_ms=float(params.get("chorus_depth_ms", 10.0)),
            mix=float(params.get("chorus_mix", 0.35)),
            seed=seed + 9001,
        )

        # reverb
        st = _schroeder_reverb_stereo(
            st, sr,
            mix=float(params.get("reverb_mix", 0.25)),
            room=float(params.get("reverb_room", 0.60)),
            predelay_ms=float(params.get("reverb_predelay_ms", 18.0)),
            damp_hz=float(params.get("reverb_damp_hz", 6500.0)),
        )

        # final tone + limiter
        st = np.tanh(st * 1.10).astype(np.float32)  # Final gentle saturation
        st = _soft_limiter_stereo(st, ceiling=0.98)
        st *= float(np.clip(amp, 0.0, 1.0))
        return AudioBuffer(st.astype(np.float32, copy=False), sr), {}


class GuitarPluck(BaseBlock):
    """
    More mix-ready pluck:
      - Better exciter (pick burst + shaped noise) into KS string
      - Damping + tone, plus a tasteful body resonance + air
      - Optional filter polish + gentle space
      - Final limiter for consistency
    """
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},  # Slightly higher default amp

        # KS / decay
        "decay": {"type": "float", "default": 0.989, "min": 0.90, "max": 0.9995, "step": 0.0005},  # Longer decay
        "damp": {"type": "float", "default": 0.99960, "min": 0.97, "max": 0.99995, "step": 0.00005},
        # Less damping for longer sustain
        "tone": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},  # Brighter tone

        # Exciter
        "pick_ms": {"type": "float", "default": 8.0, "min": 1.0, "max": 80.0, "step": 0.5},
        # Slightly longer pick transient
        "pick_bright": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},  # Brighter pick
        "pick_drive": {"type": "float", "default": 1.8, "min": 0.25, "max": 6.0, "step": 0.05},  # More pick drive

        # String nonlinearity
        "string_drive": {"type": "float", "default": 2.0, "min": 0.25, "max": 8.0, "step": 0.05},  # More string drive
        "string_fold": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.75, "step": 0.01},  # More string folding

        # Filter polish
        "cutoff_hz": {"type": "float", "default": 16000.0, "min": 80.0, "max": 20000.0, "step": 50.0},
        # Higher cutoff for open sound
        "res": {"type": "float", "default": 0.12, "min": 0.0, "max": 0.9, "step": 0.01},  # Subtle resonance
        "keytrack": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},  # Slightly more keytrack

        # Body + air
        "body": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01},  # More body resonance
        "air": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.8, "step": 0.01},  # More air

        # Space
        "width_mix": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},  # Wider microshift mix
        "width_ms": {"type": "float", "default": 7.0, "min": 0.0, "max": 25.0, "step": 0.5},
        # Slightly longer microshift delay
        "chorus_mix": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01},  # More chorus
        "reverb_mix": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01},  # More reverb
        "reverb_room": {"type": "float", "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01},  # Larger room size
        "reverb_predelay_ms": {"type": "float", "default": 12.0, "min": 0.0, "max": 80.0, "step": 1.0},
        # Longer predelay
        "reverb_damp_hz": {"type": "float", "default": 7000.0, "min": 1500.0, "max": 14000.0, "step": 100.0},
        # Slightly darker damping

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq = float(payload["freq"])
        dur = float(payload["duration"])
        sr = int(payload.get("sr", 48000))
        vel = float(payload.get("vel", 1.0))

        n = max(1, int(round(dur * sr)))

        amp = float(params.get("amp", 0.45))
        decay = float(params.get("decay", 0.989))
        damp = float(params.get("damp", 0.99960))
        tone = float(params.get("tone", 0.75))

        pick_ms = float(params.get("pick_ms", 8.0))
        pick_bright = float(params.get("pick_bright", 0.75))
        pick_drive = float(params.get("pick_drive", 1.8))

        pan = float(params.get("pan", 0.0))
        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.50 + 0.80 * (vel ** 0.85)  # More velocity gain
        vel_bright = 0.70 + 0.70 * (vel ** 0.9)  # More velocity brightness

        # KS delay length
        d = max(2, int(round(sr / max(20.0, freq))))

        # Exciter burst
        pick_n = max(1, int(round((pick_ms / 1000.0) * sr)))
        pick_n = min(pick_n, max(32, int(0.08 * sr)))
        t_pick = (np.arange(pick_n, dtype=np.float32) / float(sr))

        burst = rng.normal(0.0, 1.0, size=pick_n).astype(np.float32)
        burst = np.tanh(burst * (0.45 + 0.75 * pick_drive)).astype(np.float32)  # More drive on burst

        # brightness via high-ish partial + tilt
        partial = np.sin(_TWOPI * (freq * 2.0) * t_pick + float(rng.uniform(0, 2 * np.pi))).astype(np.float32)
        partial *= (0.22 + 0.28 * vel_bright)  # More velocity on partial brightness

        exc = (0.80 * burst + 0.20 * partial).astype(np.float32)  # Slightly more partial in exciter

        # shape: fast attack, short decay
        a = min(pick_n, max(4, int(0.0010 * sr)))  # Faster attack for pluck
        exc[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
        exc *= _exp_decay(pick_n, sr, t60=0.030 + 0.025 * (1.0 - pick_bright))  # Velocity affects exciter decay

        # pre-filter exciter so it doesn't fizz
        exc_st = _pan_stereo(exc, 0.0)
        exc_st = _svf_lowpass_stereo(exc_st, sr, cutoff_hz=7000.0 + 8000.0 * pick_bright * vel_bright, res=0.0)
        exc = 0.5 * (exc_st[:, 0] + exc_st[:, 1])

        # Initialize KS buffer
        buf = (exc[:d].copy() * (amp * vel_gain)).astype(np.float32)
        if buf.shape[0] < d:
            buf = np.pad(buf, (0, d - buf.shape[0]))

        str_drive = float(params.get("string_drive", 2.0))
        str_fold = float(params.get("string_fold", 0.15))

        y = np.zeros(n, dtype=np.float32)
        idx = 0
        for i in range(n):
            cur = float(buf[idx])
            nxt = float(buf[(idx + 1) % d])
            avg = 0.5 * (cur + nxt)

            # tone blend
            filt = float(tone) * avg + (1.0 - float(tone)) * cur

            # in-string nonlinearity
            filt = float(np.tanh(filt * (0.9 + str_drive)))  # More drive in string
            if str_fold > 1e-6:
                k = 1.0 + 7.0 * (float(str_fold) ** 0.9)  # More folding amount
                z = filt * k
                z = ((z + 1.0) % 4.0) - 2.0
                z = 2.0 - abs(z)
                filt = float(z - 1.0)

            buf[idx] = filt * float(decay) * float(damp)
            y[i] = cur
            idx = (idx + 1) % d

        st = _pan_stereo(y, pan)

        # filter polish
        cutoff = float(params.get("cutoff_hz", 16000.0))
        keytrack = float(params.get("keytrack", 0.25))
        cutoff_eff = _keytracked_cutoff(cutoff, freq, keytrack) * vel_bright
        res = float(params.get("res", 0.12))
        st = _svf_lowpass_stereo(st, sr, cutoff_eff, res)

        # body resonance
        body = float(params.get("body", 0.45))
        st = _body_resonance_stereo(
            st, sr, body,
            modes=[(100.0, 0.20), (200.0, 0.16), (300.0, 0.12), (600.0, 0.08)]  # Slightly adjusted body modes
        )

        # air (tiny lift by mixing a brighter filtered copy)
        air = float(params.get("air", 0.15))
        if air > 1e-6:
            bright = _svf_lowpass_stereo(st, sr, cutoff_hz=18000.0, res=0.0)  # Brighter air filter
            st = ((1.0 - air) * st + air * (1.15 * bright)).astype(np.float32)  # More air mix

        # width + chorus + reverb
        st = _microshift_stereo(st, sr, amount_ms=float(params.get("width_ms", 7.0)),
                                mix=float(params.get("width_mix", 0.25)), seed=seed + 500)

        st = _chorus_stereo(st, sr, rate_hz=0.20, depth_ms=8.0,  # Slower, deeper chorus
                            mix=float(params.get("chorus_mix", 0.15)), seed=seed + 777)

        st = _schroeder_reverb_stereo(
            st, sr,
            mix=float(params.get("reverb_mix", 0.15)),
            room=float(params.get("reverb_room", 0.50)),
            predelay_ms=float(params.get("reverb_predelay_ms", 12.0)),
            damp_hz=float(params.get("reverb_damp_hz", 7000.0)),
        )

        st = np.tanh(st * 1.12).astype(np.float32)  # Final gentle saturation
        st = _soft_limiter_stereo(st, ceiling=0.98)
        return AudioBuffer(st.astype(np.float32, copy=False), sr), {}


class BellFM(BaseBlock):
    """
    More realistic "Omni-ish" bell:
      - Modal-ish inharmonic partial stack (closer to real bell than pure FM)
      - Optional FM "bite" per partial
      - Strike transient + shimmer wash
      - Body resonance + lush space
      - Controlled tail damping
    """
    KIND = "instrument"
    PARAMS = {
        "amp": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},  # Slightly higher default amp

        # Modal partials / tone
        "brightness": {"type": "float", "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01},  # Brighter bell
        "inharm": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},  # More inharmonicity
        "decay": {"type": "float", "default": 3.5, "min": 0.05, "max": 12.0, "step": 0.05},  # Longer decay

        # FM bite (optional)
        "fm_ratio": {"type": "float", "default": 3.0, "min": 0.0, "max": 12.0, "step": 0.05},  # Adjusted FM ratio
        "fm_index": {"type": "float", "default": 8.0, "min": 0.0, "max": 24.0, "step": 0.1},
        # Higher FM index for more bite

        # Strike + shimmer
        "strike": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01},  # More strike
        "shimmer": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},  # More shimmer

        # Space / body
        "body": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},  # More body resonance
        "chorus_mix": {"type": "float", "default": 0.15, "min": 0.0, "max": 0.6, "step": 0.01},  # More chorus
        "reverb_mix": {"type": "float", "default": 0.30, "min": 0.0, "max": 0.8, "step": 0.01},  # More reverb
        "reverb_room": {"type": "float", "default": 0.70, "min": 0.0, "max": 1.0, "step": 0.01},  # Larger room size
        "reverb_predelay_ms": {"type": "float", "default": 22.0, "min": 0.0, "max": 80.0, "step": 1.0},
        # Longer predelay
        "reverb_damp_hz": {"type": "float", "default": 6000.0, "min": 1500.0, "max": 14000.0, "step": 100.0},
        # Darker damping for longer, smoother tails

        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        freq = float(payload["freq"])
        dur = float(payload["duration"])
        sr = int(payload.get("sr", 48000))
        vel = float(payload.get("vel", 1.0))

        n = max(1, int(round(dur * sr)))
        t = (np.arange(n, dtype=np.float32) / float(sr))

        amp = float(params.get("amp", 0.35))
        brightness = float(params.get("brightness", 0.65))
        inharm = float(params.get("inharm", 0.75))
        decay = float(params.get("decay", 3.5))

        fm_ratio = float(params.get("fm_ratio", 3.0))
        fm_index = float(params.get("fm_index", 8.0))

        strike = float(params.get("strike", 0.28))
        shimmer = float(params.get("shimmer", 0.25))

        body = float(params.get("body", 0.35))
        pan = float(params.get("pan", 0.0))
        seed = int(params.get("seed", 0))
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        vel = float(np.clip(vel, 0.0, 1.0))
        vel_gain = 0.50 + 0.80 * (vel ** 0.9)  # More velocity gain
        vel_bright = 0.60 + 0.80 * (vel ** 0.9)  # More velocity brightness

        # Bell-ish inharmonic ratios (morph with inharm)
        base_ratios = np.array([1.0, 2.02, 2.78, 3.95, 5.48, 6.90, 8.10], dtype=np.float32)  # Added a partial
        harm_ratios = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)  # Corresponding harmonic
        ratios = (1.0 - inharm) * harm_ratios + inharm * base_ratios

        # Gain distribution (more brightness -> more high partial energy)
        gains = np.array([1.00, 0.60, 0.40, 0.25, 0.16, 0.10, 0.06], dtype=np.float32)  # Adjusted gains
        gains = gains * (0.80 + 0.80 * brightness)  # More brightness scaling
        gains[0] *= 1.0 - 0.20 * brightness  # Fundamental slightly reduced with brightness

        # Partial decay distribution (high partials die faster)
        base_t60 = max(0.05, decay)
        t60s = base_t60 * (0.90 - 0.15 * brightness) / (
                    0.80 + 0.30 * np.arange(len(ratios), dtype=np.float32))  # More aggressive decay for high partials
        t60s = np.clip(t60s, 0.10, 20.0)

        x = np.zeros(n, dtype=np.float32)

        # FM bite: make it mild and brighter-dependent
        fm_on = (fm_ratio > 1e-6) and (fm_index > 1e-6)
        fm_index_eff = fm_index * (0.45 + 0.80 * brightness) * vel_bright  # More velocity on FM index
        fm_ratio_eff = fm_ratio

        for i, (r, g) in enumerate(zip(ratios, gains)):
            cents = float(rng.uniform(-5.0, 5.0))  # Wider random detune per partial
            f = (freq * float(r)) * (2.0 ** (cents / 1200.0))

            # modal sine core (clean, real)
            ph = float(rng.uniform(0.0, 2 * np.pi))
            part = np.sin(_TWOPI * f * t + ph).astype(np.float32)

            # optional mild FM to add "metal bite"
            if fm_on:
                mod = np.sin(_TWOPI * (f * fm_ratio_eff) * t + ph * 0.45).astype(
                    np.float32)  # Slightly different phase for FM mod
                part = np.sin(_TWOPI * f * t + ph + (0.10 * fm_index_eff * float(g)) * mod).astype(
                    np.float32)  # More FM depth

            # decay per partial
            env = _exp_decay(n, sr, t60=float(t60s[i]))
            x += (float(g) * part * env).astype(np.float32)

        x *= (amp * vel_gain)

        # strike transient (short noise burst, bright filtered)
        if strike > 1e-6:
            hit_n = int(max(16, round(0.015 * sr)))  # Slightly longer strike
            hit_n = min(hit_n, n)
            hit = rng.normal(0.0, 1.0, size=hit_n).astype(np.float32)
            hit = np.tanh(hit * 0.85).astype(np.float32)  # More drive on strike
            hit_env = _exp_decay(hit_n, sr, t60=0.030)  # Slightly longer strike decay
            hit *= hit_env
            hit_st = _pan_stereo(hit, float(rng.uniform(-0.1, 0.1)))  # Subtle random pan for strike
            hit_st = _svf_lowpass_stereo(hit_st, sr, cutoff_hz=12000.0 * vel_bright, res=0.0)
            hit_st = np.tanh(hit_st * 1.35).astype(np.float32)  # More saturation on strike
        else:
            hit_st = np.zeros((0, 2), dtype=np.float32)

        st = _pan_stereo(x, pan)
        if hit_st.shape[0] > 0:
            st[:hit_st.shape[0]] += strike * hit_st

        # shimmer wash (octave-ish, quick decay)
        if shimmer > 1e-6:
            up = np.sin(_TWOPI * (freq * 2.05) * t + float(rng.uniform(0, 2 * np.pi))).astype(
                np.float32)  # Slightly detuned octave
            up *= _exp_decay(n, sr, t60=max(0.15, 0.70 * decay))  # Longer shimmer decay
            st += shimmer * _pan_stereo(np.tanh(up * 0.45).astype(np.float32),
                                        -pan * 0.35)  # More shimmer saturation and pan

        # body resonance
        st = _body_resonance_stereo(st, sr, body,
                                    modes=[(600.0, 0.10), (1200.0, 0.08), (2400.0, 0.06)])  # Adjusted body modes

        # gentle LP polish
        st = _svf_lowpass_stereo(st, sr, cutoff_hz=16000.0 - 6000.0 * (1.0 - brightness),
                                 res=0.0)  # More dynamic LP for brightness

        # chorus + reverb
        st = _chorus_stereo(st, sr, rate_hz=0.18, depth_ms=9.0,  # Slower, deeper chorus
                            mix=float(params.get("chorus_mix", 0.15)), seed=seed + 202)

        st = _schroeder_reverb_stereo(
            st, sr,
            mix=float(params.get("reverb_mix", 0.30)),
            room=float(params.get("reverb_room", 0.70)),
            predelay_ms=float(params.get("reverb_predelay_ms", 22.0)),
            damp_hz=float(params.get("reverb_damp_hz", 6000.0)),
        )

        st = np.tanh(st * 1.12).astype(np.float32)  # Final gentle saturation
        st = _soft_limiter_stereo(st, ceiling=0.98)
        return AudioBuffer(st.astype(np.float32, copy=False), sr), {}


# ============================================================================
# FX blocks (simple, stable, mix-friendly)
# ============================================================================

class Gain(BaseBlock):
    KIND = "fx"
    PARAMS = {"gain_db": {"type": "float", "default": 0.0, "min": -60.0, "max": 24.0, "step": 0.5}}

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        gdb = float(params.get("gain_db", 0.0))
        g = 10.0 ** (gdb / 20.0)
        y = ensure_stereo(payload.data).astype(np.float32, copy=False) * g
        return AudioBuffer(y.astype(np.float32, copy=False), payload.sr), {}


class Delay(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "time_ms": {"type": "float", "default": 350.0, "min": 1.0, "max": 2000.0, "step": 5.0},
        # Longer default delay time
        "feedback": {"type": "float", "default": 0.55, "min": 0.0, "max": 0.95, "step": 0.01},  # More feedback
        "mix": {"type": "float", "default": 0.40, "min": 0.0, "max": 1.0, "step": 0.01},  # Higher mix
        "lowpass_hz": {"type": "float", "default": 8000.0, "min": 200.0, "max": 20000.0, "step": 100.0},
        # Slightly darker repeats
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        time_ms = float(params.get("time_ms", 350.0))
        fb = float(np.clip(float(params.get("feedback", 0.55)), 0.0, 0.95))
        mix = float(np.clip(float(params.get("mix", 0.40)), 0.0, 1.0))
        lp = float(params.get("lowpass_hz", 8000.0))

        d = max(1, int(round((time_ms / 1000.0) * sr)))
        y = x.copy()

        # feedback delay (simple, stable)
        for i in range(d, y.shape[0]):
            y[i] += y[i - d] * fb

        # tame repeats
        y = _svf_lowpass_stereo(y, sr, cutoff_hz=lp, res=0.0)
        out = np.tanh((1.0 - mix) * x + mix * y).astype(np.float32)  # Gentle tanh on output
        return AudioBuffer(out.astype(np.float32, copy=False), sr), {}


class OnePoleLP(BaseBlock):
    KIND = "fx"
    PARAMS = {"cutoff_hz": {"type": "float", "default": 5000.0, "min": 50.0, "max": 20000.0,
                            "step": 50.0}}  # Slightly lower default cutoff

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        fc = float(np.clip(float(params.get("cutoff_hz", 5000.0)), 10.0, 0.49 * sr))

        alpha = 1.0 - np.exp(-_TWOPI * fc / float(sr))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        y = np.empty_like(x, dtype=np.float32)
        # Initialize y0 with the first sample to avoid harsh transient
        y0 = x[0].astype(np.float32, copy=True)
        y[0] = y0
        for i in range(1, x.shape[0]):
            y0 = y0 + alpha * (x[i] - y0)
            y[i] = y0

        return AudioBuffer(y.astype(np.float32, copy=False), sr), {}


class SoftClip(BaseBlock):
    KIND = "fx"
    PARAMS = {"drive": {"type": "float", "default": 2.0, "min": 0.1, "max": 10.0,
                        "step": 0.1}}  # More default drive for warmth

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        drive = float(params.get("drive", 2.0))
        d = 0.35 + (float(np.clip(drive, 0.0, 20.0)) ** 1.1)
        y = np.tanh(ensure_stereo(payload.data).astype(np.float32, copy=False) * d).astype(np.float32)
        return AudioBuffer(y.astype(np.float32, copy=False), payload.sr), {}


# ----------------------------------------------------------------------------
# Biquad helpers (HP / BP)
# ----------------------------------------------------------------------------
def _biquad_process_mono(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x, dtype=np.float32)
    x1 = 0.0;
    x2 = 0.0
    y1 = 0.0;
    y2 = 0.0
    for i in range(x.shape[0]):
        x0 = float(x[i])
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = y0
        x2 = x1;
        x1 = x0
        y2 = y1;
        y1 = y0
    return y.astype(np.float32)


def _biquad_process_stereo(x: np.ndarray, b0: float, b1: float, b2: float, a1: float, a2: float) -> np.ndarray:
    yL = _biquad_process_mono(x[:, 0], b0, b1, b2, a1, a2)
    yR = _biquad_process_mono(x[:, 1], b0, b1, b2, a1, a2)
    return np.stack([yL, yR], axis=1).astype(np.float32)


def _biquad_highpass_coeff(sr: int, f0: float, q: float) -> Tuple[float, float, float, float, float]:
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    q = float(np.clip(q, 0.05, 24.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)
    alpha = s / (2.0 * q)

    b0 = (1.0 + c) / 2.0
    b1 = -(1.0 + c)
    b2 = (1.0 + c) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * c
    a2 = 1.0 - alpha

    b0 /= a0;
    b1 /= a0;
    b2 /= a0
    a1 /= a0;
    a2 /= a0
    return (float(b0), float(b1), float(b2), float(a1), float(a2))


def _biquad_bandpass_coeff(sr: int, f0: float, q: float) -> Tuple[float, float, float, float, float]:
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    q = float(np.clip(q, 0.05, 24.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)
    alpha = s / (2.0 * q)

    b0 = alpha
    b1 = 0.0
    b2 = -alpha
    a0 = 1.0 + alpha
    a1 = -2.0 * c
    a2 = 1.0 - alpha

    b0 /= a0;
    b1 /= a0;
    b2 /= a0
    a1 /= a0;
    a2 /= a0
    return (float(b0), float(b1), float(b2), float(a1), float(a2))


class Highpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 80.0, "min": 10.0, "max": 20000.0, "step": 10.0},
        # Lower default HP cutoff
        "q": {"type": "float", "default": 0.8, "min": 0.1, "max": 12.0, "step": 0.05},  # Slightly higher Q
        "drive": {"type": "float", "default": 0.5, "min": 0.0, "max": 6.0, "step": 0.1},  # Less drive by default
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        fc = float(params.get("cutoff_hz", 80.0))
        q = float(params.get("q", 0.8))
        drive = float(params.get("drive", 0.5))

        b0, b1, b2, a1, a2 = _biquad_highpass_coeff(sr, fc, q)
        y = _biquad_process_stereo(x, b0, b1, b2, a1, a2)

        if drive > 1e-6:
            d = 0.35 + (float(np.clip(drive, 0.0, 20.0)) ** 1.1)
            y = np.tanh(y * d).astype(np.float32)

        return AudioBuffer(y.astype(np.float32, copy=False), sr), {}


class Bandpass(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "center_hz": {"type": "float", "default": 1500.0, "min": 20.0, "max": 20000.0, "step": 10.0},
        # Slightly higher center freq
        "q": {"type": "float", "default": 1.2, "min": 0.1, "max": 24.0, "step": 0.05},  # Slightly higher Q
        "gain_db": {"type": "float", "default": 0.0, "min": -24.0, "max": 24.0, "step": 0.5},
        "drive": {"type": "float", "default": 0.0, "min": 0.0, "max": 6.0, "step": 0.1},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        f0 = float(params.get("center_hz", 1500.0))
        q = float(params.get("q", 1.2))
        gain_db = float(params.get("gain_db", 0.0))
        drive = float(params.get("drive", 0.0))

        b0, b1, b2, a1, a2 = _biquad_bandpass_coeff(sr, f0, q)
        y = _biquad_process_stereo(x, b0, b1, b2, a1, a2)

        if abs(gain_db) > 1e-6:
            g = 10.0 ** (gain_db / 20.0)
            y = (y * g).astype(np.float32)

        if drive > 1e-6:
            d = 0.35 + (float(np.clip(drive, 0.0, 20.0)) ** 1.1)
            y = np.tanh(y * d).astype(np.float32)

        return AudioBuffer(y.astype(np.float32, copy=False), sr), {}


# ============================================================================
# Register blocks
# ============================================================================
BLOCKS.register("synth_keys", SynthKeys)
BLOCKS.register("guitar_pluck", GuitarPluck)
BLOCKS.register("bell_fm", BellFM)

BLOCKS.register("gain", Gain)
BLOCKS.register("delay", Delay)
BLOCKS.register("lowpass", OnePoleLP)
BLOCKS.register("highpass", Highpass)
BLOCKS.register("bandpass", Bandpass)
BLOCKS.register("softclip", SoftClip)
