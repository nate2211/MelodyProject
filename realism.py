# ---- Instrument Finishers ----------------------------------------------------
# Drop into realism.py (below helpers) OR put into a new file and import it.

import numpy as np
from typing import Any, Dict, Tuple
from pipeline import BaseBlock, BLOCKS, AudioBuffer, ensure_stereo

_TWOPI = 2.0 * np.pi


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), lo, hi))


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _mix(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    m = _clamp(mix, 0.0, 1.0)
    return ((1.0 - m) * dry + m * wet).astype(np.float32)


def _softsat(x: np.ndarray, drive: float) -> np.ndarray:
    drive = _clamp(drive, 0.01, 20.0)
    d = 0.35 + (drive ** 1.15)
    return np.tanh(x * d).astype(np.float32)


def _onepole_lp(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sr = int(sr)
    fc = _clamp(cutoff_hz, 10.0, 0.49 * sr)
    a = 1.0 - np.exp(-_TWOPI * fc / sr)
    a = float(np.clip(a, 0.0, 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y0 = x[0].copy()
    y[0] = y0
    for i in range(1, x.shape[0]):
        y0 = y0 + a * (x[i] - y0)
        y[i] = y0
    return y


def _onepole_hp(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    lp = _onepole_lp(x, sr, cutoff_hz)
    return (x - lp).astype(np.float32)


# ----------------------------- Biquad EQ -------------------------------------
def _biquad_coeff_peaking(sr: int, f0: float, q: float, gain_db: float):
    # RBJ Audio EQ Cookbook (peaking EQ)
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    q = float(np.clip(q, 0.05, 24.0))
    A = 10.0 ** (gain_db / 40.0)
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

    # normalize
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    return (b0, b1, b2, a1, a2)


def _biquad_coeff_lowpass(sr: int, f0: float, q: float):
    # RBJ lowpass
    sr = float(sr)
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    q = float(np.clip(q, 0.05, 24.0))
    w0 = _TWOPI * f0 / sr
    c = np.cos(w0)
    s = np.sin(w0)
    alpha = s / (2.0 * q)

    b0 = (1.0 - c) / 2.0
    b1 = 1.0 - c
    b2 = (1.0 - c) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * c
    a2 = 1.0 - alpha

    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0
    return (b0, b1, b2, a1, a2)


def _biquad_process_mono(x: np.ndarray, b0, b1, b2, a1, a2) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    y = np.empty_like(x, dtype=np.float32)
    x1 = 0.0
    x2 = 0.0
    y1 = 0.0
    y2 = 0.0
    for i in range(x.shape[0]):
        x0 = float(x[i])
        y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[i] = y0
        x2 = x1
        x1 = x0
        y2 = y1
        y1 = y0
    return y.astype(np.float32)


def _biquad_process_stereo(x: np.ndarray, coeffs) -> np.ndarray:
    b0, b1, b2, a1, a2 = coeffs
    yL = _biquad_process_mono(x[:, 0], b0, b1, b2, a1, a2)
    yR = _biquad_process_mono(x[:, 1], b0, b1, b2, a1, a2)
    return np.stack([yL, yR], axis=1).astype(np.float32)


def _apply_peaks(x: np.ndarray, sr: int, peaks: list[tuple[float, float, float]]) -> np.ndarray:
    # peaks: (freq_hz, q, gain_db)
    y = x
    for f, q, g in peaks:
        y = _biquad_process_stereo(y, _biquad_coeff_peaking(sr, f, q, g))
    return y.astype(np.float32)


# -------------------------- Transient Shaper ---------------------------------
def _transient_shaper(x: np.ndarray, sr: int, attack: float, sustain: float) -> np.ndarray:
    """
    attack: -1..+1  (boost/reduce attack)
    sustain: -1..+1 (boost/reduce sustain/body)
    """
    attack = _clamp(attack, -1.0, 1.0)
    sustain = _clamp(sustain, -1.0, 1.0)

    # envelope follower on mono abs
    mono = 0.5 * (np.abs(x[:, 0]) + np.abs(x[:, 1]))
    env = np.zeros_like(mono, dtype=np.float32)

    att_t = 0.005
    rel_t = 0.080
    a_a = np.exp(-1.0 / (sr * att_t))
    a_r = np.exp(-1.0 / (sr * rel_t))

    e = 0.0
    for i in range(mono.shape[0]):
        v = float(mono[i])
        if v > e:
            e = a_a * e + (1.0 - a_a) * v
        else:
            e = a_r * e + (1.0 - a_r) * v
        env[i] = e

    # attack detector = positive slope of env
    d = np.zeros_like(env)
    d[1:] = env[1:] - env[:-1]
    d = np.maximum(d, 0.0)

    # normalize attack signal for stability
    dnorm = d / (np.max(d) + 1e-8)

    # gains
    g_attack = 1.0 + 1.2 * attack * dnorm
    g_sustain = 1.0 + 0.8 * sustain * (env / (np.max(env) + 1e-8))

    g = (g_attack * g_sustain).astype(np.float32)
    y = x * g[:, None]
    return y.astype(np.float32)


# ---------------------------- Simple Comp ------------------------------------
def _compress_stereo(x: np.ndarray, sr: int, thr_db: float, ratio: float, attack_ms: float, release_ms: float, makeup_db: float):
    thr = _db_to_gain(thr_db)
    ratio = _clamp(ratio, 1.0, 50.0)
    att = _clamp(attack_ms, 0.1, 2000.0) / 1000.0
    rel = _clamp(release_ms, 1.0, 5000.0) / 1000.0
    makeup = _db_to_gain(makeup_db)

    mono = 0.5 * (np.abs(x[:, 0]) + np.abs(x[:, 1]))
    env = np.zeros_like(mono, dtype=np.float32)

    a_a = np.exp(-1.0 / (sr * att)) if att > 0 else 0.0
    a_r = np.exp(-1.0 / (sr * rel)) if rel > 0 else 0.0

    e = 0.0
    for i in range(mono.shape[0]):
        v = float(mono[i])
        if v > e:
            e = a_a * e + (1.0 - a_a) * v
        else:
            e = a_r * e + (1.0 - a_r) * v
        env[i] = e

    g = np.ones_like(env, dtype=np.float32)
    over = env > thr
    if np.any(over):
        env_db = 20.0 * np.log10(env[over] + 1e-12)
        thr_db2 = 20.0 * np.log10(thr + 1e-12)
        gain_db = thr_db2 + (env_db - thr_db2) / ratio - env_db
        g[over] = (10.0 ** (gain_db / 20.0)).astype(np.float32)

    y = x * g[:, None] * makeup
    return y.astype(np.float32)


# ---------------------------- Comb/Resonator ---------------------------------
def _comb_resonator_mono(x: np.ndarray, delay_samp: int, feedback: float) -> np.ndarray:
    delay_samp = int(max(1, delay_samp))
    feedback = _clamp(feedback, 0.0, 0.98)
    buf = np.zeros(delay_samp, dtype=np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    w = 0
    for i in range(x.shape[0]):
        d = buf[w]
        out = x[i] + feedback * d
        y[i] = out
        buf[w] = out
        w = (w + 1) % delay_samp
    return y.astype(np.float32)


def _string_resonance(x: np.ndarray, sr: int, freq_hint: float, amount: float) -> np.ndarray:
    """
    Adds short comb resonance around freq_hint (Hz). Works even if freq_hint is wrong.
    """
    amount = _clamp(amount, 0.0, 1.0)
    if amount <= 1e-6:
        return x.astype(np.float32, copy=False)

    # choose a few nearby combs
    f = max(40.0, float(freq_hint))
    delays = []
    for mul in (1.0, 0.5, 2.0):
        dd = int(round(sr / (f * mul)))
        delays.append(int(np.clip(dd, 12, int(0.06 * sr))))  # <=60ms

    fb = 0.35 + 0.55 * amount
    y = x.copy()
    for d in delays:
        y[:, 0] = _comb_resonator_mono(y[:, 0], d, fb)
        y[:, 1] = _comb_resonator_mono(y[:, 1], d + 3, fb * 0.98)  # tiny stereo offset

    # keep it under control
    y = np.tanh(y * (1.0 + 0.6 * amount)).astype(np.float32)
    return y


# =============================================================================
# Guitar String Finisher
# =============================================================================
class GuitarString(BaseBlock):
    """
    "Stringy" guitar polish:
      - tight lowcut, cab-ish hi roll, presence peak
      - transient bite (attack up)
      - string resonance (comb)
      - mild compression + saturation
    """
    KIND = "fx"
    PARAMS = {
        "string_freq_hint": {"type": "float", "default": 110.0, "min": 40.0, "max": 440.0, "step": 1.0},
        "string_amount": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01},

        "lowcut_hz": {"type": "float", "default": 85.0, "min": 0.0, "max": 400.0, "step": 5.0},
        "hi_cut_hz": {"type": "float", "default": 7500.0, "min": 1500.0, "max": 18000.0, "step": 100.0},
        "presence_db": {"type": "float", "default": 3.0, "min": -6.0, "max": 12.0, "step": 0.5},

        "attack": {"type": "float", "default": 0.45, "min": -1.0, "max": 1.0, "step": 0.05},
        "sustain": {"type": "float", "default": -0.10, "min": -1.0, "max": 1.0, "step": 0.05},

        "comp_thresh_db": {"type": "float", "default": -18.0, "min": -50.0, "max": 0.0, "step": 0.5},
        "comp_ratio": {"type": "float", "default": 3.5, "min": 1.0, "max": 12.0, "step": 0.25},
        "drive": {"type": "float", "default": 1.6, "min": 0.1, "max": 8.0, "step": 0.1},

        "mix": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        f_hint = float(params.get("string_freq_hint", 110.0))
        s_amt = float(params.get("string_amount", 0.55))
        lowcut = float(params.get("lowcut_hz", 85.0))
        hicut = float(params.get("hi_cut_hz", 7500.0))
        pres = float(params.get("presence_db", 3.0))
        atk = float(params.get("attack", 0.45))
        sus = float(params.get("sustain", -0.10))
        thr = float(params.get("comp_thresh_db", -18.0))
        ratio = float(params.get("comp_ratio", 3.5))
        drive = float(params.get("drive", 1.6))
        mix = float(params.get("mix", 1.0))

        wet = x.copy()
        if lowcut > 0:
            wet[:, 0] = _onepole_hp(wet[:, 0], sr, lowcut)
            wet[:, 1] = _onepole_hp(wet[:, 1], sr, lowcut)

        # cab-ish hi roll (biquad lowpass feels nicer than 1-pole)
        wet = _biquad_process_stereo(wet, _biquad_coeff_lowpass(sr, hicut, q=0.707))

        # presence around 3.2k
        wet = _apply_peaks(wet, sr, [(3200.0, 1.0, pres)])

        # transient shaping
        wet = _transient_shaper(wet, sr, attack=atk, sustain=sus)

        # string resonance
        wet = _string_resonance(wet, sr, f_hint, s_amt)

        # compression + saturation
        wet = _compress_stereo(wet, sr, thr_db=thr, ratio=ratio, attack_ms=6.0, release_ms=90.0, makeup_db=2.0)
        wet = _softsat(wet, drive)

        out = _mix(x, wet, mix)
        out = np.tanh(out * 1.05).astype(np.float32)
        return AudioBuffer(out, sr), {}


# =============================================================================
# Piano Key Finisher (bright "hammer/key" clarity)
# =============================================================================
class PianoKey(BaseBlock):
    """
    Piano-key polish:
      - cleans sub, adds "hammer" attack click zone (2k-6k)
      - tames harsh highs, mild glue compression
      - optional "felt noise" (tiny) and stereo widen
    """
    KIND = "fx"
    PARAMS = {
        "lowcut_hz": {"type": "float", "default": 45.0, "min": 0.0, "max": 300.0, "step": 5.0},
        "hammer_db": {"type": "float", "default": 4.0, "min": -6.0, "max": 12.0, "step": 0.5},
        "body_db": {"type": "float", "default": 2.0, "min": -12.0, "max": 12.0, "step": 0.5},
        "air_lp_hz": {"type": "float", "default": 15500.0, "min": 4000.0, "max": 20000.0, "step": 200.0},

        "attack": {"type": "float", "default": 0.35, "min": -1.0, "max": 1.0, "step": 0.05},
        "sustain": {"type": "float", "default": 0.05, "min": -1.0, "max": 1.0, "step": 0.05},

        "comp_thresh_db": {"type": "float", "default": -22.0, "min": -50.0, "max": 0.0, "step": 0.5},
        "comp_ratio": {"type": "float", "default": 2.5, "min": 1.0, "max": 10.0, "step": 0.25},
        "drive": {"type": "float", "default": 1.25, "min": 0.1, "max": 6.0, "step": 0.1},

        "key_noise": {"type": "float", "default": 0.0015, "min": 0.0, "max": 0.02, "step": 0.0005},
        "seed": {"type": "int", "default": 0, "min": 0, "max": 999999, "step": 1},

        "mix": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        lowcut = float(params.get("lowcut_hz", 45.0))
        hammer_db = float(params.get("hammer_db", 4.0))
        body_db = float(params.get("body_db", 2.0))
        lp_hz = float(params.get("air_lp_hz", 15500.0))
        atk = float(params.get("attack", 0.35))
        sus = float(params.get("sustain", 0.05))
        thr = float(params.get("comp_thresh_db", -22.0))
        ratio = float(params.get("comp_ratio", 2.5))
        drive = float(params.get("drive", 1.25))
        key_noise = float(params.get("key_noise", 0.0015))
        seed = int(params.get("seed", 0))
        mix = float(params.get("mix", 1.0))

        wet = x.copy()
        if lowcut > 0:
            wet[:, 0] = _onepole_hp(wet[:, 0], sr, lowcut)
            wet[:, 1] = _onepole_hp(wet[:, 1], sr, lowcut)

        # body around 220Hz + 440Hz (gentle), hammer around 3.5k + 6k
        wet = _apply_peaks(
            wet, sr,
            [
                (220.0, 1.0, body_db * 0.55),
                (440.0, 1.0, body_db * 0.45),
                (3500.0, 1.0, hammer_db),
                (6000.0, 1.2, hammer_db * 0.45),
            ],
        )

        # transient shaping for "key/hammer"
        wet = _transient_shaper(wet, sr, attack=atk, sustain=sus)

        # gentle glue compression + light saturation
        wet = _compress_stereo(wet, sr, thr_db=thr, ratio=ratio, attack_ms=4.0, release_ms=120.0, makeup_db=2.0)
        wet = _softsat(wet, drive)

        # tame top just a bit
        wet = _biquad_process_stereo(wet, _biquad_coeff_lowpass(sr, lp_hz, q=0.707))

        # optional key noise (very low, adds realism)
        if key_noise > 0:
            rng = np.random.RandomState(seed & 0xFFFFFFFF)
            n = rng.normal(0.0, 1.0, size=wet.shape).astype(np.float32)
            # keep noise mostly in upper mids
            n[:, 0] = _onepole_hp(n[:, 0], sr, 1800.0)
            n[:, 1] = _onepole_hp(n[:, 1], sr, 1800.0)
            wet = wet + (key_noise * 0.5) * n

        out = _mix(x, wet, mix)
        out = np.tanh(out * 1.03).astype(np.float32)
        return AudioBuffer(out, sr), {}


# =============================================================================
# Horn Finisher (brass-ish formants + bite + glue)
# =============================================================================
class Horn(BaseBlock):
    """
    Horn-ish polish:
      - band-focus (reduce sub + extreme highs)
      - formant-ish peaks (700/1200/2500) + bite
      - compression + saturation for "brass"
    """
    KIND = "fx"
    PARAMS = {
        "lowcut_hz": {"type": "float", "default": 90.0, "min": 0.0, "max": 500.0, "step": 5.0},
        "hi_cut_hz": {"type": "float", "default": 9500.0, "min": 2000.0, "max": 18000.0, "step": 100.0},

        "formant_db": {"type": "float", "default": 4.5, "min": 0.0, "max": 12.0, "step": 0.5},
        "bite_db": {"type": "float", "default": 2.5, "min": -6.0, "max": 12.0, "step": 0.5},
        "nasal_db": {"type": "float", "default": -1.5, "min": -12.0, "max": 6.0, "step": 0.5},

        "attack": {"type": "float", "default": 0.15, "min": -1.0, "max": 1.0, "step": 0.05},
        "sustain": {"type": "float", "default": 0.25, "min": -1.0, "max": 1.0, "step": 0.05},

        "comp_thresh_db": {"type": "float", "default": -24.0, "min": -60.0, "max": 0.0, "step": 0.5},
        "comp_ratio": {"type": "float", "default": 4.0, "min": 1.0, "max": 20.0, "step": 0.25},
        "drive": {"type": "float", "default": 1.8, "min": 0.1, "max": 10.0, "step": 0.1},

        "mix": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)

        lowcut = float(params.get("lowcut_hz", 90.0))
        hicut = float(params.get("hi_cut_hz", 9500.0))
        formant_db = float(params.get("formant_db", 4.5))
        bite_db = float(params.get("bite_db", 2.5))
        nasal_db = float(params.get("nasal_db", -1.5))
        atk = float(params.get("attack", 0.15))
        sus = float(params.get("sustain", 0.25))
        thr = float(params.get("comp_thresh_db", -24.0))
        ratio = float(params.get("comp_ratio", 4.0))
        drive = float(params.get("drive", 1.8))
        mix = float(params.get("mix", 1.0))

        wet = x.copy()

        if lowcut > 0:
            wet[:, 0] = _onepole_hp(wet[:, 0], sr, lowcut)
            wet[:, 1] = _onepole_hp(wet[:, 1], sr, lowcut)

        # horn-ish formants + bite; reduce nasal 1k if needed
        wet = _apply_peaks(
            wet, sr,
            [
                (700.0, 1.0, formant_db * 0.70),
                (1200.0, 1.1, formant_db * 0.55),
                (2500.0, 1.2, formant_db * 0.45),
                (1000.0, 1.0, nasal_db),
                (3800.0, 1.2, bite_db),
            ],
        )

        # focus highs
        wet = _biquad_process_stereo(wet, _biquad_coeff_lowpass(sr, hicut, q=0.707))

        # a touch of transient and sustain
        wet = _transient_shaper(wet, sr, attack=atk, sustain=sus)

        # compress + saturate (brass wants density)
        wet = _compress_stereo(wet, sr, thr_db=thr, ratio=ratio, attack_ms=8.0, release_ms=140.0, makeup_db=3.0)
        wet = _softsat(wet, drive)

        out = _mix(x, wet, mix)
        out = np.tanh(out * 1.02).astype(np.float32)
        return AudioBuffer(out, sr), {}


# =============================================================================
# Dark Piano Finisher (felt / moody / cinematic)
# =============================================================================
class DarkPiano(BaseBlock):
    """
    Dark piano polish:
      - roll top end, add low-mid warmth, tame hammer click
      - gentle sustain shaping + tape-ish saturation
      - optional slight "room" tail via short multi-tap
    """
    KIND = "fx"
    PARAMS = {
        "lowcut_hz": {"type": "float", "default": 35.0, "min": 0.0, "max": 250.0, "step": 5.0},
        "warmth_db": {"type": "float", "default": 3.0, "min": -12.0, "max": 12.0, "step": 0.5},
        "dark_lp_hz": {"type": "float", "default": 9000.0, "min": 1500.0, "max": 20000.0, "step": 100.0},
        "declick_db": {"type": "float", "default": -2.5, "min": -12.0, "max": 6.0, "step": 0.5},

        "attack": {"type": "float", "default": -0.10, "min": -1.0, "max": 1.0, "step": 0.05},
        "sustain": {"type": "float", "default": 0.25, "min": -1.0, "max": 1.0, "step": 0.05},

        "drive": {"type": "float", "default": 1.7, "min": 0.1, "max": 10.0, "step": 0.1},
        "glue_thresh_db": {"type": "float", "default": -26.0, "min": -60.0, "max": 0.0, "step": 0.5},
        "glue_ratio": {"type": "float", "default": 2.2, "min": 1.0, "max": 10.0, "step": 0.25},

        "room_amount": {"type": "float", "default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01},
        "room_time_ms": {"type": "float", "default": 55.0, "min": 10.0, "max": 180.0, "step": 5.0},

        "mix": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    }

    def execute(self, payload: AudioBuffer, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        x = ensure_stereo(payload.data).astype(np.float32, copy=False)
        sr = int(payload.sr)
        n = int(x.shape[0])

        lowcut = float(params.get("lowcut_hz", 35.0))
        warmth_db = float(params.get("warmth_db", 3.0))
        lp_hz = float(params.get("dark_lp_hz", 9000.0))
        declick_db = float(params.get("declick_db", -2.5))
        atk = float(params.get("attack", -0.10))
        sus = float(params.get("sustain", 0.25))
        drive = float(params.get("drive", 1.7))
        thr = float(params.get("glue_thresh_db", -26.0))
        ratio = float(params.get("glue_ratio", 2.2))
        room_amt = float(params.get("room_amount", 0.20))
        room_ms = float(params.get("room_time_ms", 55.0))
        mix = float(params.get("mix", 1.0))

        wet = x.copy()

        if lowcut > 0:
            wet[:, 0] = _onepole_hp(wet[:, 0], sr, lowcut)
            wet[:, 1] = _onepole_hp(wet[:, 1], sr, lowcut)

        # warmth around 180-320, tame click around 3-6k
        wet = _apply_peaks(
            wet, sr,
            [
                (220.0, 0.9, warmth_db * 0.65),
                (320.0, 1.0, warmth_db * 0.35),
                (3500.0, 1.1, declick_db),
                (6000.0, 1.3, declick_db * 0.5),
            ],
        )

        # darker top end
        wet = _biquad_process_stereo(wet, _biquad_coeff_lowpass(sr, lp_hz, q=0.707))

        # transient: reduce attack, boost sustain a bit
        wet = _transient_shaper(wet, sr, attack=atk, sustain=sus)

        # glue compression + tape-ish sat
        wet = _compress_stereo(wet, sr, thr_db=thr, ratio=ratio, attack_ms=10.0, release_ms=180.0, makeup_db=3.0)
        wet = _softsat(wet, drive)

        # tiny "room" (short multi-tap) to feel recorded
        room_amt = _clamp(room_amt, 0.0, 1.0)
        if room_amt > 1e-6:
            d = int(round((room_ms / 1000.0) * sr))
            d = int(np.clip(d, 8, max(8, n - 1)))
            taps = [d, int(d * 1.33), int(d * 1.77)]
            taps = [int(np.clip(t, 8, max(8, n - 1))) for t in taps]
            room = np.zeros_like(wet, dtype=np.float32)
            for t in taps:
                room[t:] += wet[:-t] * (0.45 / (1.0 + 0.15 * taps.index(t)))
            wet = wet + room_amt * room

        out = _mix(x, wet, mix)
        out = np.tanh(out * 1.03).astype(np.float32)
        return AudioBuffer(out, sr), {}


# -----------------------------------------------------------------------------
# Register blocks
# -----------------------------------------------------------------------------
BLOCKS.register("guitar_string", GuitarString)
BLOCKS.register("piano_key", PianoKey)
BLOCKS.register("horn", Horn)
BLOCKS.register("dark_piano", DarkPiano)
