from __future__ import annotations

import ctypes
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


MP_WAVE_SINE = 0
MP_WAVE_SQUARE = 1
MP_WAVE_SAW = 2
MP_WAVE_TRIANGLE = 3

C_FLOAT_P = ctypes.POINTER(ctypes.c_float)
C_INT_P = ctypes.POINTER(ctypes.c_int)


def ensure_stereo(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 0:
        arr = np.zeros((0, 2), dtype=np.float32)
    elif arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    elif arr.ndim == 2:
        if arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
    else:
        raise ValueError(f"Audio arrays must be mono or stereo, got shape={arr.shape}")

    return np.ascontiguousarray(arr.astype(np.float32, copy=False))


def waveform_id(value: Any) -> int:
    if isinstance(value, str):
        v = value.strip().lower().replace("_", "").replace("-", "").replace(" ", "")

        if v in {"sine", "sin"}:
            return MP_WAVE_SINE
        if v in {"square", "sq", "sqr", "pulse", "pwm"}:
            return MP_WAVE_SQUARE
        if v in {"saw", "sawtooth", "tooth"}:
            return MP_WAVE_SAW
        if v in {"tri", "triangle"}:
            return MP_WAVE_TRIANGLE

    try:
        return int(value)
    except Exception:
        return MP_WAVE_SINE


def _wave_sample(waveform: int, phase: np.ndarray) -> np.ndarray:
    phase = phase - np.floor(phase)

    if waveform == MP_WAVE_SQUARE:
        return np.where(phase < 0.5, 1.0, -1.0).astype(np.float32)

    if waveform == MP_WAVE_SAW:
        return ((2.0 * phase) - 1.0).astype(np.float32)

    if waveform == MP_WAVE_TRIANGLE:
        return (2.0 * np.abs(2.0 * phase - 1.0) - 1.0).astype(np.float32)

    return np.sin(2.0 * np.pi * phase).astype(np.float32)


def _python_render_synth_notes(
    midi_notes: np.ndarray,
    start_frames: np.ndarray,
    length_frames: np.ndarray,
    velocities: np.ndarray,
    *,
    total_frames: int,
    sample_rate: int,
    waveform: int,
    master_gain: float,
    attack_seconds: float,
    release_seconds: float,
    pan: float,
) -> np.ndarray:
    total_frames = max(0, int(total_frames))
    sample_rate = int(sample_rate)

    out = np.zeros((total_frames, 2), dtype=np.float32)

    if total_frames <= 0:
        return out

    pan = float(np.clip(pan, -1.0, 1.0))
    left_gain = 1.0 if pan <= 0.0 else 1.0 - pan
    right_gain = 1.0 if pan >= 0.0 else 1.0 + pan

    attack_frames = max(0, int(round(float(attack_seconds) * sample_rate)))
    release_frames = max(0, int(round(float(release_seconds) * sample_rate)))

    for midi, start, length, velocity in zip(midi_notes, start_frames, length_frames, velocities):
        midi = int(midi)

        if midi < 0 or midi > 127:
            continue

        note_start = int(start)
        note_len = max(1, int(length))
        note_end = note_start + note_len

        if note_end <= 0 or note_start >= total_frames:
            continue

        dst_a = max(0, note_start)
        dst_b = min(total_frames, note_end)

        if dst_b <= dst_a:
            continue

        local = np.arange(dst_a - note_start, dst_b - note_start, dtype=np.float64)

        freq = 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))
        phase = local * (freq / float(sample_rate))

        env = np.ones_like(local, dtype=np.float32)

        if attack_frames > 0:
            env *= np.minimum(1.0, local.astype(np.float32) / float(attack_frames))

        if release_frames > 0:
            remaining = np.maximum(0.0, float(note_len) - local.astype(np.float32))
            env *= np.minimum(1.0, remaining / float(release_frames))

        sample = (
            _wave_sample(waveform, phase)
            * float(np.clip(velocity, 0.0, 2.0))
            * float(master_gain)
            * env
        )

        out[dst_a:dst_b, 0] += sample * left_gain
        out[dst_a:dst_b, 1] += sample * right_gain


    return out.astype(np.float32, copy=False)



def _adsr_held_value_py(
    i: int,
    frames: int,
    hold_frames: int,
    sample_rate: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    curve: float = 1.0,
) -> float:
    if frames <= 0:
        return 0.0

    sample_rate = int(max(1, sample_rate))
    hold_frames = int(max(1, min(int(frames), int(hold_frames))))
    attack = max(0.0, float(attack))
    decay = max(0.0, float(decay))
    sustain = float(np.clip(float(sustain), 0.0, 2.0))
    release = max(0.0, float(release))
    curve = float(np.clip(float(curve), 0.15, 2.5))

    attack_n = max(0, int(round(attack * sample_rate)))
    decay_n = max(0, int(round(decay * sample_rate)))
    release_n = max(0, int(round(release * sample_rate)))

    if attack_n + decay_n > hold_frames:
        total = max(1, attack_n + decay_n)
        scale = float(hold_frames) / float(total)
        attack_n = int(round(attack_n * scale))
        decay_n = max(0, hold_frames - attack_n)

    def hold_env_at(j: int) -> float:
        j = int(max(0, min(hold_frames - 1, int(j))))
        if attack_n == 0 and decay_n == 0:
            return float(max(sustain, 1.0)) if j == 0 else float(sustain)
        if attack_n > 0 and j < attack_n:
            x = float(np.clip(j / float(max(1, attack_n)), 0.0, 1.0))
            return float(np.clip(x ** (1.0 / curve), 0.0, 2.0))
        if decay_n > 0 and j < attack_n + decay_n:
            x = float(np.clip((j - attack_n) / float(max(1, decay_n)), 0.0, 1.0))
            shaped = x ** curve
            return float(np.clip(1.0 + (sustain - 1.0) * shaped, 0.0, 2.0))
        return float(sustain)

    if i < 0 or i >= frames:
        return 0.0
    if i < hold_frames:
        return hold_env_at(i)
    if release_n > 0 and i < hold_frames + release_n:
        x = float(np.clip((i - hold_frames) / float(max(1, release_n)), 0.0, 1.0))
        return float(np.clip(hold_env_at(hold_frames - 1) * ((1.0 - x) ** (1.0 / curve)), 0.0, 2.0))
    return 0.0


def _python_render_synth_notes_adsr(
    midi_notes: np.ndarray,
    start_frames: np.ndarray,
    length_frames: np.ndarray,
    velocities: np.ndarray,
    attacks: np.ndarray,
    decays: np.ndarray,
    sustains: np.ndarray,
    releases: np.ndarray,
    *,
    total_frames: int,
    sample_rate: int,
    waveform: int,
    master_gain: float,
    pan: float,
) -> np.ndarray:
    total_frames = max(0, int(total_frames))
    sample_rate = int(max(1, sample_rate))
    out = np.zeros((total_frames, 2), dtype=np.float32)

    if total_frames <= 0:
        return out

    pan = float(np.clip(pan, -1.0, 1.0))
    left_gain = 1.0 if pan <= 0.0 else 1.0 - pan
    right_gain = 1.0 if pan >= 0.0 else 1.0 + pan

    count = min(
        len(midi_notes), len(start_frames), len(length_frames), len(velocities),
        len(attacks), len(decays), len(sustains), len(releases),
    )

    for note_i in range(count):
        midi = int(midi_notes[note_i])
        if midi < 0 or midi > 127:
            continue

        note_start = int(start_frames[note_i])
        hold_frames = max(1, int(length_frames[note_i]))
        release_frames = max(0, int(round(max(0.0, float(releases[note_i])) * sample_rate)))
        render_frames = max(1, hold_frames + release_frames)
        note_end = note_start + render_frames

        if note_end <= 0 or note_start >= total_frames:
            continue

        dst_a = max(0, note_start)
        dst_b = min(total_frames, note_end)
        if dst_b <= dst_a:
            continue

        local = np.arange(dst_a - note_start, dst_b - note_start, dtype=np.float64)
        freq = 440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0))
        phase = local * (freq / float(sample_rate))
        env = np.asarray([
            _adsr_held_value_py(
                int(i), render_frames, hold_frames, sample_rate,
                float(attacks[note_i]), float(decays[note_i]), float(sustains[note_i]), float(releases[note_i]), 1.0,
            )
            for i in local
        ], dtype=np.float32)
        sample = (
            _wave_sample(waveform, phase)
            * float(np.clip(velocities[note_i], 0.0, 2.0))
            * float(master_gain)
            * env
        )
        out[dst_a:dst_b, 0] += sample * left_gain
        out[dst_a:dst_b, 1] += sample * right_gain

    return out.astype(np.float32, copy=False)


def _python_dc_block(audio: np.ndarray, *, amount: float = 0.995) -> np.ndarray:
    x = ensure_stereo(audio).copy()
    if x.shape[0] <= 1:
        return x

    amount = float(np.clip(float(amount), 0.90, 0.9999))
    out = np.empty_like(x, dtype=np.float32)
    x1_l = x1_r = 0.0
    y1_l = y1_r = 0.0

    for i in range(x.shape[0]):
        xl = float(x[i, 0])
        xr = float(x[i, 1])
        yl = xl - x1_l + amount * y1_l
        yr = xr - x1_r + amount * y1_r
        out[i, 0] = yl
        out[i, 1] = yr
        x1_l, x1_r = xl, xr
        y1_l, y1_r = yl, yr

    return out.astype(np.float32, copy=False)


def _python_master_finish(
    audio: np.ndarray,
    *,
    sample_rate: int,
    master_gain: float = 1.0,
    master_drive: float = 1.04,
    master_ceiling: float = 0.98,
    master_lowpass_hz: float = 20500.0,
    lowpass_wet: float = 0.35,
    dc_amount: float = 0.995,
) -> np.ndarray:
    x = _python_dc_block(audio, amount=dc_amount)
    if x.size == 0:
        return x

    x = (x * float(master_gain)).astype(np.float32, copy=False)

    if float(master_lowpass_hz) > 0.0 and float(lowpass_wet) > 1.0e-8:
        x = _python_lowpass(
            x,
            sample_rate=int(sample_rate),
            cutoff_hz=float(master_lowpass_hz),
            wet=float(lowpass_wet),
        )

    drive = float(np.clip(master_drive, 0.05, 20.0))
    ceiling = float(np.clip(master_ceiling, 0.05, 1.0))
    y = np.tanh(x * (0.30 + drive ** 1.10)).astype(np.float32, copy=False)
    mx = float(np.max(np.abs(y))) if y.size else 0.0
    if mx > ceiling and mx > 1.0e-12:
        y *= ceiling / mx
    return np.clip(y, -ceiling, ceiling).astype(np.float32, copy=False)



def _python_apply_note_adsr(
    audio: np.ndarray,
    *,
    sample_rate: int,
    hold_frames: int,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    curve: float = 1.0,
) -> np.ndarray:
    x = ensure_stereo(audio).copy()

    if x.size == 0:
        return x

    frames = int(x.shape[0])
    sample_rate = int(max(1, sample_rate))
    hold_frames = int(max(1, min(frames, int(hold_frames))))

    attack = max(0.0, float(attack))
    decay = max(0.0, float(decay))
    sustain = float(np.clip(float(sustain), 0.0, 2.0))
    release = max(0.0, float(release))
    curve = float(np.clip(float(curve), 0.15, 2.5))

    attack_n = max(0, int(round(attack * sample_rate)))
    decay_n = max(0, int(round(decay * sample_rate)))
    release_n = max(0, int(round(release * sample_rate)))

    if attack_n + decay_n > hold_frames:
        total = max(1, attack_n + decay_n)
        scale = float(hold_frames) / float(total)
        attack_n = int(round(attack_n * scale))
        decay_n = max(0, hold_frames - attack_n)

    def hold_env_at(j: int) -> float:
        j = int(max(0, min(hold_frames - 1, j)))

        if attack_n == 0 and decay_n == 0:
            if j == 0:
                return float(max(sustain, 1.0))
            return float(sustain)

        if attack_n > 0 and j < attack_n:
            v = float(np.clip(j / float(max(1, attack_n)), 0.0, 1.0))
            return float(np.clip(v ** (1.0 / curve), 0.0, 2.0))

        if decay_n > 0 and j < attack_n + decay_n:
            v = float(np.clip((j - attack_n) / float(max(1, decay_n)), 0.0, 1.0))
            shaped = v ** curve
            return float(np.clip(1.0 + (sustain - 1.0) * shaped, 0.0, 2.0))

        return float(sustain)

    env = np.zeros((frames,), dtype=np.float32)

    for i in range(hold_frames):
        env[i] = hold_env_at(i)

    if release_n > 0 and hold_frames < frames:
        rel_end = min(frames, hold_frames + release_n)
        hold_level = hold_env_at(hold_frames - 1)
        for i in range(hold_frames, rel_end):
            v = float(np.clip((i - hold_frames) / float(max(1, release_n)), 0.0, 1.0))
            env[i] = float(np.clip(hold_level * ((1.0 - v) ** (1.0 / curve)), 0.0, 2.0))

    x *= env[:, None]
    return x.astype(np.float32, copy=False)

def _python_soft_clip_normalize(
    audio: np.ndarray,
    *,
    ceiling: float,
    peak: float,
    only_if_over: bool,
) -> np.ndarray:
    x = ensure_stereo(audio).copy()

    if x.size == 0:
        return x

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    ceiling = float(np.clip(ceiling, 0.05, 1.0))
    peak = float(np.clip(peak, 0.05, 1.0))

    x = (np.tanh(x / ceiling) * ceiling).astype(np.float32, copy=False)

    mx = float(np.max(np.abs(x))) if x.size else 0.0

    if mx > 1.0e-9 and ((not only_if_over) or mx > peak):
        x = (x * (peak / mx)).astype(np.float32, copy=False)

    return x


def _python_lowpass(
    audio: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float,
    wet: float,
) -> np.ndarray:
    x = ensure_stereo(audio).copy()

    if x.size == 0:
        return x

    sample_rate = int(sample_rate)
    cutoff_hz = float(np.clip(cutoff_hz, 10.0, float(sample_rate) * 0.45))
    wet = float(np.clip(wet, 0.0, 1.0))

    dt = 1.0 / float(sample_rate)
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    yl = float(x[0, 0])
    yr = float(x[0, 1])

    for i in range(x.shape[0]):
        xl = float(x[i, 0])
        xr = float(x[i, 1])

        yl = yl + alpha * (xl - yl)
        yr = yr + alpha * (xr - yr)

        x[i, 0] = xl * (1.0 - wet) + yl * wet
        x[i, 1] = xr * (1.0 - wet) + yr * wet

    return x.astype(np.float32, copy=False)


def _python_delay(
    audio: np.ndarray,
    *,
    sample_rate: int,
    delay_ms: float,
    feedback: float,
    wet: float,
) -> np.ndarray:
    x = ensure_stereo(audio).copy()

    if x.size == 0:
        return x

    delay_ms = float(np.clip(delay_ms, 1.0, 5000.0))
    feedback = float(np.clip(feedback, 0.0, 0.98))
    wet = float(np.clip(wet, 0.0, 2.0))

    delay_frames = max(1, int(round((delay_ms / 1000.0) * int(sample_rate))))
    ring = np.zeros((delay_frames, 2), dtype=np.float32)
    write_pos = 0

    for i in range(x.shape[0]):
        delayed = ring[write_pos].copy()
        incoming = x[i].copy()

        x[i] = incoming + delayed * wet
        ring[write_pos] = incoming + delayed * feedback

        write_pos += 1

        if write_pos >= delay_frames:
            write_pos = 0

    return x.astype(np.float32, copy=False)


class MelodyProjectNative:
    def __init__(self, dll_path: Optional[str | os.PathLike[str]] = None) -> None:
        self.dll_path: Optional[str] = None
        self.load_error: Optional[str] = None
        self.available: bool = False
        self.native_sounds_available: bool = False
        self.note_envelope_available: bool = False
        self.synth_adsr_available: bool = False
        self.dc_block_available: bool = False
        self.master_finish_available: bool = False
        self.mix_rendered_layer_available: bool = False
        self.version: Optional[int] = None
        self.lib: Optional[ctypes.CDLL] = None
        self._lock = threading.RLock()

        self._load(dll_path)

    def _candidate_paths(self, dll_path: Optional[str | os.PathLike[str]]) -> List[Path]:
        candidates: List[Path] = []

        if dll_path:
            candidates.append(Path(dll_path))

        env_path = os.getenv("MELODYPROJECT_DLL", "").strip()
        if env_path:
            candidates.append(Path(env_path))

        here = Path(__file__).resolve().parent
        cwd = Path.cwd()

        candidates.append(here / "MelodyProject.dll")
        candidates.append(cwd / "MelodyProject.dll")
        candidates.append(Path("MelodyProject.dll"))

        out: List[Path] = []
        seen: set[str] = set()

        for p in candidates:
            key = str(p)
            if key not in seen:
                seen.add(key)
                out.append(p)

        return out

    def _bind_required(self, lib: ctypes.CDLL, name: str, argtypes: list[Any], restype: Any) -> Any:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    def _bind_optional(self, lib: ctypes.CDLL, name: str, argtypes: list[Any], restype: Any) -> bool:
        try:
            fn = getattr(lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
            return True
        except Exception:
            return False

    def _load(self, dll_path: Optional[str | os.PathLike[str]]) -> None:
        last_error: Optional[str] = None

        for path in self._candidate_paths(dll_path):
            try:
                lib = ctypes.CDLL(str(path))

                self._bind_required(lib, "mp_version", [], ctypes.c_int)
                self._bind_required(lib, "mp_last_error", [], ctypes.c_char_p)

                self._bind_required(lib, "mp_clear_stereo_f32", [C_FLOAT_P, ctypes.c_int], ctypes.c_int)
                self._bind_required(lib, "mp_copy_stereo_f32", [C_FLOAT_P, ctypes.c_int, C_FLOAT_P], ctypes.c_int)
                self._bind_required(lib, "mp_gain_stereo_f32", [C_FLOAT_P, ctypes.c_int, ctypes.c_float, C_FLOAT_P], ctypes.c_int)
                self._bind_required(lib, "mp_mix_into_stereo_f32", [C_FLOAT_P, C_FLOAT_P, ctypes.c_int, ctypes.c_float], ctypes.c_int)
                self._bind_required(
                    lib,
                    "mp_mix_two_stereo_f32",
                    [C_FLOAT_P, C_FLOAT_P, ctypes.c_int, ctypes.c_float, ctypes.c_float, C_FLOAT_P],
                    ctypes.c_int,
                )
                self._bind_required(
                    lib,
                    "mp_soft_clip_normalize_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int],
                    ctypes.c_int,
                )
                self._bind_required(
                    lib,
                    "mp_render_synth_notes_stereo_f32",
                    [
                        C_INT_P,
                        C_INT_P,
                        C_INT_P,
                        C_FLOAT_P,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        C_FLOAT_P,
                    ],
                    ctypes.c_int,
                )
                self._bind_required(
                    lib,
                    "mp_convolve_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, C_FLOAT_P, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, C_FLOAT_P],
                    ctypes.c_int,
                )
                self._bind_required(
                    lib,
                    "mp_one_pole_lowpass_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )
                self._bind_required(
                    lib,
                    "mp_delay_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )
                self._bind_required(lib, "mp_stereo_width_f32", [C_FLOAT_P, ctypes.c_int, ctypes.c_float], ctypes.c_int)

                optional_ok = True

                optional_ok &= self._bind_optional(
                    lib,
                    "mp_apply_adsr_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )

                self.note_envelope_available = self._bind_optional(
                    lib,
                    "mp_apply_note_adsr_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )

                self.synth_adsr_available = self._bind_optional(
                    lib,
                    "mp_render_synth_notes_adsr_stereo_f32",
                    [
                        C_INT_P,
                        C_INT_P,
                        C_INT_P,
                        C_FLOAT_P,
                        C_FLOAT_P,
                        C_FLOAT_P,
                        C_FLOAT_P,
                        C_FLOAT_P,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        C_FLOAT_P,
                    ],
                    ctypes.c_int,
                )

                self.dc_block_available = self._bind_optional(
                    lib,
                    "mp_dc_block_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_float],
                    ctypes.c_int,
                )

                self.master_finish_available = self._bind_optional(
                    lib,
                    "mp_master_finish_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )

                self.mix_rendered_layer_available = self._bind_optional(
                    lib,
                    "mp_mix_rendered_note_layer_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float],
                    ctypes.c_int,
                )

                optional_ok &= self._bind_optional(
                    lib,
                    "mp_waveshape_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_svf_lowpass_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_microshift_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, C_FLOAT_P],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_chorus_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, C_FLOAT_P],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_schroeder_reverb_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, C_FLOAT_P],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_biquad_filter_stereo_f32",
                    [C_FLOAT_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_render_sound_synth_keys_f32",
                    [
                        ctypes.c_float,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        C_FLOAT_P,
                    ],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_render_sound_guitar_pluck_f32",
                    [
                        ctypes.c_float,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        C_FLOAT_P,
                    ],
                    ctypes.c_int,
                )
                optional_ok &= self._bind_optional(
                    lib,
                    "mp_render_sound_bell_fm_f32",
                    [
                        ctypes.c_float,
                        ctypes.c_int,
                        ctypes.c_int,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_float,
                        ctypes.c_int,
                        C_FLOAT_P,
                    ],
                    ctypes.c_int,
                )

                self.lib = lib
                self.dll_path = str(path)
                self.version = int(lib.mp_version())
                self.available = True
                self.native_sounds_available = bool(optional_ok)
                self.load_error = None
                return

            except Exception as exc:
                last_error = f"{path}: {exc}"

        self.lib = None
        self.dll_path = None
        self.version = None
        self.available = False
        self.native_sounds_available = False
        self.note_envelope_available = False
        self.synth_adsr_available = False
        self.dc_block_available = False
        self.master_finish_available = False
        self.mix_rendered_layer_available = False
        self.load_error = last_error or "MelodyProject.dll not found"

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "available": bool(self.available),
            "native_sounds_available": bool(self.native_sounds_available),
            "note_envelope_available": bool(self.note_envelope_available),
            "synth_adsr_available": bool(self.synth_adsr_available),
            "dc_block_available": bool(self.dc_block_available),
            "master_finish_available": bool(self.master_finish_available),
            "mix_rendered_layer_available": bool(self.mix_rendered_layer_available),
            "dll_path": self.dll_path,
            "load_error": self.load_error,
            "version": self.version,
        }

    def _float_ptr(self, arr: np.ndarray) -> C_FLOAT_P:
        return arr.ctypes.data_as(C_FLOAT_P)

    def _int_ptr(self, arr: np.ndarray) -> C_INT_P:
        return arr.ctypes.data_as(C_INT_P)

    def _last_error(self) -> str:
        if not self.lib:
            return self.load_error or "MelodyProject native DSP unavailable"

        try:
            raw = self.lib.mp_last_error()
            if raw:
                return raw.decode("utf-8", errors="replace")
        except Exception:
            pass

        return "MelodyProject native DSP call failed"

    def _check(self, rc: int) -> None:
        if int(rc) != 0:
            raise RuntimeError(self._last_error())

    def _has(self, name: str) -> bool:
        return bool(self.available and self.lib is not None and hasattr(self.lib, name))

    def clear(self, frames: int) -> np.ndarray:
        out = np.zeros((max(0, int(frames)), 2), dtype=np.float32)

        if self.available and self.lib and out.size:
            self._check(self.lib.mp_clear_stereo_f32(self._float_ptr(out), ctypes.c_int(int(out.shape[0]))))

        return out

    def copy(self, src: np.ndarray) -> np.ndarray:
        x = ensure_stereo(src)
        out = np.empty_like(x)

        if not self.available or not self.lib or x.size == 0:
            return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)

        self._check(self.lib.mp_copy_stereo_f32(self._float_ptr(x), ctypes.c_int(int(x.shape[0])), self._float_ptr(out)))
        return out

    def gain(self, src: np.ndarray, gain: float) -> np.ndarray:
        x = ensure_stereo(src)
        out = np.empty_like(x)

        if x.size == 0:
            return out

        if not self.available or not self.lib:
            return (x * float(gain)).astype(np.float32, copy=False)

        self._check(
            self.lib.mp_gain_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_float(float(gain)),
                self._float_ptr(out),
            )
        )

        return out

    def mix_into(self, dst: np.ndarray, src: np.ndarray, gain: float = 1.0) -> np.ndarray:
        d = ensure_stereo(dst)
        s = ensure_stereo(src)

        n = min(int(d.shape[0]), int(s.shape[0]))

        if n <= 0:
            return d

        if not self.available or not self.lib:
            d[:n] += s[:n] * float(gain)
            return d

        self._check(
            self.lib.mp_mix_into_stereo_f32(
                self._float_ptr(d),
                self._float_ptr(s),
                ctypes.c_int(n),
                ctypes.c_float(float(gain)),
            )
        )

        return d

    def mix_two(self, a: np.ndarray, b: np.ndarray, *, gain_a: float = 1.0, gain_b: float = 1.0) -> np.ndarray:
        aa = ensure_stereo(a)
        bb = ensure_stereo(b)

        n = min(int(aa.shape[0]), int(bb.shape[0]))
        out = np.zeros((n, 2), dtype=np.float32)

        if n <= 0:
            return out

        if not self.available or not self.lib:
            out[:] = aa[:n] * float(gain_a) + bb[:n] * float(gain_b)
            return out

        self._check(
            self.lib.mp_mix_two_stereo_f32(
                self._float_ptr(aa),
                self._float_ptr(bb),
                ctypes.c_int(n),
                ctypes.c_float(float(gain_a)),
                ctypes.c_float(float(gain_b)),
                self._float_ptr(out),
            )
        )

        return out

    def soft_clip_normalize(
        self,
        audio: np.ndarray,
        *,
        ceiling: float = 0.995,
        peak: float = 0.98,
        only_if_over: bool = True,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self.available or not self.lib:
            return _python_soft_clip_normalize(
                x,
                ceiling=float(ceiling),
                peak=float(peak),
                only_if_over=bool(only_if_over),
            )

        self._check(
            self.lib.mp_soft_clip_normalize_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_float(float(ceiling)),
                ctypes.c_float(float(peak)),
                ctypes.c_int(1 if only_if_over else 0),
            )
        )

        return x

    def sanitize(self, audio: np.ndarray, *, ceiling: float = 0.995) -> np.ndarray:
        return self.soft_clip_normalize(audio, ceiling=float(ceiling), peak=1.0, only_if_over=True)

    def render_synth_notes(
        self,
        midi_notes: np.ndarray,
        start_frames: np.ndarray,
        length_frames: np.ndarray,
        velocities: np.ndarray,
        *,
        total_frames: int,
        sample_rate: int,
        waveform: int | str = MP_WAVE_SINE,
        master_gain: float = 0.22,
        attack_seconds: float = 0.004,
        release_seconds: float = 0.035,
        pan: float = 0.0,
        clear_output: bool = True,
    ) -> np.ndarray:
        total_frames = max(0, int(total_frames))
        out = np.zeros((total_frames, 2), dtype=np.float32)

        if total_frames <= 0:
            return out

        midi = np.ascontiguousarray(midi_notes, dtype=np.int32)
        starts = np.ascontiguousarray(start_frames, dtype=np.int32)
        lengths = np.ascontiguousarray(length_frames, dtype=np.int32)
        vels = np.ascontiguousarray(velocities, dtype=np.float32)

        note_count = int(min(midi.shape[0], starts.shape[0], lengths.shape[0], vels.shape[0]))

        if note_count <= 0:
            return out

        midi = midi[:note_count]
        starts = starts[:note_count]
        lengths = lengths[:note_count]
        vels = vels[:note_count]

        wf = waveform_id(waveform)

        if not self.available or not self.lib:
            return _python_render_synth_notes(
                midi,
                starts,
                lengths,
                vels,
                total_frames=total_frames,
                sample_rate=int(sample_rate),
                waveform=wf,
                master_gain=float(master_gain),
                attack_seconds=float(attack_seconds),
                release_seconds=float(release_seconds),
                pan=float(pan),
            )

        self._check(
            self.lib.mp_render_synth_notes_stereo_f32(
                self._int_ptr(midi),
                self._int_ptr(starts),
                self._int_ptr(lengths),
                self._float_ptr(vels),
                ctypes.c_int(note_count),
                ctypes.c_int(total_frames),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_int(wf),
                ctypes.c_float(float(master_gain)),
                ctypes.c_float(float(attack_seconds)),
                ctypes.c_float(float(release_seconds)),
                ctypes.c_float(float(pan)),
                ctypes.c_int(1 if clear_output else 0),
                self._float_ptr(out),
            )
        )

        return out

    def render_synth_notes_adsr(
        self,
        midi_notes: np.ndarray,
        start_frames: np.ndarray,
        length_frames: np.ndarray,
        velocities: np.ndarray,
        attacks: np.ndarray,
        decays: np.ndarray,
        sustains: np.ndarray,
        releases: np.ndarray,
        *,
        total_frames: int,
        sample_rate: int,
        waveform: int | str = MP_WAVE_SINE,
        master_gain: float = 0.22,
        pan: float = 0.0,
        clear_output: bool = True,
    ) -> np.ndarray:
        total_frames = max(0, int(total_frames))
        out = np.zeros((total_frames, 2), dtype=np.float32)

        if total_frames <= 0:
            return out

        midi = np.ascontiguousarray(midi_notes, dtype=np.int32)
        starts = np.ascontiguousarray(start_frames, dtype=np.int32)
        lengths = np.ascontiguousarray(length_frames, dtype=np.int32)
        vels = np.ascontiguousarray(velocities, dtype=np.float32)
        atk = np.ascontiguousarray(attacks, dtype=np.float32)
        dec = np.ascontiguousarray(decays, dtype=np.float32)
        sus = np.ascontiguousarray(sustains, dtype=np.float32)
        rel = np.ascontiguousarray(releases, dtype=np.float32)

        note_count = int(min(
            midi.shape[0], starts.shape[0], lengths.shape[0], vels.shape[0],
            atk.shape[0], dec.shape[0], sus.shape[0], rel.shape[0],
        ))

        if note_count <= 0:
            return out

        midi = midi[:note_count]
        starts = starts[:note_count]
        lengths = lengths[:note_count]
        vels = vels[:note_count]
        atk = atk[:note_count]
        dec = dec[:note_count]
        sus = sus[:note_count]
        rel = rel[:note_count]

        wf = waveform_id(waveform)

        if not self._has("mp_render_synth_notes_adsr_stereo_f32"):
            return _python_render_synth_notes_adsr(
                midi,
                starts,
                lengths,
                vels,
                atk,
                dec,
                sus,
                rel,
                total_frames=total_frames,
                sample_rate=int(sample_rate),
                waveform=wf,
                master_gain=float(master_gain),
                pan=float(pan),
            )

        self._check(
            self.lib.mp_render_synth_notes_adsr_stereo_f32(
                self._int_ptr(midi),
                self._int_ptr(starts),
                self._int_ptr(lengths),
                self._float_ptr(vels),
                self._float_ptr(atk),
                self._float_ptr(dec),
                self._float_ptr(sus),
                self._float_ptr(rel),
                ctypes.c_int(note_count),
                ctypes.c_int(total_frames),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_int(wf),
                ctypes.c_float(float(master_gain)),
                ctypes.c_float(float(pan)),
                ctypes.c_int(1 if clear_output else 0),
                self._float_ptr(out),
            )
        )

        return out

    def dc_block(self, audio: np.ndarray, *, amount: float = 0.995) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_dc_block_stereo_f32"):
            return _python_dc_block(x, amount=float(amount))

        self._check(
            self.lib.mp_dc_block_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_float(float(amount)),
            )
        )

        return x

    def master_finish(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        master_gain: float = 1.0,
        master_drive: float = 1.04,
        master_ceiling: float = 0.98,
        master_lowpass_hz: float = 20500.0,
        lowpass_wet: float = 0.35,
        dc_amount: float = 0.995,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_master_finish_stereo_f32"):
            return _python_master_finish(
                x,
                sample_rate=int(sample_rate),
                master_gain=float(master_gain),
                master_drive=float(master_drive),
                master_ceiling=float(master_ceiling),
                master_lowpass_hz=float(master_lowpass_hz),
                lowpass_wet=float(lowpass_wet),
                dc_amount=float(dc_amount),
            )

        self._check(
            self.lib.mp_master_finish_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(master_gain)),
                ctypes.c_float(float(master_drive)),
                ctypes.c_float(float(master_ceiling)),
                ctypes.c_float(float(master_lowpass_hz)),
                ctypes.c_float(float(lowpass_wet)),
                ctypes.c_float(float(dc_amount)),
            )
        )

        return x

    def mix_rendered_note_layer(
        self,
        dst: np.ndarray,
        src: np.ndarray,
        *,
        dst_offset: int = 0,
        src_offset: int = 0,
        frames_to_mix: Optional[int] = None,
        gain: float = 1.0,
    ) -> np.ndarray:
        d = ensure_stereo(dst)
        s = ensure_stereo(src)

        if d.size == 0 or s.size == 0:
            return d

        if frames_to_mix is None:
            frames_to_mix = min(int(d.shape[0]) - int(dst_offset), int(s.shape[0]) - int(src_offset))

        frames_to_mix = int(frames_to_mix)

        if frames_to_mix <= 0:
            return d

        if not self._has("mp_mix_rendered_note_layer_stereo_f32"):
            da = int(dst_offset)
            sa = int(src_offset)
            n = int(frames_to_mix)
            if da < 0:
                sa -= da
                n += da
                da = 0
            if sa < 0:
                da -= sa
                n += sa
                sa = 0
            n = max(0, min(n, int(d.shape[0]) - da, int(s.shape[0]) - sa))
            if n > 0:
                d[da:da + n] += s[sa:sa + n] * float(gain)
            return d

        self._check(
            self.lib.mp_mix_rendered_note_layer_stereo_f32(
                self._float_ptr(d),
                ctypes.c_int(int(d.shape[0])),
                self._float_ptr(s),
                ctypes.c_int(int(s.shape[0])),
                ctypes.c_int(int(dst_offset)),
                ctypes.c_int(int(src_offset)),
                ctypes.c_int(int(frames_to_mix)),
                ctypes.c_float(float(gain)),
            )
        )

        return d

    def convolve(
        self,
        audio: np.ndarray,
        ir: np.ndarray,
        *,
        dry: float = 0.7,
        wet: float = 0.3,
        normalize_ir: bool = True,
    ) -> np.ndarray:
        x = ensure_stereo(audio)
        h = ensure_stereo(ir)

        out = np.zeros_like(x)

        if x.size == 0:
            return out

        if h.size == 0:
            return self.gain(x, float(dry))

        if not self.available or not self.lib:
            out_l = np.convolve(x[:, 0], h[:, 0], mode="full")[: x.shape[0]]
            out_r = np.convolve(x[:, 1], h[:, 1], mode="full")[: x.shape[0]]
            wet_signal = np.stack([out_l, out_r], axis=1).astype(np.float32)
            return (x * float(dry) + wet_signal * float(wet)).astype(np.float32)

        self._check(
            self.lib.mp_convolve_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                self._float_ptr(h),
                ctypes.c_int(int(h.shape[0])),
                ctypes.c_float(float(dry)),
                ctypes.c_float(float(wet)),
                ctypes.c_int(1 if normalize_ir else 0),
                self._float_ptr(out),
            )
        )

        return out

    def lowpass(self, audio: np.ndarray, *, sample_rate: int, cutoff_hz: float, wet: float = 1.0) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self.available or not self.lib:
            return _python_lowpass(x, sample_rate=int(sample_rate), cutoff_hz=float(cutoff_hz), wet=float(wet))

        self._check(
            self.lib.mp_one_pole_lowpass_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(cutoff_hz)),
                ctypes.c_float(float(wet)),
            )
        )

        return x

    def delay(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        delay_ms: float,
        feedback: float,
        wet: float,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self.available or not self.lib:
            return _python_delay(
                x,
                sample_rate=int(sample_rate),
                delay_ms=float(delay_ms),
                feedback=float(feedback),
                wet=float(wet),
            )

        self._check(
            self.lib.mp_delay_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(delay_ms)),
                ctypes.c_float(float(feedback)),
                ctypes.c_float(float(wet)),
            )
        )

        return x

    def stereo_width(self, audio: np.ndarray, *, width: float = 1.0) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self.available or not self.lib:
            width = float(np.clip(width, 0.0, 3.0))
            l = x[:, 0]
            r = x[:, 1]
            mid = 0.5 * (l + r)
            side = 0.5 * (l - r) * width
            return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

        self._check(
            self.lib.mp_stereo_width_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_float(float(width)),
            )
        )

        return x

    def apply_adsr(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        curve: float = 0.55,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_apply_adsr_stereo_f32"):
            return x

        self._check(
            self.lib.mp_apply_adsr_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(attack)),
                ctypes.c_float(float(decay)),
                ctypes.c_float(float(sustain)),
                ctypes.c_float(float(release)),
                ctypes.c_float(float(curve)),
            )
        )

        return x


    def apply_note_adsr(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        hold_frames: Optional[int] = None,
        hold_seconds: Optional[float] = None,
        attack: float,
        decay: float,
        sustain: float,
        release: float,
        curve: float = 1.0,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if hold_frames is None:
            if hold_seconds is None:
                hold_frames = int(x.shape[0])
            else:
                hold_frames = int(round(max(0.0, float(hold_seconds)) * int(sample_rate)))

        hold_frames = int(max(1, min(int(x.shape[0]), int(hold_frames))))

        if not self._has("mp_apply_note_adsr_stereo_f32"):
            return _python_apply_note_adsr(
                x,
                sample_rate=int(sample_rate),
                hold_frames=int(hold_frames),
                attack=float(attack),
                decay=float(decay),
                sustain=float(sustain),
                release=float(release),
                curve=float(curve),
            )

        self._check(
            self.lib.mp_apply_note_adsr_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_int(int(hold_frames)),
                ctypes.c_float(float(attack)),
                ctypes.c_float(float(decay)),
                ctypes.c_float(float(sustain)),
                ctypes.c_float(float(release)),
                ctypes.c_float(float(curve)),
            )
        )

        return x

    def waveshape(self, audio: np.ndarray, *, drive: float, fold: float, tilt: float) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_waveshape_stereo_f32"):
            return self.soft_clip_normalize(x, ceiling=0.98, peak=0.98, only_if_over=True)

        self._check(
            self.lib.mp_waveshape_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_float(float(drive)),
                ctypes.c_float(float(fold)),
                ctypes.c_float(float(tilt)),
            )
        )

        return x

    def svf_lowpass(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        cutoff_hz: float,
        resonance: float = 0.0,
        wet: float = 1.0,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_svf_lowpass_stereo_f32"):
            return self.lowpass(x, sample_rate=sample_rate, cutoff_hz=cutoff_hz, wet=wet)

        self._check(
            self.lib.mp_svf_lowpass_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(cutoff_hz)),
                ctypes.c_float(float(resonance)),
                ctypes.c_float(float(wet)),
            )
        )

        return x

    def microshift(self, audio: np.ndarray, *, sample_rate: int, amount_ms: float, mix: float, seed: int = 0) -> np.ndarray:
        x = ensure_stereo(audio)
        out = np.empty_like(x)

        if x.size == 0:
            return out

        if not self._has("mp_microshift_stereo_f32"):
            return x.copy()

        self._check(
            self.lib.mp_microshift_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(amount_ms)),
                ctypes.c_float(float(mix)),
                ctypes.c_int(int(seed)),
                self._float_ptr(out),
            )
        )

        return out

    def chorus(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        rate_hz: float,
        depth_ms: float,
        mix: float,
        seed: int = 0,
    ) -> np.ndarray:
        x = ensure_stereo(audio)
        out = np.empty_like(x)

        if x.size == 0:
            return out

        if not self._has("mp_chorus_stereo_f32"):
            return x.copy()

        self._check(
            self.lib.mp_chorus_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(rate_hz)),
                ctypes.c_float(float(depth_ms)),
                ctypes.c_float(float(mix)),
                ctypes.c_int(int(seed)),
                self._float_ptr(out),
            )
        )

        return out

    def reverb(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        mix: float,
        room: float,
        predelay_ms: float,
        damp_hz: float,
    ) -> np.ndarray:
        x = ensure_stereo(audio)
        out = np.empty_like(x)

        if x.size == 0:
            return out

        if not self._has("mp_schroeder_reverb_stereo_f32"):
            return x.copy()

        self._check(
            self.lib.mp_schroeder_reverb_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(mix)),
                ctypes.c_float(float(room)),
                ctypes.c_float(float(predelay_ms)),
                ctypes.c_float(float(damp_hz)),
                self._float_ptr(out),
            )
        )

        return out

    def biquad_filter(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        filter_type: int,
        freq_hz: float,
        q: float,
        gain_db: float = 0.0,
        drive: float = 0.0,
    ) -> np.ndarray:
        x = ensure_stereo(audio).copy()

        if x.size == 0:
            return x

        if not self._has("mp_biquad_filter_stereo_f32"):
            return x

        self._check(
            self.lib.mp_biquad_filter_stereo_f32(
                self._float_ptr(x),
                ctypes.c_int(int(x.shape[0])),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_int(int(filter_type)),
                ctypes.c_float(float(freq_hz)),
                ctypes.c_float(float(q)),
                ctypes.c_float(float(gain_db)),
                ctypes.c_float(float(drive)),
            )
        )

        return x

    def render_sound_synth_keys(self, *, freq: float, frames: int, sample_rate: int, velocity: float, **p: Any) -> np.ndarray:
        frames = max(1, int(frames))
        out = np.zeros((frames, 2), dtype=np.float32)

        if not self._has("mp_render_sound_synth_keys_f32"):
            midi = int(round(69.0 + 12.0 * math.log2(max(1.0e-9, float(freq)) / 440.0)))
            return self.render_synth_notes(
                np.asarray([midi], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([frames], dtype=np.int32),
                np.asarray([velocity], dtype=np.float32),
                total_frames=frames,
                sample_rate=sample_rate,
                waveform=waveform_id(p.get("wave", p.get("waveform", "saw"))),
                master_gain=float(p.get("amp", 0.35)),
                attack_seconds=float(p.get("attack", 0.012)),
                release_seconds=float(p.get("release", 0.45)),
                pan=float(p.get("pan", 0.0)),
            )

        self._check(
            self.lib.mp_render_sound_synth_keys_f32(
                ctypes.c_float(float(freq)),
                ctypes.c_int(frames),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_int(waveform_id(p.get("wave", p.get("waveform", "saw")))),
                ctypes.c_float(float(velocity)),
                ctypes.c_float(float(p.get("amp", 0.35))),
                ctypes.c_float(float(p.get("attack", 0.012))),
                ctypes.c_float(float(p.get("decay", 0.150))),
                ctypes.c_float(float(p.get("sustain", 0.78))),
                ctypes.c_float(float(p.get("release", 0.45))),
                ctypes.c_float(float(p.get("env_curve", 0.65))),
                ctypes.c_int(int(p.get("unison", 9))),
                ctypes.c_float(float(p.get("detune_cents", 18.0))),
                ctypes.c_float(float(p.get("spread", 0.90))),
                ctypes.c_float(float(p.get("pwm", 0.48))),
                ctypes.c_float(float(p.get("sync", 0.15))),
                ctypes.c_float(float(p.get("sync_ratio", 3.2))),
                ctypes.c_float(float(p.get("fm_ratio", 1.6))),
                ctypes.c_float(float(p.get("fm_index", 5.5))),
                ctypes.c_float(float(p.get("pm_amount", 0.08))),
                ctypes.c_float(float(p.get("pd_amount", 0.15))),
                ctypes.c_float(float(p.get("drive", 1.80))),
                ctypes.c_float(float(p.get("fold", 0.12))),
                ctypes.c_float(float(p.get("tilt", 0.15))),
                ctypes.c_float(float(p.get("sub", 0.28))),
                ctypes.c_float(float(p.get("noise", 0.10))),
                ctypes.c_float(float(p.get("sparkle", 0.15))),
                ctypes.c_float(float(p.get("drift", 0.0045))),
                ctypes.c_float(float(p.get("drift_rate", 0.25))),
                ctypes.c_float(float(p.get("cutoff_hz", 6500.0))),
                ctypes.c_float(float(p.get("res", p.get("resonance", 0.40)))),
                ctypes.c_float(float(p.get("keytrack", 0.45))),
                ctypes.c_float(float(p.get("fenv_amt", 0.45))),
                ctypes.c_float(float(p.get("fenv_attack", 0.003))),
                ctypes.c_float(float(p.get("fenv_decay", 0.180))),
                ctypes.c_float(float(p.get("width_mix", 0.35))),
                ctypes.c_float(float(p.get("width_ms", 8.0))),
                ctypes.c_float(float(p.get("chorus_mix", 0.35))),
                ctypes.c_float(float(p.get("chorus_rate", 0.20))),
                ctypes.c_float(float(p.get("chorus_depth_ms", 10.0))),
                ctypes.c_float(float(p.get("reverb_mix", 0.25))),
                ctypes.c_float(float(p.get("reverb_room", 0.70))),
                ctypes.c_float(float(p.get("reverb_predelay_ms", 18.0))),
                ctypes.c_float(float(p.get("reverb_damp_hz", 6500.0))),
                ctypes.c_float(float(p.get("pan", 0.0))),
                ctypes.c_int(int(p.get("seed", 0))),
                self._float_ptr(out),
            )
        )

        return out

    def render_sound_guitar_pluck(self, *, freq: float, frames: int, sample_rate: int, velocity: float, **p: Any) -> np.ndarray:
        frames = max(1, int(frames))
        out = np.zeros((frames, 2), dtype=np.float32)

        if not self._has("mp_render_sound_guitar_pluck_f32"):
            midi = int(round(69.0 + 12.0 * math.log2(max(1.0e-9, float(freq)) / 440.0)))
            return self.render_synth_notes(
                np.asarray([midi], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([frames], dtype=np.int32),
                np.asarray([velocity], dtype=np.float32),
                total_frames=frames,
                sample_rate=sample_rate,
                waveform=MP_WAVE_TRIANGLE,
                master_gain=float(p.get("amp", 0.42)),
                attack_seconds=0.001,
                release_seconds=float(p.get("decay", 2.2)),
                pan=float(p.get("pan", 0.0)),
            )

        self._check(
            self.lib.mp_render_sound_guitar_pluck_f32(
                ctypes.c_float(float(freq)),
                ctypes.c_int(frames),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(velocity)),
                ctypes.c_float(float(p.get("amp", 0.42))),
                ctypes.c_float(float(p.get("decay", 2.2))),
                ctypes.c_float(float(p.get("brightness", 0.58))),
                ctypes.c_float(float(p.get("pick", 0.45))),
                ctypes.c_float(float(p.get("body", 0.38))),
                ctypes.c_float(float(p.get("noise", 0.10))),
                ctypes.c_float(float(p.get("chorus_mix", 0.08))),
                ctypes.c_float(float(p.get("reverb_mix", 0.18))),
                ctypes.c_float(float(p.get("reverb_room", 0.55))),
                ctypes.c_float(float(p.get("pan", 0.0))),
                ctypes.c_int(int(p.get("seed", 0))),
                self._float_ptr(out),
            )
        )

        return out

    def render_sound_bell_fm(self, *, freq: float, frames: int, sample_rate: int, velocity: float, **p: Any) -> np.ndarray:
        frames = max(1, int(frames))
        out = np.zeros((frames, 2), dtype=np.float32)

        if not self._has("mp_render_sound_bell_fm_f32"):
            midi = int(round(69.0 + 12.0 * math.log2(max(1.0e-9, float(freq)) / 440.0)))
            return self.render_synth_notes(
                np.asarray([midi], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([frames], dtype=np.int32),
                np.asarray([velocity], dtype=np.float32),
                total_frames=frames,
                sample_rate=sample_rate,
                waveform=MP_WAVE_SINE,
                master_gain=float(p.get("amp", 0.35)),
                attack_seconds=0.001,
                release_seconds=float(p.get("decay", 3.5)),
                pan=float(p.get("pan", 0.0)),
            )

        self._check(
            self.lib.mp_render_sound_bell_fm_f32(
                ctypes.c_float(float(freq)),
                ctypes.c_int(frames),
                ctypes.c_int(int(sample_rate)),
                ctypes.c_float(float(velocity)),
                ctypes.c_float(float(p.get("amp", 0.35))),
                ctypes.c_float(float(p.get("brightness", 0.65))),
                ctypes.c_float(float(p.get("inharm", 0.75))),
                ctypes.c_float(float(p.get("decay", 3.5))),
                ctypes.c_float(float(p.get("fm_ratio", 3.0))),
                ctypes.c_float(float(p.get("fm_index", 8.0))),
                ctypes.c_float(float(p.get("strike", 0.28))),
                ctypes.c_float(float(p.get("shimmer", 0.25))),
                ctypes.c_float(float(p.get("body", 0.35))),
                ctypes.c_float(float(p.get("chorus_mix", 0.15))),
                ctypes.c_float(float(p.get("reverb_mix", 0.30))),
                ctypes.c_float(float(p.get("reverb_room", 0.70))),
                ctypes.c_float(float(p.get("reverb_predelay_ms", 22.0))),
                ctypes.c_float(float(p.get("reverb_damp_hz", 6000.0))),
                ctypes.c_float(float(p.get("pan", 0.0))),
                ctypes.c_int(int(p.get("seed", 0))),
                self._float_ptr(out),
            )
        )

        return out


NATIVE_DSP = MelodyProjectNative()


def native_status() -> Dict[str, Any]:
    return NATIVE_DSP.status()