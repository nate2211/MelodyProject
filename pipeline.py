from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import wave
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from melodyproject_native import NATIVE_DSP, ensure_stereo
except Exception:
    NATIVE_DSP = None  # type: ignore

    def ensure_stereo(x: Any) -> np.ndarray:
        a = np.asarray(x, dtype=np.float32)

        if a.ndim == 0:
            a = np.zeros((0, 2), dtype=np.float32)
        elif a.ndim == 1:
            a = a[:, None]

        if a.shape[1] == 1:
            a = np.repeat(a, 2, axis=1)
        elif a.shape[1] > 2:
            a = a[:, :2]

        return np.ascontiguousarray(a.astype(np.float32, copy=False))


# ============================================================================
# Audio buffer + IO helpers
# ============================================================================

@dataclass
class AudioBuffer:
    data: np.ndarray
    sr: int = 48000

    @property
    def frames(self) -> int:
        return int(self.data.shape[0]) if self.data is not None else 0

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(ensure_stereo(self.data).copy(), int(self.sr))

    @classmethod
    def silence(cls, frames: int, sr: int = 48000) -> "AudioBuffer":
        return cls(np.zeros((max(0, int(frames)), 2), dtype=np.float32), int(sr))


def sanitize_audio(x: Any, *, ceiling: float = 0.995) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    ceiling = float(max(0.05, ceiling))
    return np.clip(y, -ceiling, ceiling).astype(np.float32, copy=False)


def normalize_for_playback(buf: AudioBuffer, *, peak: float = 0.98, only_if_over: bool = True) -> AudioBuffer:
    x = sanitize_audio(buf.data, ceiling=1.25)
    peak = float(max(0.05, min(1.0, peak)))

    mx = float(np.max(np.abs(x))) if x.size else 0.0

    if mx > 1.0e-9 and ((not only_if_over) or mx > peak):
        x = x * (peak / mx)

    return AudioBuffer(sanitize_audio(x, ceiling=0.995), int(buf.sr))


def write_wav(path: str | os.PathLike[str], buf: AudioBuffer) -> None:
    b = normalize_for_playback(buf, peak=0.98, only_if_over=True)
    pcm = np.clip(b.data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(b.sr))
        wf.writeframes(pcm.tobytes())


def _pcm_to_float(raw: bytes, channels: int, sampwidth: int) -> np.ndarray:
    if sampwidth == 1:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) / 128.0

    elif sampwidth == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    elif sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        a32 = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )
        sign = a32 & 0x800000
        a32 = a32 - (sign << 1)
        a = a32.astype(np.float32) / 8388608.0

    elif sampwidth == 4:
        a = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0

    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if channels <= 0:
        raise ValueError("WAV has no channels")

    a = a.reshape(-1, channels)

    if channels == 1:
        a = np.repeat(a, 2, axis=1)
    elif channels > 2:
        a = a[:, :2]

    return ensure_stereo(a)


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    x = ensure_stereo(x)
    src_sr = int(src_sr)
    dst_sr = int(dst_sr)

    if src_sr == dst_sr or x.shape[0] <= 1:
        return x.astype(np.float32, copy=False)

    ratio = float(dst_sr) / float(src_sr)
    n2 = max(1, int(round(x.shape[0] * ratio)))

    old = np.linspace(0.0, 1.0, x.shape[0], endpoint=False, dtype=np.float64)
    new = np.linspace(0.0, 1.0, n2, endpoint=False, dtype=np.float64)

    left = np.interp(new, old, x[:, 0]).astype(np.float32)
    right = np.interp(new, old, x[:, 1]).astype(np.float32)

    return np.stack([left, right], axis=1).astype(np.float32, copy=False)


def read_wav(path: str | os.PathLike[str], *, target_sr: Optional[int] = None) -> AudioBuffer:
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sampwidth = int(wf.getsampwidth())
        sr = int(wf.getframerate())
        raw = wf.readframes(int(wf.getnframes()))

    x = _pcm_to_float(raw, channels, sampwidth)

    if target_sr and int(target_sr) != sr:
        x = _resample_linear(x, sr, int(target_sr))
        sr = int(target_sr)

    return AudioBuffer(sanitize_audio(x), sr)


# ============================================================================
# Blocks framework
# ============================================================================

class BaseBlock:
    KIND: str = "fx"
    PARAMS: Dict[str, Dict[str, Any]] = {}

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(f"{self.__class__.__name__}.execute() not implemented")


class Registry:
    def __init__(self) -> None:
        self._by_name: Dict[str, type[BaseBlock]] = {}
        self._lock = threading.RLock()

    def register(self, name: str, cls: type[BaseBlock]) -> None:
        key = str(name).strip().lower()

        if not key:
            raise ValueError("Block name cannot be empty")

        with self._lock:
            self._by_name[key] = cls

    def names(self, *, kind: Optional[str] = None) -> List[str]:
        with self._lock:
            if kind is None:
                return sorted(self._by_name.keys())

            k = str(kind).lower()
            return sorted(
                n for n, c in self._by_name.items()
                if str(getattr(c, "KIND", "fx")).lower() == k
            )

    def create(self, name: str, **kwargs: Any) -> BaseBlock:
        key = str(name).strip().lower()

        with self._lock:
            cls = self._by_name.get(key)
            available = ", ".join(sorted(self._by_name.keys())) or "(none)"

        if cls is None:
            raise KeyError(f"Unknown block '{name}'. Available: {available}")

        return cls(**kwargs)

    def cls(self, name: str) -> type[BaseBlock]:
        key = str(name).strip().lower()

        with self._lock:
            cls = self._by_name.get(key)

        if cls is None:
            raise KeyError(f"Unknown block '{name}'")

        return cls


BLOCKS = Registry()


def register_block(*names: str):
    def decorator(cls: type[BaseBlock]) -> type[BaseBlock]:
        for name in names:
            BLOCKS.register(name, cls)
        return cls

    return decorator


def _as_audio_buffer(x: Any, sr: int) -> AudioBuffer:
    if isinstance(x, AudioBuffer):
        return AudioBuffer(sanitize_audio(x.data), int(x.sr))

    return AudioBuffer(sanitize_audio(x), int(sr))


def run_chain(
    payload: Any,
    chain: List[Tuple[BaseBlock, Dict[str, Any]]],
    *,
    common: Dict[str, Any] | None = None,
):
    common = common or {}
    x = payload

    for blk, p in chain:
        merged = {**common, **(p or {})}
        x, _ = blk.execute(x, params=merged)

    return x, {}


# ============================================================================
# Pitch helpers
# ============================================================================

NOTE_TO_SEMI = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def midi_from_note(note: str, octave: int) -> int:
    key = str(note).strip().upper().replace("♯", "#").replace("♭", "B")

    if key not in NOTE_TO_SEMI:
        raise KeyError(f"Unknown note name: {note}")

    return int((int(octave) + 1) * 12 + NOTE_TO_SEMI[key])


def hz_from_note(note: str, octave: int) -> float:
    midi = midi_from_note(note, octave)
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def note_from_midi(midi: int) -> Tuple[str, int]:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    midi = int(midi)
    return names[midi % 12], (midi // 12) - 1


# ============================================================================
# Sequencer model
# ============================================================================

@dataclass
class BlockInstance:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "BlockInstance":
        return BlockInstance(str(self.name), dict(self.params or {}))


@dataclass
class Track:
    name: str
    instruments: List[BlockInstance] = field(default_factory=list)
    fx: List[BlockInstance] = field(default_factory=list)
    volume: float = 1.0
    mute: bool = False
    solo: bool = False

    def clone(self) -> "Track":
        return Track(
            name=str(self.name),
            instruments=[b.clone() for b in self.instruments],
            fx=[b.clone() for b in self.fx],
            volume=float(self.volume),
            mute=bool(self.mute),
            solo=bool(self.solo),
        )


@dataclass
class NoteEvent:
    pitch: Tuple[str, int] = ("C", 4)
    start_step: int = 0
    length_steps: int = 1
    velocity: float = 1.0

    # Per-note amplitude envelope. These are seconds except sustain, which is a
    # level multiplier. They are applied after the selected instrument renders,
    # so every block hears the actual note-level ADSR during playback/export.
    attack: float = 0.005
    decay: float = 0.040
    sustain: float = 0.80
    release: float = 0.080

    def clone(self) -> "NoteEvent":
        return NoteEvent(
            pitch=(str(self.pitch[0]), int(self.pitch[1])),
            start_step=int(self.start_step),
            length_steps=max(1, int(self.length_steps)),
            velocity=float(np.clip(self.velocity, 0.0, 2.0)),
            attack=float(np.clip(float(self.attack), 0.0, 8.0)),
            decay=float(np.clip(float(self.decay), 0.0, 8.0)),
            sustain=float(np.clip(float(self.sustain), 0.0, 2.0)),
            release=float(np.clip(float(self.release), 0.0, 8.0)),
        )


@dataclass(frozen=True)
class SequenceSnapshot:
    sr: int
    bpm: float
    steps_per_bar: int
    bars: int
    tracks: Tuple[Track, ...]
    notes: Tuple[Tuple[NoteEvent, ...], ...]
    version: int


# ============================================================================
# Render cache
# ============================================================================

_RENDER_CACHE_MAX_ITEMS = int(os.getenv("MELODY_RENDER_CACHE_ITEMS", "256"))
_RENDER_CACHE_LOCK = threading.RLock()
_RENDER_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _stable_data(obj: Any) -> Any:
    if is_dataclass(obj):
        return _stable_data(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _stable_data(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}

    if isinstance(obj, (list, tuple)):
        return [_stable_data(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def _cache_hash(obj: Any) -> str:
    raw = json.dumps(_stable_data(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def clear_render_cache() -> None:
    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE.clear()


def _cache_get(key: str) -> Optional[np.ndarray]:
    with _RENDER_CACHE_LOCK:
        x = _RENDER_CACHE.get(key)

        if x is None:
            return None

        _RENDER_CACHE.move_to_end(key)
        return x.copy()


def _cache_put(key: str, value: np.ndarray) -> None:
    with _RENDER_CACHE_LOCK:
        _RENDER_CACHE[key] = value.copy()
        _RENDER_CACHE.move_to_end(key)

        while len(_RENDER_CACHE) > _RENDER_CACHE_MAX_ITEMS:
            _RENDER_CACHE.popitem(last=False)


def _track_key(
    snap: SequenceSnapshot,
    track_i: int,
    start_sample: int,
    window_n: int,
    step_n: int,
    render_tail_n: int,
) -> str:
    track = snap.tracks[track_i]
    notes = snap.notes[track_i] if track_i < len(snap.notes) else ()

    return _cache_hash(
        {
            "type": "melody_pipeline_track_v8_advanced_sound_base",
            "version": int(snap.version),
            "track_i": int(track_i),
            "sr": int(snap.sr),
            "bpm": float(snap.bpm),
            "steps_per_bar": int(snap.steps_per_bar),
            "bars": int(snap.bars),
            "start_sample": int(start_sample),
            "window_n": int(window_n),
            "step_n": int(step_n),
            "render_tail_n": int(render_tail_n),
            "track": track,
            "notes": notes,
        }
    )


# ============================================================================
# Utility DSP for pipeline base
# ============================================================================

def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _safe_gain(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _fade_edges(x: np.ndarray, fade_n: int) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)

    if y.shape[0] <= 2 or fade_n <= 1:
        return y

    n = min(int(fade_n), y.shape[0] // 2)

    if n <= 1:
        return y

    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)

    y = y.copy()
    y[:n] *= fade_in[:, None]
    y[-n:] *= fade_out[:, None]

    return y.astype(np.float32, copy=False)


def _note_adsr_values(ev: NoteEvent) -> Tuple[float, float, float, float]:
    """Return safe attack/decay/sustain/release values for one note."""
    return (
        float(np.clip(float(getattr(ev, "attack", 0.005)), 0.0, 8.0)),
        float(np.clip(float(getattr(ev, "decay", 0.040)), 0.0, 8.0)),
        float(np.clip(float(getattr(ev, "sustain", 0.80)), 0.0, 2.0)),
        float(np.clip(float(getattr(ev, "release", 0.080)), 0.0, 8.0)),
    )


def _adsr_envelope(
    frames: int,
    sr: int,
    *,
    hold_s: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    """
    Build a note-level ADSR envelope.

    The hold section equals the piano-roll note length. Release begins exactly
    when the note length ends, which means changing release on one selected note
    changes playback and WAV export for that note only.
    """
    frames = int(max(0, frames))
    sr = int(max(1, sr))

    if frames <= 0:
        return np.zeros((0,), dtype=np.float32)

    hold_n = int(max(1, round(max(0.0, float(hold_s)) * sr)))
    hold_n = int(min(hold_n, frames))

    attack_n = int(round(max(0.0, float(attack)) * sr))
    decay_n = int(round(max(0.0, float(decay)) * sr))
    release_n = int(round(max(0.0, float(release)) * sr))
    sustain = float(np.clip(float(sustain), 0.0, 2.0))

    # Keep attack+decay inside the piano-roll note body.
    if attack_n + decay_n > hold_n:
        total = max(1, attack_n + decay_n)
        scale = float(hold_n) / float(total)
        attack_n = int(round(attack_n * scale))
        decay_n = int(max(0, hold_n - attack_n))

    env = np.zeros((frames,), dtype=np.float32)

    pos = 0

    if attack_n > 0:
        end = min(frames, attack_n)
        env[:end] = np.linspace(0.0, 1.0, end, endpoint=False, dtype=np.float32)
        pos = end
    else:
        pos = 0

    if pos < hold_n:
        if decay_n > 0:
            end = min(hold_n, pos + decay_n)
            env[pos:end] = np.linspace(1.0, sustain, end - pos, endpoint=False, dtype=np.float32)
            pos = end

        if pos < hold_n:
            env[pos:hold_n] = sustain

    if hold_n > 0:
        if attack_n == 0 and decay_n == 0 and hold_n > 0:
            env[:hold_n] = sustain
            env[0] = max(env[0], 1.0)

        hold_level = float(env[hold_n - 1])

        if release_n > 0 and hold_n < frames:
            rel_end = min(frames, hold_n + release_n)
            env[hold_n:rel_end] = np.linspace(hold_level, 0.0, rel_end - hold_n, endpoint=False, dtype=np.float32)

    return np.clip(env, 0.0, 2.0).astype(np.float32, copy=False)



def _apply_note_adsr_native_or_python(
    audio: np.ndarray,
    sr: int,
    *,
    hold_s: float,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
) -> np.ndarray:
    x = ensure_stereo(audio).astype(np.float32, copy=False)
    hold_frames = int(max(1, round(max(0.0, float(hold_s)) * int(max(1, sr)))))

    if NATIVE_DSP is not None and hasattr(NATIVE_DSP, "apply_note_adsr"):
        try:
            return NATIVE_DSP.apply_note_adsr(
                x,
                sample_rate=int(sr),
                hold_frames=int(hold_frames),
                attack=float(attack),
                decay=float(decay),
                sustain=float(sustain),
                release=float(release),
                curve=1.0,
            )
        except Exception:
            pass

    env = _adsr_envelope(
        int(x.shape[0]),
        int(sr),
        hold_s=float(hold_s),
        attack=float(attack),
        decay=float(decay),
        sustain=float(sustain),
        release=float(release),
    )
    return (x * env[:, None]).astype(np.float32, copy=False)

def _dc_block_stereo(x: np.ndarray, amount: float = 0.995) -> np.ndarray:
    """
    DC blocker. Signature is unchanged, but this now uses
    NATIVE_DSP.dc_block() / mp_dc_block_stereo_f32 when available.
    """
    y = ensure_stereo(x).astype(np.float32, copy=False)

    if y.shape[0] <= 1:
        return y

    if NATIVE_DSP is not None and hasattr(NATIVE_DSP, "dc_block"):
        try:
            return NATIVE_DSP.dc_block(y, amount=float(amount))
        except Exception:
            pass

    amount = float(np.clip(amount, 0.90, 0.9999))
    out = np.empty_like(y, dtype=np.float32)

    x1_l = x1_r = 0.0
    y1_l = y1_r = 0.0

    for i in range(y.shape[0]):
        xl = float(y[i, 0])
        xr = float(y[i, 1])

        yl = xl - x1_l + amount * y1_l
        yr = xr - x1_r + amount * y1_r

        out[i, 0] = yl
        out[i, 1] = yr

        x1_l, x1_r = xl, xr
        y1_l, y1_r = yl, yr

    return out.astype(np.float32, copy=False)

def _soft_saturate(x: np.ndarray, drive: float = 1.0, ceiling: float = 0.98) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)
    drive = float(np.clip(drive, 0.05, 12.0))
    ceiling = float(np.clip(ceiling, 0.05, 1.0))

    y = np.tanh(y * drive).astype(np.float32)
    mx = float(np.max(np.abs(y))) if y.size else 0.0

    if mx > ceiling and mx > 1.0e-9:
        y = y * (ceiling / mx)

    return y.astype(np.float32, copy=False)


def _onepole_lowpass_stereo(x: np.ndarray, sr: int, cutoff_hz: float, wet: float = 1.0) -> np.ndarray:
    y = ensure_stereo(x).astype(np.float32, copy=False)
    sr = int(max(1, sr))
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    wet = float(np.clip(wet, 0.0, 1.0))

    if wet <= 1.0e-8 or y.shape[0] <= 1:
        return y

    a = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / float(sr))
    a = float(np.clip(a, 0.0, 1.0))

    out = np.empty_like(y, dtype=np.float32)
    state = y[0].copy()
    out[0] = state

    for i in range(1, y.shape[0]):
        state = state + a * (y[i] - state)
        out[i] = state

    return ((1.0 - wet) * y + wet * out).astype(np.float32, copy=False)


def _master_finish(x: np.ndarray, sr: int, *, common: Dict[str, Any]) -> np.ndarray:
    """
    Final master safety/polish. Signature is unchanged, but this now prefers
    NATIVE_DSP.master_finish() / mp_master_finish_stereo_f32 when available.
    """
    y = ensure_stereo(x).astype(np.float32, copy=False)

    if y.shape[0] <= 0:
        return y

    master_gain = float(common.get("master_gain", 1.0))
    master_drive = float(common.get("master_drive", 1.04))
    master_ceiling = float(common.get("master_ceiling", 0.98))
    master_lowpass_hz = float(common.get("master_lowpass_hz", 20500.0))
    lowpass_wet = float(common.get("master_lowpass_wet", 0.35))
    dc_amount = float(common.get("master_dc_amount", 0.995))

    if NATIVE_DSP is not None and hasattr(NATIVE_DSP, "master_finish"):
        try:
            return NATIVE_DSP.master_finish(
                y,
                sample_rate=int(sr),
                master_gain=float(master_gain),
                master_drive=float(master_drive),
                master_ceiling=float(master_ceiling),
                master_lowpass_hz=float(master_lowpass_hz),
                lowpass_wet=float(lowpass_wet),
                dc_amount=float(dc_amount),
            )
        except Exception:
            pass

    y = _dc_block_stereo(y, amount=dc_amount)
    y *= master_gain

    if master_lowpass_hz > 0:
        y = _onepole_lowpass_stereo(y, int(sr), master_lowpass_hz, wet=lowpass_wet)

    # Tiny glue stage. This makes layered synthetic sounds feel less separate
    # without killing the block-specific timbre.
    y = _soft_saturate(y, drive=master_drive, ceiling=master_ceiling)

    return sanitize_audio(y, ceiling=master_ceiling)


# ============================================================================
# Param normalization
# ============================================================================

def _block_kind_safe(block_name: str) -> str:
    try:
        return str(getattr(BLOCKS.cls(block_name), "KIND", "fx")).lower()
    except Exception:
        return "fx"


def _normalize_block_params(
    block_name: str,
    params: Dict[str, Any] | None,
    *,
    sr: int,
    preview: bool,
    track_index: int,
    track_name: str,
    block_index: int,
    block_kind: str,
    render_seed: int,
    start_sample: int,
) -> Dict[str, Any]:
    """
    This lets old CLI/demo names work with newer sound blocks.

    Examples:
    - gain_db -> gain
    - time_ms -> delay_ms
    - mix -> wet
    - waveform -> wave
    - amp/gain/master_gain are synchronized for instruments
    """
    p: Dict[str, Any] = dict(params or {})

    p.setdefault("sr", int(sr))
    p.setdefault("sample_rate", int(sr))
    p.setdefault("preview", bool(preview))
    p.setdefault("track_index", int(track_index))
    p.setdefault("track_name", str(track_name))
    p.setdefault("block_index", int(block_index))
    p.setdefault("block_kind", str(block_kind))
    p.setdefault("render_seed", int(render_seed))
    p.setdefault("window_start_sample", int(start_sample))

    # Gain aliases.
    if "gain_db" in p and "gain" not in p:
        p["gain"] = _db_to_gain(float(p["gain_db"]))
    if "db" in p and "gain" not in p:
        p["gain"] = _db_to_gain(float(p["db"]))

    # Delay aliases.
    if "time_ms" in p and "delay_ms" not in p:
        p["delay_ms"] = p["time_ms"]
    if "delay_time_ms" in p and "delay_ms" not in p:
        p["delay_ms"] = p["delay_time_ms"]
    if "mix" in p and "wet" not in p:
        p["wet"] = p["mix"]

    # Filter aliases.
    if "cutoff" in p and "cutoff_hz" not in p:
        p["cutoff_hz"] = p["cutoff"]

    # Instrument aliases.
    if "waveform" in p and "wave" not in p:
        p["wave"] = p["waveform"]
    if "wave" in p and "waveform" not in p:
        p["waveform"] = p["wave"]

    if block_kind == "instrument":
        if "master_gain" in p and "amp" not in p:
            p["amp"] = p["master_gain"]
        if "gain" in p and "amp" not in p:
            p["amp"] = p["gain"]
        if "amp" in p and "gain" not in p:
            p["gain"] = p["amp"]
        if "amp" in p and "master_gain" not in p:
            p["master_gain"] = p["amp"]

    return p


def _param_float(params: Dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in params:
            try:
                return float(params[name])
            except Exception:
                pass
    return float(default)


def _instrument_tail_seconds(params: Dict[str, Any]) -> float:
    """
    Lets notes ring/release instead of being hard-cut at step length.
    This is one of the biggest improvements for more identifiable instruments.
    """
    release = _param_float(params, "release", "release_seconds", default=0.10)
    decay = _param_float(params, "decay", "body_decay", "pluck_decay", default=0.0)
    reverb_mix = _param_float(params, "reverb_mix", default=0.0)
    delay_wet = _param_float(params, "wet", "mix", default=0.0)
    delay_ms = _param_float(params, "delay_ms", "time_ms", default=0.0)

    tail = max(0.0, release)

    if decay > 0.5:
        tail = max(tail, min(2.5, decay * 0.35))

    if reverb_mix > 0.01:
        tail = max(tail, 0.75 + reverb_mix * 1.25)

    if delay_wet > 0.01 and delay_ms > 0.0:
        tail = max(tail, min(3.0, delay_ms / 1000.0 * 3.0))

    return float(np.clip(tail, 0.0, 4.0))


def _track_fx_tail_seconds(track: Track) -> float:
    tail = 0.0

    for fx in track.fx:
        name = str(fx.name).lower()
        p = dict(fx.params or {})

        if name in {"delay", "echo"}:
            delay_ms = _param_float(p, "delay_ms", "time_ms", "delay_time_ms", default=250.0)
            feedback = _param_float(p, "feedback", default=0.25)
            wet = _param_float(p, "wet", "mix", default=0.35)
            if wet > 0.001:
                tail = max(tail, min(5.0, (delay_ms / 1000.0) * (2.0 + 5.0 * feedback)))

        if "reverb" in name or name in {"sound_polish", "polish"}:
            mix = _param_float(p, "reverb_mix", "mix", default=0.0)
            room = _param_float(p, "reverb_room", "room", default=0.5)
            if mix > 0.001:
                tail = max(tail, 0.75 + room * 1.75)

    return float(np.clip(tail, 0.0, 5.0))


# ============================================================================
# Sequence
# ============================================================================

@dataclass
class Sequence:
    sr: int = 48000
    bpm: float = 120.0
    steps_per_bar: int = 16
    bars: int = 2
    tracks: List[Track] = field(default_factory=list)
    notes: List[List[NoteEvent]] = field(default_factory=list)

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _version: int = field(default=0, init=False, repr=False)

    def touch(self) -> None:
        with self._lock:
            self._version += 1
        clear_render_cache()

    @property
    def version(self) -> int:
        with self._lock:
            return int(self._version)

    def clear_render_cache(self) -> None:
        clear_render_cache()

    def total_steps(self) -> int:
        return int(self.steps_per_bar) * int(self.bars)

    def step_seconds(self) -> float:
        bpm = max(1.0e-6, float(self.bpm))
        return (4.0 * (60.0 / bpm)) / float(max(1, int(self.steps_per_bar)))

    def step_samples(self) -> int:
        return max(1, int(round(self.step_seconds() * int(self.sr))))

    def total_samples(self) -> int:
        return self.step_samples() * self.total_steps()

    def ensure(self) -> None:
        with self._lock:
            while len(self.notes) < len(self.tracks):
                self.notes.append([])
            while len(self.notes) > len(self.tracks):
                self.notes.pop()

    def ensure_grid(self) -> None:
        self.ensure()

    def snapshot(self) -> SequenceSnapshot:
        self.ensure()

        with self._lock:
            tracks = tuple(t.clone() for t in self.tracks)
            notes = tuple(tuple(ev.clone() for ev in track_notes) for track_notes in self.notes)

            return SequenceSnapshot(
                sr=int(self.sr),
                bpm=float(self.bpm),
                steps_per_bar=int(self.steps_per_bar),
                bars=int(self.bars),
                tracks=tracks,
                notes=notes,
                version=int(self._version),
            )

    def _clamp_len(self, start_step: int, length_steps: int) -> int:
        length_steps = max(1, int(length_steps))
        max_len = max(1, self.total_steps() - int(start_step))
        return min(length_steps, max_len)

    def add_note(
        self,
        track_i: int,
        start_step: int,
        pitch: Tuple[str, int],
        length_steps: int = 1,
        velocity: float = 1.0,
    ) -> int:
        self.ensure()

        with self._lock:
            if not self.notes:
                return -1

            track_i = int(max(0, min(len(self.notes) - 1, int(track_i))))
            start_step = int(max(0, min(self.total_steps() - 1, int(start_step))))
            length_steps = self._clamp_len(start_step, length_steps)

            end = start_step + length_steps
            pitch = (str(pitch[0]), int(pitch[1]))
            new_list: List[NoteEvent] = []

            for ev in self.notes[track_i]:
                if ev.pitch != pitch:
                    new_list.append(ev)
                    continue

                ev_end = int(ev.start_step) + int(ev.length_steps)

                if ev_end <= start_step or int(ev.start_step) >= end:
                    new_list.append(ev)

            new_list.append(
                NoteEvent(
                    pitch=pitch,
                    start_step=start_step,
                    length_steps=length_steps,
                    velocity=float(np.clip(velocity, 0.0, 2.0)),
                )
            )

            new_list.sort(key=lambda e: (int(e.start_step), int(e.pitch[1]), str(e.pitch[0])))
            self.notes[track_i] = new_list
            self._version += 1
            clear_render_cache()

            for i, ev in enumerate(self.notes[track_i]):
                if ev.start_step == start_step and ev.pitch == pitch and ev.length_steps == length_steps:
                    return i

            return len(self.notes[track_i]) - 1

    def set_note(
        self,
        track_i: int,
        start_step: int,
        pitch: Tuple[str, int],
        length_steps: int = 1,
        velocity: float = 1.0,
    ) -> int:
        return self.add_note(track_i, start_step, pitch, length_steps, velocity)

    def find_note_covering(self, track_i: int, step: int, pitch: Tuple[str, int]) -> Optional[int]:
        self.ensure()

        with self._lock:
            if not (0 <= int(track_i) < len(self.notes)):
                return None

            step = int(step)
            pitch = (str(pitch[0]), int(pitch[1]))

            for i, ev in enumerate(self.notes[int(track_i)]):
                if ev.pitch == pitch and int(ev.start_step) <= step < int(ev.start_step) + int(ev.length_steps):
                    return i

        return None

    def find_note_starting_at(self, track_i: int, start_step: int, pitch: Tuple[str, int]) -> Optional[int]:
        self.ensure()

        with self._lock:
            if not (0 <= int(track_i) < len(self.notes)):
                return None

            start_step = int(start_step)
            pitch = (str(pitch[0]), int(pitch[1]))

            for i, ev in enumerate(self.notes[int(track_i)]):
                if int(ev.start_step) == start_step and ev.pitch == pitch:
                    return i

        return None

    def remove_note(self, track_i: int, note_index: int) -> None:
        self.ensure()

        with self._lock:
            if 0 <= int(track_i) < len(self.notes) and 0 <= int(note_index) < len(self.notes[int(track_i)]):
                self.notes[int(track_i)].pop(int(note_index))
                self._version += 1
                clear_render_cache()

    def set_note_length(self, track_i: int, note_index: int, length_steps: int) -> None:
        self.ensure()

        with self._lock:
            if not (0 <= int(track_i) < len(self.notes)):
                return
            if not (0 <= int(note_index) < len(self.notes[int(track_i)])):
                return

            ev = self.notes[int(track_i)][int(note_index)]
            ev.length_steps = self._clamp_len(int(ev.start_step), int(length_steps))
            self._version += 1
            clear_render_cache()

    def set_note_velocity(self, track_i: int, note_index: int, velocity: float) -> None:
        self.ensure()

        with self._lock:
            if 0 <= int(track_i) < len(self.notes) and 0 <= int(note_index) < len(self.notes[int(track_i)]):
                self.notes[int(track_i)][int(note_index)].velocity = float(np.clip(velocity, 0.0, 2.0))
                self._version += 1
                clear_render_cache()

    def set_note_adsr(
        self,
        track_i: int,
        note_index: int,
        *,
        attack: Optional[float] = None,
        decay: Optional[float] = None,
        sustain: Optional[float] = None,
        release: Optional[float] = None,
    ) -> None:
        self.ensure()

        with self._lock:
            if not (0 <= int(track_i) < len(self.notes)):
                return
            if not (0 <= int(note_index) < len(self.notes[int(track_i)])):
                return

            ev = self.notes[int(track_i)][int(note_index)]

            if attack is not None:
                ev.attack = float(np.clip(float(attack), 0.0, 8.0))
            if decay is not None:
                ev.decay = float(np.clip(float(decay), 0.0, 8.0))
            if sustain is not None:
                ev.sustain = float(np.clip(float(sustain), 0.0, 2.0))
            if release is not None:
                ev.release = float(np.clip(float(release), 0.0, 8.0))

            self._version += 1
            clear_render_cache()

    def set_note_param(self, track_i: int, note_index: int, param_name: str, value: Any) -> None:
        name = str(param_name).strip().lower()

        if name == "velocity":
            self.set_note_velocity(track_i, note_index, float(value))
            return

        if name == "length_steps":
            self.set_note_length(track_i, note_index, int(value))
            return

        if name in {"attack", "decay", "sustain", "release"}:
            kwargs = {"attack": None, "decay": None, "sustain": None, "release": None}
            kwargs[name] = float(value)
            self.set_note_adsr(track_i, note_index, **kwargs)
            return

    def toggle_note_at(self, track_i: int, start_step: int, pitch: Tuple[str, int]) -> None:
        idx = self.find_note_starting_at(track_i, start_step, pitch)

        if idx is not None:
            self.remove_note(track_i, idx)
        else:
            self.add_note(track_i, start_step, pitch, 1)

    def render(self, *, common: Dict[str, Any] | None = None) -> AudioBuffer:
        return _render_snapshot(self.snapshot(), start_sample=0, max_samples=None, common=common)

    def render_from(self, start_sample: int, *, common: Dict[str, Any] | None = None) -> AudioBuffer:
        return _render_snapshot(self.snapshot(), start_sample=int(start_sample), max_samples=None, common=common)

    def render_window(
        self,
        start_sample: int,
        max_samples: int,
        *,
        common: Dict[str, Any] | None = None,
    ) -> AudioBuffer:
        return _render_snapshot(
            self.snapshot(),
            start_sample=int(start_sample),
            max_samples=int(max_samples),
            common=common,
        )


# ============================================================================
# Render engine
# ============================================================================

def _mix_into(dst: np.ndarray, src: np.ndarray, gain: float = 1.0, *, dst_offset: int = 0) -> np.ndarray:
    """
    Mix a stereo source into a stereo destination.

    Signature is unchanged, but this now prefers the MelodyProject.dll
    mp_mix_rendered_note_layer_stereo_f32 fast path when the wrapper exposes it.
    """
    if dst.size == 0:
        return dst

    src = ensure_stereo(src)
    dst_offset = int(dst_offset)

    n = min(int(dst.shape[0]) - max(0, dst_offset), int(src.shape[0]))

    if n <= 0:
        return dst

    if NATIVE_DSP is not None and hasattr(NATIVE_DSP, "mix_rendered_note_layer"):
        try:
            return NATIVE_DSP.mix_rendered_note_layer(
                dst,
                src,
                dst_offset=int(dst_offset),
                src_offset=0,
                frames_to_mix=int(n),
                gain=float(gain),
            )
        except Exception:
            pass

    dst_offset = int(max(0, dst_offset))

    if dst_offset >= dst.shape[0]:
        return dst

    n = min(dst.shape[0] - dst_offset, src.shape[0])

    if n > 0:
        dst[dst_offset:dst_offset + n] += src[:n] * float(gain)

    return dst

def _make_note_payload(
    snap: SequenceSnapshot,
    ev: NoteEvent,
    *,
    track_i: int,
    track_name: str,
    note_i: int,
    inst_i: int,
    freq: float,
    midi: int,
    duration_s: float,
    hold_s: float,
    start_sample_abs: int,
    step_s: float,
    step_n: int,
    render_seed: int,
) -> Dict[str, Any]:
    note, octv = ev.pitch
    velocity = float(np.clip(ev.velocity, 0.0, 2.0))

    return {
        "freq": float(freq),
        "hz": float(freq),
        "midi": int(midi),
        "note": str(note),
        "note_name": str(note),
        "octave": int(octv),
        "pitch": (str(note), int(octv)),

        "duration": float(duration_s),
        "duration_s": float(duration_s),
        "hold_duration": float(hold_s),
        "hold_duration_s": float(hold_s),

        "sr": int(snap.sr),
        "sample_rate": int(snap.sr),

        "velocity": velocity,
        "vel": velocity,

        "attack": float(getattr(ev, "attack", 0.005)),
        "attack_s": float(getattr(ev, "attack", 0.005)),
        "decay": float(getattr(ev, "decay", 0.040)),
        "decay_s": float(getattr(ev, "decay", 0.040)),
        "sustain": float(getattr(ev, "sustain", 0.80)),
        "sustain_level": float(getattr(ev, "sustain", 0.80)),
        "release": float(getattr(ev, "release", 0.080)),
        "release_s": float(getattr(ev, "release", 0.080)),

        "track_index": int(track_i),
        "track_name": str(track_name),
        "note_index": int(note_i),
        "instrument_index": int(inst_i),

        "start_step": int(ev.start_step),
        "length_steps": int(ev.length_steps),
        "step_seconds": float(step_s),
        "step_samples": int(step_n),
        "start_sample": int(start_sample_abs),

        # Stable per-note seed. Sound blocks can use this for consistent pick,
        # breath, hammer, phase, noise, and stereo micro variation.
        "seed": int(render_seed),
        "phase_seed": int(render_seed),
    }



_NATIVE_SYNTH_ADSR_NAMES = {
    "synth_keys",
    "basic_synth",
    "oscillator",
    "piano_key",
    "piano_keys",
}

_NATIVE_SYNTH_ADSR_ALLOWED_PARAMS = {
    "wave",
    "waveform",
    "amp",
    "gain",
    "master_gain",
    "pan",
    "native_fast_path",
}


def _native_method_available(name: str) -> bool:
    if NATIVE_DSP is None:
        return False
    if not hasattr(NATIVE_DSP, name):
        return False
    try:
        return bool(getattr(NATIVE_DSP, "available", True))
    except Exception:
        return True


def _param_bool(params: Dict[str, Any], name: str, default: bool = False) -> bool:
    if name not in params:
        return bool(default)

    value = params.get(name)

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}

    return bool(value)


def _native_synth_param_is_simple(key: str, value: Any) -> bool:
    k = str(key).strip().lower()

    if k in _NATIVE_SYNTH_ADSR_ALLOWED_PARAMS:
        return True

    # These are common richer synth params. If they are unset/neutral, the
    # native single-oscillator path is still faithful enough; if they are active,
    # we keep the Python block path so the sound does not change unexpectedly.
    neutral_zero = {
        "unison",
        "detune",
        "detune_cents",
        "chorus",
        "chorus_mix",
        "reverb_mix",
        "noise",
        "noise_mix",
        "vibrato",
        "vibrato_depth",
        "filter_drive",
        "drive",
        "warmth",
    }

    if k in neutral_zero:
        try:
            if k == "unison":
                return int(value) <= 1
            return abs(float(value)) <= 1.0e-8
        except Exception:
            return False

    # Attack/release on the instrument is ignored by the native per-note ADSR
    # path because the selected NoteEvent owns the note envelope.
    if k in {"attack", "decay", "sustain", "release", "attack_s", "decay_s", "release_s"}:
        return True

    # Metadata added by normalization/rendering is not part of tone design.
    if k in {
        "sr", "sample_rate", "preview", "track_index", "track_name",
        "block_index", "block_kind", "render_seed", "window_start_sample",
        "seed", "note_seed", "note_index", "instrument_index",
    }:
        return True

    return False


def _track_can_use_native_synth_adsr(track: Track) -> bool:
    if not _native_method_available("render_synth_notes_adsr"):
        return False

    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]

    if len(instruments) != 1:
        return False

    inst = instruments[0]
    name = str(inst.name).strip().lower()

    if name not in _NATIVE_SYNTH_ADSR_NAMES:
        return False

    params = dict(inst.params or {})

    if _param_bool(params, "native_fast_path", True) is False:
        return False

    for k, v in params.items():
        if not _native_synth_param_is_simple(k, v):
            return False

    return True


def _render_native_synth_adsr_track(
    snap: SequenceSnapshot,
    track: Track,
    notes: Tuple[NoteEvent, ...],
    *,
    track_i: int,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
    preview: bool,
) -> np.ndarray:
    """
    Native batched synth + per-note ADSR path.

    This keeps the existing render signatures, but when the track is a simple
    synth track it sends arrays of all notes to MelodyProject.dll in one call:
    mp_render_synth_notes_adsr_stereo_f32.
    """
    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]
    inst = instruments[0]
    p = dict(inst.params or {})

    waveform = p.get("wave", p.get("waveform", "sine"))
    gain = _safe_gain(p.get("amp", p.get("gain", p.get("master_gain", 0.22))), 0.22)
    pan = float(np.clip(float(p.get("pan", 0.0)), -1.0, 1.0))

    midi_notes: List[int] = []
    start_frames: List[int] = []
    length_frames: List[int] = []
    velocities: List[float] = []
    attacks: List[float] = []
    decays: List[float] = []
    sustains: List[float] = []
    releases: List[float] = []

    for ev in notes:
        note_start_abs = int(ev.start_step) * step_n
        hold_n = max(1, int(ev.length_steps) * step_n)
        attack_s, decay_s, sustain_level, release_s = _note_adsr_values(ev)
        release_n = int(round(max(0.0, release_s) * int(snap.sr)))
        note_end_abs = note_start_abs + hold_n + release_n

        if note_end_abs <= start_sample or note_start_abs >= start_sample + window_n:
            continue

        note, octv = ev.pitch

        try:
            midi = midi_from_note(note, octv)
        except Exception:
            continue

        midi_notes.append(int(midi))
        start_frames.append(int(note_start_abs - start_sample))
        length_frames.append(int(hold_n))
        velocities.append(float(np.clip(float(ev.velocity), 0.0, 2.0)))
        attacks.append(float(attack_s))
        decays.append(float(decay_s))
        sustains.append(float(sustain_level))
        releases.append(float(release_s))

    if not midi_notes:
        return np.zeros((window_n, 2), dtype=np.float32)

    try:
        out = NATIVE_DSP.render_synth_notes_adsr(
            np.asarray(midi_notes, dtype=np.int32),
            np.asarray(start_frames, dtype=np.int32),
            np.asarray(length_frames, dtype=np.int32),
            np.asarray(velocities, dtype=np.float32),
            np.asarray(attacks, dtype=np.float32),
            np.asarray(decays, dtype=np.float32),
            np.asarray(sustains, dtype=np.float32),
            np.asarray(releases, dtype=np.float32),
            total_frames=int(window_n),
            sample_rate=int(snap.sr),
            waveform=waveform,
            master_gain=float(gain),
            pan=float(pan),
            clear_output=True,
        )
    except Exception as exc:
        print(f"[pipeline] native synth ADSR fast path failed on track '{track.name}': {exc}")
        return _render_python_instrument_track(
            snap,
            track,
            notes,
            track_i=track_i,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
            preview=preview,
        )

    out = _dc_block_stereo(out)
    out = sanitize_audio(out, ceiling=1.25)
    return out.astype(np.float32, copy=False)


def _render_python_instrument_track(
    snap: SequenceSnapshot,
    track: Track,
    notes: Tuple[NoteEvent, ...],
    *,
    track_i: int,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
    preview: bool,
) -> np.ndarray:
    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]
    inst_gens: List[Tuple[BaseBlock, Dict[str, Any], str]] = []

    for inst_i, inst in enumerate(instruments):
        name = str(inst.name).strip().lower()

        try:
            blk = BLOCKS.create(name)
            kind = _block_kind_safe(name)
            p = _normalize_block_params(
                name,
                dict(inst.params or {}),
                sr=int(snap.sr),
                preview=bool(preview),
                track_index=int(track_i),
                track_name=str(track.name),
                block_index=int(inst_i),
                block_kind=kind,
                render_seed=1009 + track_i * 131 + inst_i * 17,
                start_sample=int(start_sample),
            )
            inst_gens.append((blk, p, name))
        except Exception as exc:
            print(f"[pipeline] missing instrument '{name}': {exc}")

    if not inst_gens:
        return np.zeros((window_n, 2), dtype=np.float32)

    tbuf = np.zeros((window_n, 2), dtype=np.float32)
    fade_n = max(8, int(round(0.0015 * int(snap.sr))))

    for note_i, ev in enumerate(notes):
        dur_steps = max(1, int(ev.length_steps))
        hold_s = dur_steps * step_s
        note_start_abs = int(ev.start_step) * step_n

        note, octv = ev.pitch

        try:
            midi = midi_from_note(note, octv)
            freq = hz_from_note(note, octv)
        except Exception:
            continue

        # Compute the longest layer length for this note.
        voice_specs: List[Tuple[BaseBlock, Dict[str, Any], str, int, float]] = []
        max_voice_n = 0

        attack_s, decay_s, sustain_level, release_s = _note_adsr_values(ev)

        for inst_i, (gen, p, name) in enumerate(inst_gens):
            tail_s = max(_instrument_tail_seconds(p), release_s)
            voice_s = float(max(0.001, hold_s + tail_s))
            voice_n = max(1, int(round(voice_s * int(snap.sr))))
            max_voice_n = max(max_voice_n, voice_n)
            voice_specs.append((gen, p, name, inst_i, voice_s))

        if max_voice_n <= 0:
            continue

        note_end_abs = note_start_abs + max_voice_n

        if note_end_abs <= start_sample or note_start_abs >= start_sample + window_n:
            continue

        layer = np.zeros((max_voice_n, 2), dtype=np.float32)

        for gen, p, name, inst_i, voice_s in voice_specs:
            render_seed = int(
                1000003
                + snap.version * 101
                + track_i * 1009
                + note_i * 9176
                + inst_i * 431
                + int(ev.start_step) * 13
                + midi * 7
            )

            pp = dict(p)
            pp["seed"] = int(pp.get("seed", render_seed))
            pp["render_seed"] = render_seed
            pp["note_seed"] = render_seed
            pp["note_index"] = int(note_i)
            pp["instrument_index"] = int(inst_i)

            payload = _make_note_payload(
                snap,
                ev,
                track_i=track_i,
                track_name=track.name,
                note_i=note_i,
                inst_i=inst_i,
                freq=freq,
                midi=midi,
                duration_s=voice_s,
                hold_s=hold_s,
                start_sample_abs=note_start_abs,
                step_s=step_s,
                step_n=step_n,
                render_seed=render_seed,
            )

            try:
                raw, _meta = gen.execute(payload, params=pp)
                y = _as_audio_buffer(raw, int(snap.sr)).data
            except Exception as exc:
                print(f"[pipeline] instrument '{name}' failed: {exc}")
                continue

            y = ensure_stereo(y).astype(np.float32, copy=False)
            y = _fade_edges(y, fade_n)

            if y.shape[0] < max_voice_n:
                padded = np.zeros((max_voice_n, 2), dtype=np.float32)
                padded[:y.shape[0]] = y
                y = padded
            elif y.shape[0] > max_voice_n:
                y = y[:max_voice_n]

            # Apply the selected note's own ADSR after instrument synthesis.
            # Uses the C++ DLL when available so live playback/export do not
            # spend time building and multiplying a Python envelope array.
            y = _apply_note_adsr_native_or_python(
                y,
                int(snap.sr),
                hold_s=hold_s,
                attack=attack_s,
                decay=decay_s,
                sustain=sustain_level,
                release=release_s,
            )

            # Layer headroom. More instruments should sound layered, not clipped.
            layer_gain = 1.0 / math.sqrt(max(1, len(inst_gens)))
            layer = _mix_into(layer, y, layer_gain)

        # Put rendered note layer into the requested render window.
        a_abs = max(note_start_abs, start_sample)
        b_abs = min(note_end_abs, start_sample + window_n, total_n)

        if b_abs <= a_abs:
            continue

        dst_a = a_abs - start_sample
        dst_b = b_abs - start_sample
        src_a = a_abs - note_start_abs
        src_b = src_a + (dst_b - dst_a)

        tbuf[dst_a:dst_b] += layer[src_a:src_b]

    # Per-track safety before FX.
    tbuf = _dc_block_stereo(tbuf)
    tbuf = sanitize_audio(tbuf, ceiling=1.25)
    return tbuf.astype(np.float32, copy=False)


def _render_dry_track(
    snap: SequenceSnapshot,
    track_i: int,
    *,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
    render_tail_n: int,
    preview: bool,
) -> np.ndarray:
    track = snap.tracks[track_i]
    notes = snap.notes[track_i] if track_i < len(snap.notes) else ()

    key = _track_key(snap, track_i, start_sample, window_n, step_n, render_tail_n)
    cached = _cache_get(key)

    if cached is not None:
        return cached

    if _track_can_use_native_synth_adsr(track):
        out = _render_native_synth_adsr_track(
            snap,
            track,
            notes,
            track_i=track_i,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
            preview=preview,
        )
    else:
        out = _render_python_instrument_track(
            snap,
            track,
            notes,
            track_i=track_i,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
            preview=preview,
        )

    _cache_put(key, out)
    return out


def _apply_fx_chain(
    tbuf: np.ndarray,
    snap: SequenceSnapshot,
    track: Track,
    *,
    track_i: int,
    start_sample: int,
    preview: bool,
    common: Dict[str, Any],
) -> np.ndarray:
    fx_chain: List[Tuple[BaseBlock, Dict[str, Any]]] = []

    for fx_i, bi in enumerate(track.fx):
        name = str(bi.name).strip().lower()

        try:
            blk = BLOCKS.create(name)
            kind = _block_kind_safe(name)
            p = _normalize_block_params(
                name,
                dict(bi.params or {}),
                sr=int(snap.sr),
                preview=bool(preview),
                track_index=int(track_i),
                track_name=str(track.name),
                block_index=int(fx_i),
                block_kind=kind,
                render_seed=9001 + track_i * 353 + fx_i * 29,
                start_sample=int(start_sample),
            )
            fx_chain.append((blk, p))
        except Exception as exc:
            print(f"[pipeline] missing fx '{name}': {exc}")

    if not fx_chain:
        return tbuf

    try:
        fx_out, _ = run_chain(
            AudioBuffer(tbuf, int(snap.sr)),
            fx_chain,
            common={
                **common,
                "sr": int(snap.sr),
                "sample_rate": int(snap.sr),
                "preview": bool(preview),
                "track_index": int(track_i),
                "track_name": str(track.name),
                "window_start_sample": int(start_sample),
            },
        )
        y = _as_audio_buffer(fx_out, int(snap.sr)).data
    except Exception as exc:
        print(f"[pipeline] fx chain failed: {exc}")
        y = tbuf

    y = _dc_block_stereo(y)
    y = sanitize_audio(y, ceiling=1.25)
    return y.astype(np.float32, copy=False)


def _render_snapshot(
    snap: SequenceSnapshot,
    *,
    start_sample: int = 0,
    max_samples: Optional[int] = None,
    common: Dict[str, Any] | None = None,
) -> AudioBuffer:
    common = dict(common or {})

    sr = int(snap.sr)
    step_s = (4.0 * (60.0 / max(1.0e-6, float(snap.bpm)))) / float(max(1, int(snap.steps_per_bar)))
    step_n = max(1, int(round(step_s * sr)))
    base_total_n = int(snap.steps_per_bar) * int(snap.bars) * step_n

    # Export/full render should preserve instrument release and FX tails.
    # Preview windows still stay capped by max_samples for responsiveness.
    tail_s = float(common.get("render_tail_seconds", 0.75))

    for track_notes in snap.notes:
        for ev in track_notes:
            _a, _d, _s, release_s = _note_adsr_values(ev)
            tail_s = max(tail_s, release_s)

    for track in snap.tracks:
        for inst in track.instruments:
            p = _normalize_block_params(
                inst.name,
                dict(inst.params or {}),
                sr=sr,
                preview=max_samples is not None,
                track_index=0,
                track_name=track.name,
                block_index=0,
                block_kind="instrument",
                render_seed=0,
                start_sample=int(start_sample),
            )
            tail_s = max(tail_s, _instrument_tail_seconds(p))

        tail_s = max(tail_s, _track_fx_tail_seconds(track))

    if max_samples is not None:
        # Keep live preview fast, but still leave a little space for release.
        tail_s = min(tail_s, float(common.get("preview_tail_seconds", 0.75)))

    render_tail_n = int(round(max(0.0, tail_s) * sr))
    total_n = max(0, base_total_n + render_tail_n)

    start_sample = int(max(0, start_sample))

    if max_samples is None:
        window_n = max(0, total_n - start_sample)
    else:
        window_n = max(0, min(int(max_samples), total_n - start_sample))

    if window_n <= 0:
        return AudioBuffer.silence(0, sr)

    preview = max_samples is not None
    any_solo = any(bool(t.solo) for t in snap.tracks)
    out = np.zeros((window_n, 2), dtype=np.float32)

    active_tracks = [
        i for i, t in enumerate(snap.tracks)
        if not bool(t.mute) and (not any_solo or bool(t.solo))
    ]

    if not active_tracks:
        return AudioBuffer.silence(window_n, sr)

    # Track summing headroom. Keeps layers loud enough without instant clipping.
    active_gain = 1.0 / math.sqrt(max(1, len(active_tracks)))

    for ti in active_tracks:
        track = snap.tracks[ti]

        dry = _render_dry_track(
            snap,
            ti,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
            render_tail_n=render_tail_n,
            preview=preview,
        )

        tbuf = _apply_fx_chain(
            dry,
            snap,
            track,
            track_i=ti,
            start_sample=start_sample,
            preview=preview,
            common=common,
        )

        track_volume = float(track.volume)
        if "track_gain_db" in common:
            track_volume *= _db_to_gain(float(common["track_gain_db"]))

        out = _mix_into(out, tbuf, track_volume * active_gain)

    out = _master_finish(out, sr, common=common)

    return AudioBuffer(out.astype(np.float32, copy=False), sr)


# ============================================================================
# Demo / ballast helpers used by GUI
# ============================================================================

class MemoryBallast:
    def __init__(self) -> None:
        self._chunks: list[bytearray] = []
        self._lock = threading.RLock()

    def set_target_mb(self, mb: int, *args: Any, **kwargs: Any) -> None:
        mb = int(max(0, mb))
        chunk_mb = int(kwargs.get("chunk_mb", 16))
        chunk_mb = max(1, chunk_mb)

        with self._lock:
            self._chunks.clear()
            remaining = mb

            while remaining > 0:
                take = min(chunk_mb, remaining)
                self._chunks.append(bytearray(take * 1024 * 1024))
                remaining -= take

    def release(self) -> None:
        with self._lock:
            self._chunks.clear()

    def clear(self) -> None:
        self.release()

    def held_mb(self) -> float:
        with self._lock:
            return float(sum(len(c) for c in self._chunks) / (1024 * 1024))


class CpuBallast:
    def __init__(self) -> None:
        self._target_pct = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def set_target_pct(self, pct: int) -> None:
        pct = int(max(0, min(80, pct)))

        with self._lock:
            self._target_pct = pct

        if pct <= 0:
            self.stop()
            return

        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="CpuBallast")
            self._thread.start()

    def _loop(self) -> None:
        period = 0.05

        while not self._stop.is_set():
            with self._lock:
                pct = int(self._target_pct)

            if pct <= 0:
                time.sleep(0.1)
                continue

            busy = period * (pct / 100.0)
            idle = max(0.001, period - busy)
            t0 = time.perf_counter()

            while time.perf_counter() - t0 < busy:
                _ = math.sqrt(12345.6789)

            time.sleep(idle)

    def stop(self) -> None:
        self._stop.set()


def build_demo_project() -> Sequence:
    seq = Sequence(sr=48000, bpm=120.0, steps_per_bar=16, bars=2)
    seq.tracks = [
        Track(
            name="Track 1",
            instruments=[
                BlockInstance(
                    "synth_keys",
                    {
                        "wave": "saw",
                        "amp": 0.22,
                        "attack": 0.006,
                        "release": 0.18,
                        "unison": 3,
                        "detune_cents": 7.0,
                        "chorus_mix": 0.12,
                    },
                )
            ],
            fx=[
                BlockInstance("sound_polish", {"drive": 0.45, "warmth": 0.20}),
            ],
        ),
    ]
    seq.ensure()

    notes = [
        ("C", 4, 0),
        ("E", 4, 4),
        ("G", 4, 8),
        ("C", 5, 12),
        ("G", 4, 16),
        ("E", 4, 20),
        ("D", 4, 24),
        ("C", 4, 28),
    ]

    for note, octv, step in notes:
        seq.add_note(0, step, (note, octv), length_steps=3, velocity=1.0)

    return seq


class AnimationRenderer:
    pass