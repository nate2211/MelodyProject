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

from melodyproject_native import (
    NATIVE_DSP,
    ensure_stereo,
    native_status,
    waveform_id,
)

try:
    import psutil
except Exception:
    psutil = None  # type: ignore


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
    return NATIVE_DSP.sanitize(ensure_stereo(x), ceiling=float(ceiling))


def normalize_for_playback(buf: AudioBuffer, *, peak: float = 0.98, only_if_over: bool = True) -> AudioBuffer:
    x = NATIVE_DSP.soft_clip_normalize(
        buf.data,
        ceiling=0.995,
        peak=float(peak),
        only_if_over=bool(only_if_over),
    )
    return AudioBuffer(x, int(buf.sr))


def write_wav(path: str | os.PathLike[str], buf: AudioBuffer) -> None:
    x = normalize_for_playback(buf, peak=0.98, only_if_over=True).data
    pcm = np.clip(x, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(buf.sr))
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


# ============================================================================
# Native-backed built-in blocks
# ============================================================================

@register_block("synth_keys", "synth", "native_synth")
class SynthKeysBlock(BaseBlock):
    KIND = "instrument"
    PARAMS = {
        "waveform": {"type": "choice", "default": "sine", "choices": ["sine", "square", "saw", "triangle"]},
        "gain": {"type": "float", "default": 0.22, "min": 0.0, "max": 2.0},
        "attack": {"type": "float", "default": 0.004, "min": 0.0, "max": 2.0},
        "release": {"type": "float", "default": 0.035, "min": 0.0, "max": 5.0},
        "pan": {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if not isinstance(payload, dict):
            return AudioBuffer.silence(0, int(params.get("sr", 48000))), {}

        sr = int(payload.get("sr", params.get("sr", 48000)))
        duration = float(payload.get("duration", params.get("duration", 0.25)))
        frames = max(1, int(round(duration * sr)))

        freq = float(payload.get("freq", 440.0))
        midi = int(round(69.0 + 12.0 * math.log2(max(1.0e-9, freq) / 440.0)))
        velocity = float(payload.get("velocity", 1.0))

        y = NATIVE_DSP.render_synth_notes(
            np.asarray([midi], dtype=np.int32),
            np.asarray([0], dtype=np.int32),
            np.asarray([frames], dtype=np.int32),
            np.asarray([velocity], dtype=np.float32),
            total_frames=frames,
            sample_rate=sr,
            waveform=waveform_id(params.get("waveform", "sine")),
            master_gain=float(params.get("gain", params.get("master_gain", 0.22))),
            attack_seconds=float(params.get("attack", params.get("attack_seconds", 0.004))),
            release_seconds=float(params.get("release", params.get("release_seconds", 0.035))),
            pan=float(params.get("pan", 0.0)),
        )

        return AudioBuffer(y, sr), {"native": NATIVE_DSP.available}


@register_block("gain", "volume")
class GainBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "gain": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = NATIVE_DSP.gain(buf.data, float(params.get("gain", 1.0)))
        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available}


@register_block("lowpass", "filter_lowpass")
class LowpassBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "cutoff_hz": {"type": "float", "default": 6000.0, "min": 10.0, "max": 22000.0},
        "wet": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = NATIVE_DSP.lowpass(
            buf.data,
            sample_rate=buf.sr,
            cutoff_hz=float(params.get("cutoff_hz", params.get("cutoff", 6000.0))),
            wet=float(params.get("wet", 1.0)),
        )
        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available}


@register_block("delay")
class DelayBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "delay_ms": {"type": "float", "default": 250.0, "min": 1.0, "max": 5000.0},
        "feedback": {"type": "float", "default": 0.25, "min": 0.0, "max": 0.98},
        "wet": {"type": "float", "default": 0.35, "min": 0.0, "max": 2.0},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = NATIVE_DSP.delay(
            buf.data,
            sample_rate=buf.sr,
            delay_ms=float(params.get("delay_ms", 250.0)),
            feedback=float(params.get("feedback", 0.25)),
            wet=float(params.get("wet", 0.35)),
        )
        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available}


@register_block("stereo_width", "width")
class StereoWidthBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "width": {"type": "float", "default": 1.2, "min": 0.0, "max": 3.0},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = NATIVE_DSP.stereo_width(buf.data, width=float(params.get("width", 1.2)))
        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available}


@register_block("normalize", "limiter", "clip")
class ClipNormalizeBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "ceiling": {"type": "float", "default": 0.995, "min": 0.05, "max": 1.0},
        "peak": {"type": "float", "default": 0.98, "min": 0.05, "max": 1.0},
        "only_if_over": {"type": "bool", "default": True},
    }

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        y = NATIVE_DSP.soft_clip_normalize(
            buf.data,
            ceiling=float(params.get("ceiling", 0.995)),
            peak=float(params.get("peak", 0.98)),
            only_if_over=bool(params.get("only_if_over", True)),
        )
        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available}


@register_block("tonal_imprint", "convolution")
class TonalImprintBlock(BaseBlock):
    KIND = "fx"
    PARAMS = {
        "ir_path": {"type": "str", "default": ""},
        "dry": {"type": "float", "default": 0.7, "min": 0.0, "max": 2.0},
        "wet": {"type": "float", "default": 0.3, "min": 0.0, "max": 2.0},
        "normalize_ir": {"type": "bool", "default": True},
        "preview_max_ir_frames": {"type": "int", "default": 4096, "min": 64, "max": 96000},
        "full_max_ir_frames": {"type": "int", "default": 24000, "min": 64, "max": 192000},
    }

    _cache_lock = threading.RLock()
    _ir_cache: Dict[Tuple[str, int, int], np.ndarray] = {}

    def _default_ir(self, sr: int, max_frames: int) -> np.ndarray:
        frames = max(32, min(int(max_frames), int(sr * 0.035)))
        ir = np.zeros((frames, 2), dtype=np.float32)
        ir[0, :] = 1.0

        for i in range(1, frames):
            t = i / float(sr)
            decay = math.exp(-t * 45.0)
            ir[i, 0] = math.sin(i * 0.037) * decay * 0.18
            ir[i, 1] = math.sin(i * 0.041) * decay * 0.18

        return ir

    def _load_ir(self, path: str, sr: int, max_frames: int) -> np.ndarray:
        path = str(path or "").strip()

        if not path:
            return self._default_ir(sr, max_frames)

        p = Path(path).expanduser()

        if not p.exists():
            return self._default_ir(sr, max_frames)

        key = (str(p.resolve()), int(sr), int(max_frames))

        with self._cache_lock:
            hit = self._ir_cache.get(key)
            if hit is not None:
                return hit.copy()

        ir = read_wav(p, target_sr=sr).data
        ir = ensure_stereo(ir)[:max_frames]

        with self._cache_lock:
            self._ir_cache[key] = ir.copy()

        return ir

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        buf = _as_audio_buffer(payload, int(params.get("sr", 48000)))
        preview = bool(params.get("preview", False))

        max_ir = int(
            params.get(
                "preview_max_ir_frames" if preview else "full_max_ir_frames",
                4096 if preview else 24000,
            )
        )

        ir = self._load_ir(str(params.get("ir_path", "")), buf.sr, max_ir)

        y = NATIVE_DSP.convolve(
            buf.data,
            ir,
            dry=float(params.get("dry", 0.7)),
            wet=float(params.get("wet", 0.3)),
            normalize_ir=bool(params.get("normalize_ir", True)),
        )

        return AudioBuffer(y, buf.sr), {"native": NATIVE_DSP.available, "ir_frames": int(ir.shape[0])}


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

    def clone(self) -> "NoteEvent":
        return NoteEvent(
            pitch=(str(self.pitch[0]), int(self.pitch[1])),
            start_step=int(self.start_step),
            length_steps=max(1, int(self.length_steps)),
            velocity=float(np.clip(self.velocity, 0.0, 2.0)),
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

_RENDER_CACHE_MAX_ITEMS = int(os.getenv("MELODY_RENDER_CACHE_ITEMS", "48"))
_RENDER_CACHE_LOCK = threading.RLock()
_RENDER_CACHE: "OrderedDict[str, np.ndarray]" = OrderedDict()


def _stable_data(obj: Any) -> Any:
    if is_dataclass(obj):
        return _stable_data(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _stable_data(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}

    if isinstance(obj, (list, tuple)):
        return [_stable_data(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
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


def _dry_track_key(
    snap: SequenceSnapshot,
    track_i: int,
    start_sample: int,
    window_n: int,
    step_n: int,
) -> str:
    track = snap.tracks[track_i]
    notes = snap.notes[track_i] if track_i < len(snap.notes) else ()

    return _cache_hash(
        {
            "type": "native_dry_track_v3",
            "track_i": int(track_i),
            "sr": int(snap.sr),
            "bpm": float(snap.bpm),
            "steps_per_bar": int(snap.steps_per_bar),
            "bars": int(snap.bars),
            "start_sample": int(start_sample),
            "window_n": int(window_n),
            "step_n": int(step_n),
            "instruments": track.instruments,
            "notes": notes,
        }
    )


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

    @property
    def version(self) -> int:
        with self._lock:
            return int(self._version)

    def clear_render_cache(self) -> None:
        clear_render_cache()

    def total_steps(self) -> int:
        return int(self.steps_per_bar) * int(self.bars)

    def step_seconds(self) -> float:
        bpm = max(1e-6, float(self.bpm))
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
                    velocity=velocity,
                )
            )

            new_list.sort(key=lambda e: (int(e.start_step), int(e.pitch[1]), str(e.pitch[0])))
            self.notes[track_i] = new_list
            self._version += 1

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

    def set_note_velocity(self, track_i: int, note_index: int, velocity: float) -> None:
        self.ensure()

        with self._lock:
            if 0 <= int(track_i) < len(self.notes) and 0 <= int(note_index) < len(self.notes[int(track_i)]):
                self.notes[int(track_i)][int(note_index)].velocity = float(np.clip(velocity, 0.0, 2.0))
                self._version += 1

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
# Native render path
# ============================================================================

_NATIVE_SYNTH_NAMES = {"synth_keys", "synth", "native_synth"}


def _render_native_synth_track(
    snap: SequenceSnapshot,
    track: Track,
    notes: Tuple[NoteEvent, ...],
    *,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
) -> np.ndarray:
    if not notes:
        return np.zeros((window_n, 2), dtype=np.float32)

    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]
    tbuf = np.zeros((window_n, 2), dtype=np.float32)

    midi_values: List[int] = []
    starts: List[int] = []
    lengths: List[int] = []
    velocities: List[float] = []

    for ev in notes:
        try:
            midi = midi_from_note(ev.pitch[0], ev.pitch[1])
        except Exception:
            continue

        dur_steps = max(1, int(ev.length_steps))
        dur_s = dur_steps * step_s
        dur_n = max(1, int(round(dur_s * int(snap.sr))))

        note_a = int(ev.start_step) * step_n
        note_b = note_a + dur_n

        if note_b <= start_sample or note_a >= start_sample + window_n:
            continue

        midi_values.append(int(midi))
        starts.append(int(note_a - start_sample))
        lengths.append(int(dur_n))
        velocities.append(float(np.clip(ev.velocity, 0.0, 2.0)))

    if not midi_values:
        return tbuf

    midi_arr = np.asarray(midi_values, dtype=np.int32)
    start_arr = np.asarray(starts, dtype=np.int32)
    length_arr = np.asarray(lengths, dtype=np.int32)
    velocity_arr = np.asarray(velocities, dtype=np.float32)

    for inst in instruments:
        name = str(inst.name).strip().lower()

        if name not in _NATIVE_SYNTH_NAMES:
            continue

        p = dict(inst.params or {})

        layer = NATIVE_DSP.render_synth_notes(
            midi_arr,
            start_arr,
            length_arr,
            velocity_arr,
            total_frames=window_n,
            sample_rate=int(snap.sr),
            waveform=waveform_id(p.get("waveform", "sine")),
            master_gain=float(p.get("gain", p.get("master_gain", 0.22))),
            attack_seconds=float(p.get("attack", p.get("attack_seconds", 0.004))),
            release_seconds=float(p.get("release", p.get("release_seconds", 0.035))),
            pan=float(p.get("pan", 0.0)),
        )

        tbuf = NATIVE_DSP.mix_into(tbuf, layer, 1.0)

    return sanitize_audio(tbuf)


def _render_python_instrument_track(
    snap: SequenceSnapshot,
    track: Track,
    notes: Tuple[NoteEvent, ...],
    *,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
) -> np.ndarray:
    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]

    inst_gens: List[Tuple[BaseBlock, Dict[str, Any]]] = []

    for inst in instruments:
        try:
            inst_gens.append((BLOCKS.create(inst.name), dict(inst.params or {})))
        except Exception as exc:
            print(f"[pipeline] missing instrument '{inst.name}': {exc}")

    if not inst_gens:
        return np.zeros((window_n, 2), dtype=np.float32)

    tbuf = np.zeros((window_n, 2), dtype=np.float32)

    for ev in notes:
        dur_steps = max(1, int(ev.length_steps))
        dur_s = dur_steps * step_s
        dur_n = max(1, int(round(dur_s * int(snap.sr))))

        note_a = int(ev.start_step) * step_n
        note_b = note_a + dur_n

        if note_b <= start_sample or note_a >= start_sample + window_n:
            continue

        note, octv = ev.pitch

        try:
            freq = hz_from_note(note, octv)
        except Exception:
            continue

        velocity = float(np.clip(ev.velocity, 0.0, 2.0))

        payload = {
            "freq": float(freq),
            "duration": float(dur_s),
            "sr": int(snap.sr),
            "velocity": velocity,
            "note": str(note),
            "octave": int(octv),
        }

        layer = np.zeros((dur_n, 2), dtype=np.float32)

        for gen, p in inst_gens:
            try:
                raw, _ = gen.execute(payload, params=p)
                y = _as_audio_buffer(raw, int(snap.sr)).data
            except Exception as exc:
                print(f"[pipeline] block '{gen.__class__.__name__}' failed: {exc}")
                continue

            if y.shape[0] < dur_n:
                padded = np.zeros((dur_n, 2), dtype=np.float32)
                padded[: y.shape[0]] = y
                y = padded
            elif y.shape[0] > dur_n:
                y = y[:dur_n]

            layer = NATIVE_DSP.mix_into(layer, y, 1.0)

        a = max(note_a, start_sample)
        b = min(note_b, start_sample + window_n, total_n)

        if b <= a:
            continue

        da = a - start_sample
        db = b - start_sample
        sa = a - note_a
        sb = sa + (db - da)

        tbuf[da:db] += layer[sa:sb]

    return sanitize_audio(tbuf)


def _render_dry_track(
    snap: SequenceSnapshot,
    track_i: int,
    *,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
) -> np.ndarray:
    track = snap.tracks[track_i]
    notes = snap.notes[track_i] if track_i < len(snap.notes) else ()

    instruments = list(track.instruments) or [BlockInstance("synth_keys", {})]
    all_native = all(str(inst.name).strip().lower() in _NATIVE_SYNTH_NAMES for inst in instruments)

    if all_native:
        return _render_native_synth_track(
            snap,
            track,
            notes,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
        )

    return _render_python_instrument_track(
        snap,
        track,
        notes,
        start_sample=start_sample,
        window_n=window_n,
        step_s=step_s,
        step_n=step_n,
        total_n=total_n,
    )


def _get_dry_track_cached(
    snap: SequenceSnapshot,
    track_i: int,
    *,
    start_sample: int,
    window_n: int,
    step_s: float,
    step_n: int,
    total_n: int,
) -> np.ndarray:
    key = _dry_track_key(snap, track_i, start_sample, window_n, step_n)
    hit = _cache_get(key)

    if hit is not None:
        return hit

    dry = _render_dry_track(
        snap,
        track_i,
        start_sample=start_sample,
        window_n=window_n,
        step_s=step_s,
        step_n=step_n,
        total_n=total_n,
    )

    _cache_put(key, dry)
    return dry


def _render_snapshot(
    snap: SequenceSnapshot,
    *,
    start_sample: int = 0,
    max_samples: Optional[int] = None,
    common: Dict[str, Any] | None = None,
) -> AudioBuffer:
    common = common or {}

    step_s = (4.0 * (60.0 / max(1.0e-6, float(snap.bpm)))) / float(max(1, snap.steps_per_bar))
    step_n = max(1, int(round(step_s * int(snap.sr))))

    total_steps = int(snap.steps_per_bar) * int(snap.bars)
    total_n = max(0, step_n * total_steps)

    start_sample = int(max(0, min(total_n, int(start_sample))))

    if max_samples is None:
        window_n = total_n - start_sample
    else:
        window_n = int(max(0, min(total_n - start_sample, int(max_samples))))

    if window_n <= 0:
        return AudioBuffer.silence(0, snap.sr)

    mix = np.zeros((window_n, 2), dtype=np.float32)
    any_solo = any(bool(t.solo) for t in snap.tracks)

    for ti, track in enumerate(snap.tracks):
        if bool(track.mute):
            continue

        if any_solo and not bool(track.solo):
            continue

        dry = _get_dry_track_cached(
            snap,
            ti,
            start_sample=start_sample,
            window_n=window_n,
            step_s=step_s,
            step_n=step_n,
            total_n=total_n,
        )

        tbuf = dry

        fx_chain: List[Tuple[BaseBlock, Dict[str, Any]]] = []

        for bi in track.fx:
            try:
                fx_chain.append((BLOCKS.create(bi.name), dict(bi.params or {})))
            except Exception as exc:
                print(f"[pipeline] missing FX '{bi.name}': {exc}")

        if fx_chain:
            try:
                fx_out, _ = run_chain(
                    AudioBuffer(tbuf, int(snap.sr)),
                    fx_chain,
                    common={
                        **common,
                        "preview": bool(max_samples is not None),
                        "sr": int(snap.sr),
                    },
                )

                tbuf = _as_audio_buffer(fx_out, int(snap.sr)).data

                if tbuf.shape[0] != window_n:
                    fixed = np.zeros((window_n, 2), dtype=np.float32)
                    n = min(window_n, tbuf.shape[0])
                    fixed[:n] = tbuf[:n]
                    tbuf = fixed

            except Exception as exc:
                print(f"[pipeline] track FX failed on '{track.name}': {exc}")

        mix = NATIVE_DSP.mix_into(mix, tbuf, float(track.volume))

    return AudioBuffer(sanitize_audio(mix), int(snap.sr))


# ============================================================================
# Backward-compatible resource ballast helpers
# ============================================================================

class MemoryBallast:
    def __init__(self) -> None:
        self._chunks: list[bytearray] = []
        self._held_bytes = 0
        self._lock = threading.RLock()

    @property
    def held_mb(self) -> float:
        with self._lock:
            return self._held_bytes / (1024.0 * 1024.0)

    def set_target_mb(self, mb: int | float) -> None:
        target = max(0, int(float(mb) * 1024 * 1024))
        chunk_size = 16 * 1024 * 1024

        with self._lock:
            while self._held_bytes < target:
                n = min(chunk_size, target - self._held_bytes)
                self._chunks.append(bytearray(n))
                self._held_bytes += n

            while self._chunks and self._held_bytes > target:
                chunk = self._chunks.pop()
                self._held_bytes -= len(chunk)

    def release(self) -> None:
        self.set_target_mb(0)

    def stats(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()._asdict() if psutil else {}
        return {
            "held_mb": self.held_mb,
            "system": vm,
        }


class CpuBallast:
    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._target_pct = 0
        self._lock = threading.RLock()

    @property
    def target_pct(self) -> int:
        with self._lock:
            return int(self._target_pct)

    def set_target_pct(self, pct: int | float) -> None:
        pct = int(max(0, min(80, float(pct))))

        with self._lock:
            self._target_pct = pct

            if pct <= 0:
                self.stop()
                return

            if self._threads:
                return

            self._stop.clear()
            t = threading.Thread(target=self._run, daemon=True)
            self._threads = [t]
            t.start()

    def _run(self) -> None:
        period = 0.05

        while not self._stop.is_set():
            pct = self.target_pct
            busy = period * (pct / 100.0)
            end = time.perf_counter() + busy
            x = 0.0

            while time.perf_counter() < end and not self._stop.is_set():
                x += math.sin(x + 0.1)

            sleep = max(0.0, period - busy)

            if sleep:
                time.sleep(sleep)

    def stop(self) -> None:
        self._stop.set()

        for t in list(self._threads):
            if t.is_alive():
                t.join(timeout=0.1)

        self._threads.clear()

