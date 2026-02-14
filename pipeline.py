import wave
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os
import psutil
import threading
import time

# ----------------- Audio Buffer -----------------

@dataclass
class AudioBuffer:
    data: np.ndarray  # (n,2) float32
    sr: int


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    return x.astype(np.float32, copy=False)


def write_wav(path: str, buf: AudioBuffer) -> None:
    x = ensure_stereo(buf.data)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(buf.sr)
        wf.writeframes(pcm.tobytes())


# ----------------- Blocks Framework -----------------

@dataclass
class BaseBlock:
    """
    Block contract:

      - execute(payload, params=...) -> (payload_out, meta)

    Optional class attrs:
      - PARAMS: Dict[str, Dict[str, Any]]   (schema for GUI)
      - KIND: str  ('instrument' or 'fx' or 'output')
    """

    # Override in subclasses (class attribute)
    KIND: str = "fx"   # "instrument" | "fx" | "output"
    PARAMS: Dict[str, Dict[str, Any]] = None  # type: ignore[assignment]

    def execute(self, payload: Any, *, params: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError(f"{self.__class__.__name__}.execute() not implemented")
class Registry:
    def __init__(self) -> None:
        self._by_name: Dict[str, type[BaseBlock]] = {}

    def register(self, name: str, cls: type[BaseBlock]) -> None:
        self._by_name[name.strip().lower()] = cls

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def create(self, name: str, **kwargs: Any) -> BaseBlock:
        key = name.strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown block '{name}'. Available: {', '.join(self.names()) or '(none)'}")
        return self._by_name[key](**kwargs)

    def cls(self, name: str) -> type[BaseBlock]:
        key = name.strip().lower()
        if key not in self._by_name:
            raise KeyError(f"Unknown block '{name}'")
        return self._by_name[key]


BLOCKS = Registry()


def run_chain(payload: Any, chain: List[Tuple[BaseBlock, Dict[str, Any]]], *, common: Dict[str, Any] | None = None):
    common = common or {}
    x = payload
    for blk, p in chain:
        merged = {**common, **(p or {})}
        x, _ = blk.execute(x, params=merged)
    return x, {}


# ----------------- Pitch Helpers -----------------

NOTE_TO_SEMI = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}


def hz_from_note(note: str, octave: int) -> float:
    midi = (octave + 1) * 12 + NOTE_TO_SEMI[note]  # C4=60
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


# ----------------- Sequencer Core -----------------

@dataclass
class BlockInstance:
    """A block name + its parameter dict."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    name: str

    instruments: List[BlockInstance] = field(default_factory=list)  # layered
    fx: List[BlockInstance] = field(default_factory=list)           # serial
    volume: float = 1.0





@dataclass
class NoteEvent:
    pitch: Tuple[str, int] = ("C", 4)
    start_step: int = 0
    length_steps: int = 1
    velocity: float = 1.0


@dataclass
class Sequence:
    sr: int = 48000
    bpm: float = 120.0
    steps_per_bar: int = 16
    bars: int = 2
    tracks: List[Track] = field(default_factory=list)

    # notes per track (polyphonic)
    notes: List[List[NoteEvent]] = field(default_factory=list)

    def total_steps(self) -> int:
        return self.steps_per_bar * self.bars

    def step_seconds(self) -> float:
        # 4/4: 1 bar = 4 beats
        return (4.0 * (60.0 / float(self.bpm))) / float(self.steps_per_bar)

    def ensure(self):
        # ensure notes list matches tracks
        while len(self.notes) < len(self.tracks):
            self.notes.append([])
        while len(self.notes) > len(self.tracks):
            self.notes.pop()

    # ---- note operations (polyphonic) ----

    def _clamp_len(self, start_step: int, length_steps: int) -> int:
        length_steps = max(1, int(length_steps))
        max_len = max(1, self.total_steps() - int(start_step))
        return min(length_steps, max_len)

    def add_note(self, track_i: int, start_step: int, pitch: Tuple[str, int], length_steps: int = 1) -> int:
        """
        Add a note event. Does NOT remove other pitches -> chords supported.
        Removes/merges only same-pitch overlaps for sanity.
        Returns index of new event in notes[track_i].
        """
        self.ensure()
        start_step = int(start_step)
        length_steps = self._clamp_len(start_step, length_steps)
        end = start_step + length_steps

        # Remove overlapping events of the same pitch
        new_list: List[NoteEvent] = []
        for ev in self.notes[track_i]:
            if ev.pitch != pitch:
                new_list.append(ev)
                continue
            ev_end = ev.start_step + ev.length_steps
            # overlap?
            if not (ev_end <= start_step or ev.start_step >= end):
                pass # Overlapping events of same pitch are effectively removed by not being added to new_list
            else:
                new_list.append(ev) # Keep non-overlapping notes of same pitch

        new_list.append(NoteEvent(start_step=start_step, length_steps=length_steps, pitch=pitch))
        # keep ordered for predictable behavior
        new_list.sort(key=lambda e: (e.start_step, e.pitch[1], e.pitch[0]))
        self.notes[track_i] = new_list

        # return index of matching newly added event
        for i, ev in enumerate(self.notes[track_i]):
            if ev.start_step == start_step and ev.pitch == pitch and ev.length_steps == length_steps:
                return i
        return len(self.notes[track_i]) - 1 # Should not happen if note was just added

    def find_note_covering(self, track_i: int, step: int, pitch: Tuple[str, int]) -> Optional[int]:
        """
        Find event index whose region covers 'step' AND matches pitch.
        """
        self.ensure()
        step = int(step)
        for i, ev in enumerate(self.notes[track_i]):
            if ev.pitch != pitch:
                continue
            if ev.start_step <= step < (ev.start_step + ev.length_steps):
                return i
        return None

    def find_note_starting_at(self, track_i: int, start_step: int, pitch: Tuple[str, int]) -> Optional[int]:
        self.ensure()
        start_step = int(start_step)
        for i, ev in enumerate(self.notes[track_i]):
            if ev.start_step == start_step and ev.pitch == pitch:
                return i
        return None

    def remove_note(self, track_i: int, note_index: int) -> None:
        self.ensure()
        if 0 <= note_index < len(self.notes[track_i]):
            self.notes[track_i].pop(note_index)

    def set_note_length(self, track_i: int, note_index: int, length_steps: int) -> None:
        self.ensure()
        if not (0 <= note_index < len(self.notes[track_i])):
            return
        ev = self.notes[track_i][note_index]
        ev.length_steps = self._clamp_len(ev.start_step, length_steps)

    def toggle_note_at(self, track_i: int, start_step: int, pitch: Tuple[str, int]) -> None:
        """
        Toggle note starting exactly at start_step with pitch.
        If exists -> remove. Else -> add len=1.
        """
        idx = self.find_note_starting_at(track_i, start_step, pitch)
        if idx is not None:
            self.remove_note(track_i, idx)
        else:
            self.add_note(track_i, start_step, pitch, 1)

    # ---- render ----

    def render(self, *, common: Dict[str, Any] | None = None) -> AudioBuffer:
        """
        Polyphonic render (chords + long notes), optimized:
          - Create instrument/FX blocks ONCE per track per render (no per-note instantiation)
          - Snapshot params ONCE per track per render
          - Keep per-note work to pure DSP + mixing
        """
        common = common or {}
        self.ensure()

        step_s = self.step_seconds()
        step_n = int(round(step_s * self.sr))
        total_n = step_n * self.total_steps()

        mix = np.zeros((total_n, 2), dtype=np.float32)

        for ti, track in enumerate(self.tracks):
            # default instrument if empty, to ensure sound
            if not track.instruments:
                track.instruments = [BlockInstance("synth_keys", {})]

            # ✅ Create generators ONCE per track, snapshot params ONCE
            inst_gens: list[tuple[BaseBlock, Dict[str, Any]]] = []
            for inst in track.instruments:
                inst_gens.append((BLOCKS.create(inst.name), dict(inst.params)))

            fx_chain = [(BLOCKS.create(bi.name), dict(bi.params)) for bi in track.fx]

            tbuf = np.zeros((total_n, 2), dtype=np.float32)

            for ev in self.notes[ti]:
                note, octv = ev.pitch
                freq = hz_from_note(note, octv)

                dur_steps = max(1, int(ev.length_steps))
                dur_s = dur_steps * step_s
                dur_n = int(round(dur_s * self.sr))
                if dur_n <= 0:
                    continue

                # Instrument layering for THIS note
                layer = np.zeros((dur_n, 2), dtype=np.float32)
                payload = {"freq": freq, "duration": dur_s, "sr": self.sr}

                # ✅ No per-note BLOCKS.create(), no per-note dict(inst.params)
                for gen, p in inst_gens:
                    raw, _ = gen.execute(payload, params=p)
                    y = ensure_stereo(raw.data)

                    if y.shape[0] != dur_n:
                        if y.shape[0] < dur_n:
                            padded = np.zeros((dur_n, 2), dtype=np.float32)
                            padded[:y.shape[0]] = y
                            y = padded
                        else:
                            y = y[:dur_n]

                    layer += y

                a = int(ev.start_step) * step_n
                b = min(a + dur_n, total_n)
                if a >= total_n or b <= 0:
                    continue

                tbuf[a:b] += layer[: (b - a)]

            # Apply FX chain to full track (serial)
            out, _ = run_chain(AudioBuffer(tbuf, self.sr), fx_chain, common=common)
            y = ensure_stereo(out.data) * float(track.volume)

            mix += y

        return AudioBuffer(np.tanh(mix).astype(np.float32, copy=False), self.sr)

    def render_from(self, start_sample: int, *, common: Dict[str, Any] | None = None) -> AudioBuffer:
        """
        Fast preview render, optimized:
          - Create instrument/FX blocks ONCE per track per render_from()
          - Snapshot params ONCE per track
          - Only render notes that overlap [start_sample..end)
        """
        common = common or {}
        self.ensure()

        step_s = self.step_seconds()
        step_n = int(round(step_s * self.sr))
        total_n = step_n * self.total_steps()

        start_sample = int(max(0, min(total_n, int(start_sample))))
        tail_n = total_n - start_sample

        if tail_n <= 0:
            return AudioBuffer(np.zeros((0, 2), dtype=np.float32), self.sr)

        mix = np.zeros((tail_n, 2), dtype=np.float32)

        def place_tail(dst_tail: np.ndarray, src_note: np.ndarray, note_a: int, note_b: int):
            a = max(note_a, start_sample)
            b = min(note_b, total_n)
            if b <= a:
                return

            da = a - start_sample
            db = b - start_sample

            sa = a - note_a
            sb = sa + (db - da)

            dst_tail[da:db] += src_note[sa:sb]

        for ti, track in enumerate(self.tracks):
            if not track.instruments:
                track.instruments = [BlockInstance("synth_keys", {})]

            # ✅ Create generators ONCE per track, snapshot params ONCE
            inst_gens: list[tuple[BaseBlock, Dict[str, Any]]] = []
            for inst in track.instruments:
                inst_gens.append((BLOCKS.create(inst.name), dict(inst.params)))

            fx_chain = [(BLOCKS.create(bi.name), dict(bi.params)) for bi in track.fx]

            tbuf = np.zeros((tail_n, 2), dtype=np.float32)

            for ev in self.notes[ti]:
                note, octv = ev.pitch
                freq = hz_from_note(note, octv)

                dur_steps = max(1, int(ev.length_steps))
                dur_s = dur_steps * step_s
                dur_n = int(round(dur_s * self.sr))
                if dur_n <= 0:
                    continue

                note_a = int(ev.start_step) * step_n
                note_b = note_a + dur_n

                if note_b <= start_sample:
                    continue

                layer = np.zeros((dur_n, 2), dtype=np.float32)
                payload = {"freq": freq, "duration": dur_s, "sr": self.sr}

                # ✅ No per-note BLOCKS.create(), no per-note dict(inst.params)
                for gen, p in inst_gens:
                    raw, _ = gen.execute(payload, params=p)
                    y = ensure_stereo(raw.data)

                    if y.shape[0] != dur_n:
                        if y.shape[0] < dur_n:
                            pad = np.zeros((dur_n, 2), dtype=np.float32)
                            pad[:y.shape[0]] = y
                            y = pad
                        else:
                            y = y[:dur_n]

                    layer += y

                place_tail(tbuf, layer, note_a, note_b)

            out, _ = run_chain(AudioBuffer(tbuf, self.sr), fx_chain, common=common)
            y = ensure_stereo(out.data) * float(track.volume)

            mix += y

        return AudioBuffer(np.tanh(mix).astype(np.float32, copy=False), self.sr)


class MemoryBallast:
    """
    Holds RAM on purpose so your process can keep hot working sets/caches.
    NOTE: psutil only measures; the allocation below is what increases memory.
    """
    def __init__(self) -> None:
        self._chunks: list[bytearray] = []
        self._held_bytes: int = 0

    def held_mb(self) -> int:
        return int(self._held_bytes // (1024 * 1024))

    def rss_mb(self) -> int:
        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss // (1024 * 1024))

    def avail_mb(self) -> int:
        return int(psutil.virtual_memory().available // (1024 * 1024))

    def set_target_mb(self, target_mb: int, *, touch: bool = True, chunk_mb: int = 32) -> None:
        target_mb = max(0, int(target_mb))
        target_bytes = target_mb * 1024 * 1024
        chunk_bytes = max(1, int(chunk_mb)) * 1024 * 1024

        # grow
        while self._held_bytes < target_bytes:
            need = target_bytes - self._held_bytes
            alloc = min(chunk_bytes, need)
            b = bytearray(alloc)
            if touch:
                # commit pages (Windows won't necessarily commit untouched pages)
                page = 4096
                for i in range(0, len(b), page):
                    b[i] = 1
            self._chunks.append(b)
            self._held_bytes += alloc

        # shrink
        while self._held_bytes > target_bytes and self._chunks:
            b = self._chunks.pop()
            self._held_bytes -= len(b)

    def clear(self) -> None:
        self._chunks.clear()
        self._held_bytes = 0

class CpuBallast:
    """
    Burns CPU by a controllable duty cycle.
    Target is expressed as % of TOTAL CPU capacity (0..100),
    not per-core.

    Example: on an 8-core CPU, target_pct=25 means ~2 cores loaded.
    """

    def __init__(self) -> None:
        self._target_pct = 0
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()

        self._proc = psutil.Process(os.getpid())
        # Prime cpu_percent so the first reading isn't always 0.0
        try:
            self._proc.cpu_percent(None)
        except Exception:
            pass

    def cpu_count(self) -> int:
        try:
            return int(psutil.cpu_count(logical=True) or 1)
        except Exception:
            return 1

    def set_target_pct(self, target_pct: int, *, workers: int | None = None) -> None:
        target_pct = int(max(0, min(100, int(target_pct))))
        with self._lock:
            self._target_pct = target_pct

        if target_pct <= 0:
            self.stop()
            return

        # start workers if not running
        if not self._threads:
            ncpu = self.cpu_count()
            nworkers = int(workers) if workers is not None else ncpu
            nworkers = max(1, min(nworkers, ncpu))

            self._stop.clear()
            self._threads = []
            for _ in range(nworkers):
                t = threading.Thread(target=self._worker_loop, daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._threads.clear()

    def _worker_loop(self) -> None:
        # small period -> smoother load
        period = 0.10  # seconds

        # Do NOT allocate in the loop; keep it tiny.
        x = 0.0
        while not self._stop.is_set():
            with self._lock:
                tgt = int(self._target_pct)

            if tgt <= 0:
                time.sleep(0.10)
                continue

            nworkers = max(1, len(self._threads) or 1)

            # convert "total CPU %" into per-worker share (still total-normalized)
            # total capacity is 100%. Each worker approximates 100/nworkers of capacity
            per_worker_pct = float(tgt) / float(nworkers)
            on = period * (per_worker_pct / 100.0)
            on = max(0.0, min(period, on))
            off = period - on

            # busy part
            t0 = time.perf_counter()
            while (time.perf_counter() - t0) < on and not self._stop.is_set():
                # tiny meaningless math to avoid optimizer shortcuts
                x = (x * 1.0000001) + 0.0000001
                if x > 1e9:
                    x = 0.0

            # sleep part (yields CPU)
            if off > 0:
                time.sleep(off)

    def cpu_pct_process_raw(self) -> float:
        """
        Raw process cpu_percent. Can exceed 100 on multi-core.
        """
        try:
            return float(self._proc.cpu_percent(None))
        except Exception:
            return 0.0

    def cpu_pct_process_norm(self) -> float:
        """
        Normalized to 0..100 for the whole machine.
        """
        raw = self.cpu_pct_process_raw()
        n = float(self.cpu_count())
        if n <= 0:
            return raw
        return max(0.0, min(100.0, raw / n))