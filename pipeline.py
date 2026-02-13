import wave
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
    """
    One note in the piano roll.
    start_step: step index where note begins
    length_steps: duration in steps (>=1)
    pitch: (note, octave) like ("C#", 4)
    """
    start_step: int
    length_steps: int
    pitch: Tuple[str, int]


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
        Polyphonic render (chords + long notes):

        For each track:
          - allocate a full timeline buffer
          - for each note event:
              - for each instrument: generate note buffer for (length_steps * step_seconds)
              - sum instruments into a note buffer
              - place note buffer into track timeline at start sample
          - apply FX chain to the FULL track buffer (serial)
        Then sum tracks.
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
                for inst in track.instruments:
                    # Create a new instance of the block for each render pass
                    # This ensures that internal state (if any) is fresh,
                    # and parameters are read from the BlockInstance's dict.
                    gen = BLOCKS.create(inst.name)
                    payload = {"freq": freq, "duration": dur_s, "sr": self.sr}
                    raw, _ = gen.execute(payload, params=dict(inst.params)) # Pass a copy of params
                    y = ensure_stereo(raw.data)
                    if y.shape[0] != dur_n:
                        # Pad or truncate if the block generates a different length,
                        # though ideally instrument blocks should match requested duration.
                        if y.shape[0] < dur_n:
                            padded_y = np.zeros((dur_n, 2), dtype=np.float32)
                            padded_y[:y.shape[0]] = y
                            y = padded_y
                        else:
                            y = y[:dur_n]
                    layer += y

                a = int(ev.start_step) * step_n
                b = min(a + dur_n, total_n)
                if a >= total_n or b <= 0:
                    continue

                tbuf[a:b] += layer[: (b - a)]

            # Apply FX chain to full track
            fx_chain = [(BLOCKS.create(bi.name), dict(bi.params)) for bi in track.fx]
            out, _ = run_chain(AudioBuffer(tbuf, self.sr), fx_chain, common=common)
            y = ensure_stereo(out.data) * float(track.volume)

            mix += y

        # Apply a final soft-clipper to the master mix to prevent harsh clipping
        # and give a more consistent output level.
        return AudioBuffer(np.tanh(mix).astype(np.float32, copy=False), self.sr)

    def render_from(self, start_sample: int, *, common: Dict[str, Any] | None = None) -> AudioBuffer:
        """
        Fast preview render:
        Renders ONLY the audio region [start_sample : end] and returns that tail buffer.
        Export should still use render() for full accuracy/stateful FX correctness.
        """
        common = common or {}
        self.ensure()

        step_s = self.step_seconds()
        step_n = int(round(step_s * self.sr))
        total_n = step_n * self.total_steps()

        start_sample = int(max(0, min(total_n, int(start_sample))))
        tail_n = total_n - start_sample

        # If start is at end, return empty tail
        if tail_n <= 0:
            return AudioBuffer(np.zeros((0, 2), dtype=np.float32), self.sr)

        mix = np.zeros((tail_n, 2), dtype=np.float32)

        def place_tail(dst_tail: np.ndarray, src_note: np.ndarray, note_a: int, note_b: int):
            """
            Place overlap of src_note [0..dur) into dst_tail [0..tail_n),
            but only for overlap with [start_sample..total_n).
            """
            a = max(note_a, start_sample)
            b = min(note_b, total_n)
            if b <= a:
                return

            # dst coordinates (tail starts at start_sample)
            da = a - start_sample
            db = b - start_sample

            # src coordinates (note starts at note_a)
            sa = a - note_a
            sb = sa + (db - da)

            dst_tail[da:db] += src_note[sa:sb]

        for ti, track in enumerate(self.tracks):
            if not track.instruments:
                track.instruments = [BlockInstance("synth_keys", {})]

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

                # If note ends before tail region starts, skip
                if note_b <= start_sample:
                    continue

                # Render this note (offline) then place only the overlapping part
                layer = np.zeros((dur_n, 2), dtype=np.float32)

                for inst in track.instruments:
                    gen = BLOCKS.create(inst.name)
                    payload = {"freq": freq, "duration": dur_s, "sr": self.sr}
                    raw, _ = gen.execute(payload, params=dict(inst.params))
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

            # Apply FX to the tail (fast preview)
            fx_chain = [(BLOCKS.create(bi.name), dict(bi.params)) for bi in track.fx]
            out, _ = run_chain(AudioBuffer(tbuf, self.sr), fx_chain, common=common)
            y = ensure_stereo(out.data) * float(track.volume)

            mix += y

        return AudioBuffer(np.tanh(mix).astype(np.float32, copy=False), self.sr)