from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import sounddevice as sd

from pipeline import AudioBuffer, Sequence, normalize_for_playback, write_wav

ErrorCallback = Callable[[str, str], None]
PositionCallback = Callable[[int], None]


class AudioEngine:
    """
    Fast live-edit playback engine.

    Important:
    - Audio callback never renders.
    - Param changes request a tiny preview render first.
    - Full render happens after the user stops moving controls.
    - Stale renders are skipped instead of swapped in.
    """

    def __init__(
        self,
        seq: Sequence,
        *,
        blocksize: int = 512,
        latency: str | float = "low",
        preview_seconds: float = 8.0,
        preview_debounce_ms: int = 35,
        full_debounce_ms: int = 650,
        on_error: Optional[ErrorCallback] = None,
        on_position: Optional[PositionCallback] = None,
    ) -> None:
        self.seq = seq
        self.blocksize = int(max(64, blocksize))
        self.latency = latency
        self.preview_seconds = float(max(1.0, preview_seconds))
        self.preview_debounce_ms = int(max(1, preview_debounce_ms))
        self.full_debounce_ms = int(max(50, full_debounce_ms))
        self.on_error = on_error
        self.on_position = on_position

        self._state_lock = threading.RLock()
        self._cond = threading.Condition(threading.RLock())

        self._stream: Optional[sd.OutputStream] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_worker = False

        self._buffer = AudioBuffer.silence(0, int(self.seq.sr))
        self._buffer_start_sample = 0
        self._play_pos = 0
        self._start_sample = 0
        self._loop = False
        self._playing = False

        self._request_id = 0
        self._preview_due: Optional[float] = None
        self._full_due: Optional[float] = None

        self._last_render_time = 0.0
        self._last_render_kind = "none"
        self._prepared_version = -1

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        with self._state_lock:
            return bool(self._playing)

    @property
    def position_samples(self) -> int:
        with self._state_lock:
            return int(self._play_pos)

    @property
    def last_render_time(self) -> float:
        with self._state_lock:
            return float(self._last_render_time)

    @property
    def last_render_kind(self) -> str:
        with self._state_lock:
            return str(self._last_render_kind)

    def set_loop(self, enabled: bool) -> None:
        with self._state_lock:
            self._loop = bool(enabled)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def start(self, *, start_sample: int = 0, loop: bool = False) -> None:
        with self._state_lock:
            self._playing = True
            self._loop = bool(loop)
            self._start_sample = int(max(0, start_sample))
            self._play_pos = int(max(0, start_sample))

        self._ensure_worker()

        # Preview first so play starts fast, then full render when idle.
        self.request_preview_rerender(immediate=True, full_later=True)

    def pause(self) -> None:
        with self._state_lock:
            self._playing = False

        self._close_stream()

    def stop(self) -> None:
        with self._state_lock:
            self._playing = False
            self._play_pos = int(self._start_sample)

        self._close_stream()

    def shutdown(self) -> None:
        with self._cond:
            self._stop_worker = True
            self._cond.notify_all()

        self._close_stream()

    def seek(self, sample: int) -> None:
        with self._state_lock:
            self._play_pos = int(max(0, sample))

        self.request_preview_rerender(immediate=True, full_later=True)

    # ------------------------------------------------------------------
    # Live edit requests
    # ------------------------------------------------------------------

    def request_preview_rerender(self, *, immediate: bool = False, full_later: bool = True) -> None:
        now = time.monotonic()
        preview_due = now if immediate else now + (self.preview_debounce_ms / 1000.0)
        full_due = now + (self.full_debounce_ms / 1000.0)

        with self._cond:
            self._request_id += 1

            if self._preview_due is None:
                self._preview_due = preview_due
            else:
                self._preview_due = min(self._preview_due, preview_due)

            if full_later:
                self._full_due = full_due

            self._cond.notify_all()

    def request_full_rerender(self, *, immediate: bool = False) -> None:
        now = time.monotonic()
        due = now if immediate else now + (self.full_debounce_ms / 1000.0)

        with self._cond:
            self._request_id += 1
            self._full_due = due
            self._cond.notify_all()

    # Backward-compatible method name from the older patch.
    def request_rerender(self, *, from_current_position: bool = True) -> None:
        self.request_preview_rerender(immediate=False, full_later=True)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_wav(self, path: str) -> None:
        buf = normalize_for_playback(self.seq.render())
        write_wav(path, buf)

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._cond:
            if self._worker is not None and self._worker.is_alive():
                return

            self._stop_worker = False
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="MelodyAudioEngine",
                daemon=True,
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            try:
                kind = self._wait_for_due_job()

                if kind is None:
                    return

                self._perform_render_job(kind)

            except Exception:
                import traceback
                self._emit_error("Audio engine failed", traceback.format_exc())

    def _wait_for_due_job(self) -> Optional[str]:
        with self._cond:
            while True:
                if self._stop_worker:
                    return None

                now = time.monotonic()
                due_items: list[tuple[float, str]] = []

                if self._preview_due is not None:
                    due_items.append((self._preview_due, "preview"))

                if self._full_due is not None:
                    due_items.append((self._full_due, "full"))

                if not due_items:
                    self._cond.wait(timeout=0.10)
                    continue

                due_at, kind = min(due_items, key=lambda x: x[0])

                if due_at > now:
                    self._cond.wait(timeout=max(0.001, due_at - now))
                    continue

                if kind == "preview":
                    self._preview_due = None
                else:
                    self._full_due = None

                return kind

    def _perform_render_job(self, kind: str) -> None:
        with self._cond:
            request_id = int(self._request_id)

        with self._state_lock:
            playing = bool(self._playing)
            pos = int(self._play_pos)

        if not playing:
            return

        t0 = time.perf_counter()

        if kind == "preview":
            start = max(0, pos)
            max_samples = int(round(self.preview_seconds * int(self.seq.sr)))
            buf = self.seq.render_window(start, max_samples)
            buffer_start = start
        else:
            buf = self.seq.render()
            buffer_start = 0
            kind = "full"

        buf = normalize_for_playback(buf, peak=0.98, only_if_over=True)
        elapsed = time.perf_counter() - t0

        with self._cond:
            if request_id != self._request_id and kind == "preview":
                return

        with self._state_lock:
            old_pos = int(self._play_pos)
            self._buffer = buf
            self._buffer_start_sample = int(buffer_start)
            self._play_pos = old_pos
            self._prepared_version = int(self.seq.version)
            self._last_render_time = float(elapsed)
            self._last_render_kind = str(kind)

        self._open_stream_if_needed()
    # ------------------------------------------------------------------
    # sounddevice
    # ------------------------------------------------------------------

    def _open_stream_if_needed(self) -> None:
        with self._state_lock:
            if self._stream is not None:
                return

        stream = sd.OutputStream(
            samplerate=int(self.seq.sr),
            channels=2,
            dtype="float32",
            blocksize=int(self.blocksize),
            latency=self.latency,
            callback=self._callback,
        )

        stream.start()

        with self._state_lock:
            self._stream = stream

    def _close_stream(self) -> None:
        with self._state_lock:
            stream = self._stream
            self._stream = None

        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

    def _callback(self, outdata, frames, time_info, status) -> None:  # noqa: ANN001
        outdata.fill(0.0)

        with self._state_lock:
            playing = bool(self._playing)
            loop = bool(self._loop)
            pos = int(self._play_pos)
            start_sample = int(self._start_sample)
            buffer_start = int(self._buffer_start_sample)
            data = self._buffer.data
            total_len = int(data.shape[0])
            is_full_buffer = buffer_start == 0

        if not playing or total_len <= 0:
            return

        written = 0
        local_pos = pos - buffer_start

        if local_pos < 0:
            local_pos = 0
            pos = buffer_start

        while written < frames:
            if local_pos >= total_len:
                if loop and is_full_buffer:
                    local_pos = max(0, start_sample)
                    pos = local_pos
                else:
                    # We reached the end of a short preview buffer.
                    # Ask for more audio, but never block the callback.
                    self.request_full_rerender(immediate=True)
                    break

            available = max(0, total_len - local_pos)

            if available <= 0:
                break

            take = min(int(frames - written), int(available))
            outdata[written:written + take, :] = data[local_pos:local_pos + take, :]
            written += take
            local_pos += take
            pos += take

        with self._state_lock:
            self._play_pos = int(pos)

        if self.on_position is not None:
            try:
                self.on_position(int(pos))
            except Exception:
                pass

    def _emit_error(self, title: str, text: str) -> None:
        if self.on_error is not None:
            try:
                self.on_error(title, text)
                return
            except Exception:
                pass

        print(f"[{title}]\n{text}")