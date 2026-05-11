# gui.py
from __future__ import annotations

import math
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

try:
    import psutil
except Exception:
    psutil = None  # type: ignore

from PyQt6.QtCore import Qt, QEvent, QObject, QRectF, QTimer
from PyQt6.QtGui import QBrush, QColor, QPen, QPainter, QAction, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QListWidget, QFileDialog, QGroupBox,
    QFormLayout, QDoubleSpinBox, QSpinBox, QCheckBox, QMessageBox,
    QScrollArea, QSplitter, QToolButton,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem,
    QGraphicsTextItem, QSlider, QLineEdit,
)

from audio_engine import AudioEngine

from pipeline import (
    Sequence,
    Track,
    BlockInstance,
    BLOCKS,
    MemoryBallast,
    CpuBallast,
)

import sounds  # registers blocks

try:
    import realism  # optional extra blocks / polish
except Exception:
    realism = None  # type: ignore

try:
    import melody_humanize  # optional extra blocks / polish
except Exception:
    melody_humanize = None  # type: ignore


# ============================================================================
# Block helper functions
# ============================================================================

def block_params_schema(block_name: str) -> Dict[str, Dict[str, Any]]:
    cls = BLOCKS.cls(block_name)
    return getattr(cls, "PARAMS", {}) or {}


def block_kind(block_name: str) -> str:
    cls = BLOCKS.cls(block_name)
    return str(getattr(cls, "KIND", "fx"))


def default_params(block_name: str) -> Dict[str, Any]:
    schema = block_params_schema(block_name)
    return {k: v.get("default") for k, v in schema.items()}


# ============================================================================
# Pitch / scale helpers
# ============================================================================

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_oct(midi: int) -> Tuple[str, int]:
    midi = int(midi)
    note = NOTES[midi % 12]
    octave = (midi // 12) - 1
    return note, octave


def note_oct_to_midi(note: str, octave: int) -> int:
    semi = NOTES.index(note)
    return (int(octave) + 1) * 12 + semi


SCALE_INTERVALS: Dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "natural_minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}


def _note_to_semi(note: str) -> int:
    return NOTES.index(note)


def _scale_set(root_note: str, scale_name: str) -> set[int]:
    root = _note_to_semi(root_note)
    intervals = SCALE_INTERVALS.get(scale_name, [])
    return {(root + i) % 12 for i in intervals}


# ============================================================================
# Slider helpers
# ============================================================================

def _slider_steps_for_spec(spec: Dict[str, Any], *, is_int: bool) -> Tuple[float, float, float, int]:
    mn = float(spec.get("min", 0.0))
    mx = float(spec.get("max", 1.0))

    if mx <= mn:
        mx = mn + 1.0

    if is_int:
        step = float(spec.get("step", 1.0))
        step = max(1.0, round(step))
        steps_int = int(round((mx - mn) / step))
        steps_int = max(1, steps_int)
        return mn, mx, step, steps_int

    step = float(spec.get("step", 0.01))

    if step <= 0:
        step = 0.01

    steps_int = int(round((mx - mn) / step))
    steps_int = max(1, min(20000, steps_int))
    step = (mx - mn) / float(steps_int)

    return mn, mx, step, steps_int


def _value_to_slider(v: float, mn: float, step: float, steps_int: int) -> int:
    i = int(round((float(v) - mn) / step))
    return int(max(0, min(steps_int, i)))


def _slider_to_value(i: int, mn: float, step: float) -> float:
    return float(mn + float(i) * step)


# ============================================================================
# Thread-safe callable event
# ============================================================================

class _CallableEvent(QEvent):
    TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, fn):
        super().__init__(self.TYPE)
        self.fn = fn


class _EventCatcher(QObject):
    def eventFilter(self, obj, e):
        if e.type() == _CallableEvent.TYPE:
            try:
                e.fn()
            except Exception:
                pass
            return True

        return False


# ============================================================================
# Graphics helpers
# ============================================================================

@dataclass
class PianoLayout:
    left_label_w: float = 58.0
    top_header_h: float = 20.0
    cell_w: float = 26.0
    row_h: float = 18.0


def _safe_text_font(size_pt: int) -> QFont:
    size_pt = max(7, int(size_pt))
    f = QFont()
    f.setPointSize(size_pt)
    return f


class NoteItem(QGraphicsRectItem):
    def __init__(self, r: QRectF, radius: float = 3.5):
        super().__init__(r)
        self.radius = radius
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setZValue(20)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(Qt.PenStyle.NoPen))
        painter.setBrush(self.brush())
        painter.drawRoundedRect(self.rect(), self.radius, self.radius)


# ============================================================================
# Piano roll
# ============================================================================

class PianoRollView(QGraphicsView):
    def __init__(self, owner: "MelodyGUI"):
        super().__init__()
        self.owner = owner
        self.layout = PianoLayout()

        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing |
            QPainter.RenderHint.TextAntialiasing |
            QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(18, 18, 18)))

        self._grid_items: list[Any] = []
        self._ghost_rects: list[QGraphicsRectItem] = []
        self._note_items: list[NoteItem] = []
        self._note_map: list[int] = []
        self._label_items: list[Any] = []
        self._note_item_by_index: dict[int, NoteItem] = {}

        self.playhead_line = QGraphicsLineItem()
        self.playhead_line.setZValue(100)
        self.playhead_line.setPen(QPen(QColor(255, 90, 90), 2.0))
        self.scene().addItem(self.playhead_line)

        self._dragging = False
        self._drag_start_step = 0
        self._drag_row = 0
        self._drag_note_index: Optional[int] = None
        self._drag_was_existing = False
        self._drag_moved = False
        self._drag_last_len = 1

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _cols(self) -> int:
        return int(self.owner.seq.total_steps())

    def _rows(self) -> int:
        return int(self.owner._piano_rows)

    def _grid_origin_x(self) -> float:
        return self.layout.left_label_w

    def _grid_origin_y(self) -> float:
        return self.layout.top_header_h

    def _scene_w(self) -> float:
        return self.layout.left_label_w + float(self._cols()) * self.layout.cell_w

    def _scene_h(self) -> float:
        return self.layout.top_header_h + float(self._rows()) * self.layout.row_h

    def _step_from_scene_x(self, x: float) -> int:
        gx = x - self._grid_origin_x()
        return int(max(0, min(self._cols() - 1, int(gx // self.layout.cell_w))))

    def _row_from_scene_y(self, y: float) -> int:
        gy = y - self._grid_origin_y()
        return int(max(0, min(self._rows() - 1, int(gy // self.layout.row_h))))

    def _scene_x_from_step(self, step: float) -> float:
        return self._grid_origin_x() + float(step) * self.layout.cell_w

    def _scene_y_from_row(self, row: int) -> float:
        return self._grid_origin_y() + float(row) * self.layout.row_h

    def _pitch_from_row(self, row: int) -> Tuple[str, int]:
        midi = self.owner.piano_high_midi - int(row)
        return midi_to_note_oct(midi)

    def _row_from_pitch(self, pitch: Tuple[str, int]) -> Optional[int]:
        note, octv = pitch
        midi = note_oct_to_midi(note, octv)

        if not (self.owner.piano_low_midi <= midi <= self.owner.piano_high_midi):
            return None

        return self.owner.piano_high_midi - midi

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def rebuild_grid(self):
        sc = self.scene()

        for it in self._grid_items:
            sc.removeItem(it)
        for it in self._ghost_rects:
            sc.removeItem(it)
        for it in self._label_items:
            sc.removeItem(it)

        self._grid_items.clear()
        self._ghost_rects.clear()
        self._label_items.clear()

        cols = self._cols()
        rows = self._rows()

        sc.setSceneRect(0, 0, self._scene_w(), self._scene_h())

        left_bg = QGraphicsRectItem(0, 0, self.layout.left_label_w, self._scene_h())
        left_bg.setPen(QPen(Qt.PenStyle.NoPen))
        left_bg.setBrush(QBrush(QColor(12, 12, 12)))
        left_bg.setZValue(0)
        sc.addItem(left_bg)
        self._grid_items.append(left_bg)

        top_bg = QGraphicsRectItem(0, 0, self._scene_w(), self.layout.top_header_h)
        top_bg.setPen(QPen(Qt.PenStyle.NoPen))
        top_bg.setBrush(QBrush(QColor(14, 14, 14)))
        top_bg.setZValue(0)
        sc.addItem(top_bg)
        self._grid_items.append(top_bg)

        minor = QPen(QColor(35, 35, 35), 1.0)
        major = QPen(QColor(55, 55, 55), 1.0)
        hpen = QPen(QColor(30, 30, 30), 1.0)

        bar_every = int(getattr(self.owner.seq, "steps_per_bar", 16) or 16)

        for c in range(cols + 1):
            x = self._grid_origin_x() + c * self.layout.cell_w
            pen = major if (c % bar_every == 0) else minor
            li = QGraphicsLineItem(x, self._grid_origin_y(), x, self._scene_h())
            li.setPen(pen)
            li.setZValue(1)
            sc.addItem(li)
            self._grid_items.append(li)

        for r in range(rows + 1):
            y = self._grid_origin_y() + r * self.layout.row_h
            li = QGraphicsLineItem(self._grid_origin_x(), y, self._scene_w(), y)
            li.setPen(hpen)
            li.setZValue(1)
            sc.addItem(li)
            self._grid_items.append(li)

        label_font = _safe_text_font(8)

        for r in range(rows):
            pitch = self._pitch_from_row(r)
            name = f"{pitch[0]}{pitch[1]}"
            y = self._scene_y_from_row(r)
            is_black = "#" in pitch[0]

            key_rect = QGraphicsRectItem(0, y, self.layout.left_label_w, self.layout.row_h)
            key_rect.setPen(QPen(QColor(30, 30, 30), 1.0))
            key_rect.setBrush(QBrush(QColor(20, 20, 20) if is_black else QColor(26, 26, 26)))
            key_rect.setZValue(2)
            sc.addItem(key_rect)
            self._grid_items.append(key_rect)

            t = QGraphicsTextItem(name)
            t.setDefaultTextColor(QColor(210, 210, 210))
            t.setFont(label_font)
            t.setPos(6, y + 1)
            t.setZValue(3)
            sc.addItem(t)
            self._label_items.append(t)

        header_font = _safe_text_font(8)

        for c in range(0, cols, bar_every):
            x = self._grid_origin_x() + c * self.layout.cell_w
            bar_idx = (c // bar_every) + 1
            t = QGraphicsTextItem(f"Bar {bar_idx}")
            t.setDefaultTextColor(QColor(160, 160, 160))
            t.setFont(header_font)
            t.setPos(x + 4, 2)
            t.setZValue(3)
            sc.addItem(t)
            self._label_items.append(t)

        ghost_pcs: Optional[set[int]] = None

        if self.owner._ghost_root and self.owner._ghost_scale:
            ghost_pcs = _scale_set(self.owner._ghost_root, self.owner._ghost_scale)

        if ghost_pcs is not None:
            for r in range(rows):
                midi = self.owner.piano_high_midi - r
                pc = midi % 12

                if pc in ghost_pcs:
                    rr = QGraphicsRectItem(
                        self._grid_origin_x(),
                        self._scene_y_from_row(r),
                        float(cols) * self.layout.cell_w,
                        self.layout.row_h,
                    )
                    rr.setPen(QPen(Qt.PenStyle.NoPen))
                    rr.setBrush(QBrush(QColor(255, 255, 255, 10)))
                    rr.setZValue(2)
                    sc.addItem(rr)
                    self._ghost_rects.append(rr)

        self.set_playhead_step(self.owner._start_step)

    def rebuild_notes(self):
        sc = self.scene()

        for it in self._note_items:
            sc.removeItem(it)

        self._note_items.clear()
        self._note_map.clear()
        self._note_item_by_index.clear()

        ti = self.owner.current_track_index()

        if ti < 0:
            ti = 0

        self.owner.seq.ensure()

        if ti >= len(self.owner.seq.notes):
            return

        for idx, ev in enumerate(self.owner.seq.notes[ti]):
            row = self._row_from_pitch(ev.pitch)

            if row is None:
                continue

            x = self._scene_x_from_step(float(ev.start_step))
            y = self._scene_y_from_row(row)
            w = float(max(1, ev.length_steps)) * self.layout.cell_w
            h = self.layout.row_h

            ni = NoteItem(QRectF(x + 1, y + 1, w - 2, h - 2))
            ni.setBrush(QBrush(QColor(60, 150, 255)))
            ni.setOpacity(0.95)
            sc.addItem(ni)

            self._note_items.append(ni)
            self._note_map.append(idx)
            self._note_item_by_index[idx] = ni

        self.set_playhead_step(self.owner._start_step)

    def set_playhead_step(self, step_float: float):
        x = self._scene_x_from_step(float(step_float))
        x = max(self._grid_origin_x(), min(self._scene_w(), x))
        self.playhead_line.setLine(x, self._grid_origin_y(), x, self._scene_h())

    def update_note_item(self, note_idx: int):
        ti = self.owner.current_track_index()

        if ti < 0:
            return

        item = self._note_item_by_index.get(note_idx)

        if item is None:
            return

        if ti >= len(self.owner.seq.notes):
            return

        if not (0 <= note_idx < len(self.owner.seq.notes[ti])):
            return

        ev = self.owner.seq.notes[ti][note_idx]
        row = self._row_from_pitch(ev.pitch)

        if row is None:
            return

        x = self._scene_x_from_step(float(ev.start_step))
        y = self._scene_y_from_row(row)
        w = float(max(1, ev.length_steps)) * self.layout.cell_w
        h = self.layout.row_h

        item.setRect(QRectF(x + 1, y + 1, w - 2, h - 2))

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = ev.angleDelta().y()

            if delta > 0:
                self.scale(1.12, 1.0)
            else:
                self.scale(1 / 1.12, 1.0)

            ev.accept()
            return

        super().wheelEvent(ev)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            sp = self.mapToScene(ev.pos())

            if sp.x() < self._grid_origin_x() or sp.y() < self._grid_origin_y():
                ev.accept()
                return

            step = self._step_from_scene_x(sp.x())
            row = self._row_from_scene_y(sp.y())
            pitch = self._pitch_from_row(row)

            if ev.modifiers() & Qt.KeyboardModifier.AltModifier:
                self.owner._set_start_step(step)
                ev.accept()
                return

            ti = self.owner.current_track_index()

            if ti < 0:
                ti = 0

            idx = self.owner.seq.find_note_covering(ti, step, pitch)

            if idx is not None:
                ev0 = self.owner.seq.notes[ti][idx]
                self._dragging = True
                self._drag_start_step = int(ev0.start_step)
                self._drag_row = row
                self._drag_note_index = idx
                self._drag_was_existing = True
                self._drag_moved = False
                self._drag_last_len = int(ev0.length_steps)
                ev.accept()
                return

            new_idx = self.owner.seq.add_note(ti, step, pitch, 1)
            self._dragging = True
            self._drag_start_step = int(step)
            self._drag_row = row
            self._drag_note_index = new_idx
            self._drag_was_existing = False
            self._drag_moved = False
            self._drag_last_len = 1

            self.owner.rebuild_roll(grid=False, notes=True)
            ev.accept()
            return

        if ev.button() == Qt.MouseButton.RightButton:
            sp = self.mapToScene(ev.pos())

            if sp.x() < self._grid_origin_x() or sp.y() < self._grid_origin_y():
                ev.accept()
                return

            step = self._step_from_scene_x(sp.x())
            row = self._row_from_scene_y(sp.y())
            pitch = self._pitch_from_row(row)

            ti = self.owner.current_track_index()

            if ti < 0:
                ti = 0

            idx = self.owner.seq.find_note_covering(ti, step, pitch)

            if idx is not None:
                self.owner.seq.remove_note(ti, idx)
                self.owner.rebuild_roll(grid=False, notes=True)
                self.owner._mark_dirty_and_restart_from_playhead()

            ev.accept()
            return

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging and self._drag_note_index is not None:
            sp = self.mapToScene(ev.pos())

            if sp.x() < self._grid_origin_x() or sp.y() < self._grid_origin_y():
                ev.accept()
                return

            step = self._step_from_scene_x(sp.x())
            start = int(self._drag_start_step)
            end = max(start, int(step))
            length = max(1, min((end - start + 1), self.owner.seq.total_steps() - start))

            if length != self._drag_last_len:
                self._drag_last_len = length
                self._drag_moved = True

                ti = self.owner.current_track_index()

                if ti < 0:
                    ti = 0

                self.owner.seq.set_note_length(ti, self._drag_note_index, length)
                self.update_note_item(self._drag_note_index)

            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._dragging and ev.button() == Qt.MouseButton.LeftButton:
            ti = self.owner.current_track_index()

            if ti < 0:
                ti = 0

            idx = self._drag_note_index

            if self._drag_was_existing and (not self._drag_moved) and idx is not None:
                self.owner.seq.remove_note(ti, idx)
                self.owner.rebuild_roll(grid=False, notes=True)
            else:
                self.owner.rebuild_roll(grid=False, notes=False)

            self._dragging = False
            self._drag_note_index = None
            self._drag_was_existing = False
            self._drag_moved = False
            self._drag_last_len = 1

            self.owner._mark_dirty_and_restart_from_playhead()
            ev.accept()
            return

        super().mouseReleaseEvent(ev)


# ============================================================================
# Main GUI
# ============================================================================

class MelodyGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nate's MelodyProject - Fast Live Piano Roll + Blocks")

        self.seq = Sequence(sr=48000, bpm=120.0, steps_per_bar=16, bars=2)
        self.seq.tracks = [
            Track(name="Track 1", instruments=[BlockInstance("synth_keys", default_params("synth_keys"))]),
            Track(name="Track 2", instruments=[BlockInstance("guitar_pluck", default_params("guitar_pluck"))]),
        ]
        self.seq.ensure()

        self.piano_low_midi = 12
        self.piano_high_midi = 127
        self._piano_rows = self.piano_high_midi - self.piano_low_midi + 1

        self._loop = False
        self._start_step = 0

        self._ghost_root: Optional[str] = None
        self._ghost_scale: Optional[str] = None

        self._editing: Optional[Tuple[int, str, int]] = None
        self._param_lock = threading.RLock()
        self._param_dragging = False
        self._pending_dirty_render = False

        self._ballast = MemoryBallast()
        self._ballast_target_mb = 0

        self._cpu_ballast = CpuBallast()
        self._cpu_ballast_target_pct = 0
        self._process_for_stats = psutil.Process() if psutil else None

        self.audio = AudioEngine(
            self.seq,
            blocksize=512,
            latency="low",
            preview_seconds=2.0,
            preview_debounce_ms=60,
            full_debounce_ms=1000,
            on_error=self._post_error,
        )

        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(16)
        self._ui_timer.timeout.connect(self._tick_ui)

        self._dirty_timer = QTimer(self)
        self._dirty_timer.setSingleShot(True)
        self._dirty_timer.setInterval(35)
        self._dirty_timer.timeout.connect(self._apply_dirty_restart)

        self._full_render_timer = QTimer(self)
        self._full_render_timer.setSingleShot(True)
        self._full_render_timer.setInterval(700)
        self._full_render_timer.timeout.connect(self._apply_full_idle_render)

        self.init_ui()
        self._apply_dark_daw_theme()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        lr_split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(lr_split)

        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        transport_box = QGroupBox("Transport")
        tbar = QHBoxLayout(transport_box)
        tbar.setContentsMargins(10, 10, 10, 10)
        tbar.setSpacing(8)

        self.btn_play_toggle = QToolButton()
        self.btn_play_toggle.setText("▶")
        self.btn_play_toggle.setCheckable(True)
        self.btn_play_toggle.clicked.connect(self.on_play_toggle)

        self.btn_stop = QToolButton()
        self.btn_stop.setText("■")
        self.btn_stop.clicked.connect(self.on_stop)

        self.loop_box = QCheckBox("Loop")
        self.loop_box.stateChanged.connect(lambda _: self._set_loop(self.loop_box.isChecked()))

        self.bars_spin = QSpinBox()
        self.bars_spin.setMinimum(1)
        self.bars_spin.setMaximum(64)
        self.bars_spin.setValue(int(self.seq.bars))
        self.bars_spin.setFixedWidth(70)
        self.bars_spin.valueChanged.connect(self.on_bars_changed)

        self.btn_import_midi = QPushButton("Import MIDI")
        self.btn_import_midi.clicked.connect(self.on_import_midi)

        self.btn_export = QPushButton("Export WAV")
        self.btn_export.clicked.connect(self.on_export)

        tbar.addWidget(self.btn_play_toggle)
        tbar.addWidget(self.btn_stop)
        tbar.addWidget(self.loop_box)
        tbar.addSpacing(10)
        tbar.addWidget(QLabel("Bars"))
        tbar.addWidget(self.bars_spin)
        tbar.addStretch(1)
        tbar.addWidget(QLabel("Alt+Click sets playhead"))
        tbar.addWidget(self.btn_import_midi)
        tbar.addWidget(self.btn_export)

        left_layout.addWidget(transport_box)

        mem_row = QHBoxLayout()

        self.mem_spin = QSpinBox()
        self.mem_spin.setRange(0, 8192)
        self.mem_spin.setValue(0)
        self.mem_spin.setFixedWidth(90)
        self.mem_spin.valueChanged.connect(self.on_mem_ballast_changed)

        self.cpu_spin = QSpinBox()
        self.cpu_spin.setRange(0, 80)
        self.cpu_spin.setValue(0)
        self.cpu_spin.setFixedWidth(70)
        self.cpu_spin.valueChanged.connect(self.on_cpu_ballast_changed)

        self.mem_label = QLabel("RSS: ? MB")

        mem_row.addWidget(QLabel("Warm RAM (MB)"))
        mem_row.addWidget(self.mem_spin)
        mem_row.addSpacing(12)
        mem_row.addWidget(QLabel("CPU ballast (%)"))
        mem_row.addWidget(self.cpu_spin)
        mem_row.addSpacing(12)
        mem_row.addWidget(self.mem_label)
        mem_row.addStretch(1)

        left_layout.addLayout(mem_row)

        ghost_box = QGroupBox("Ghost scale")
        ghost_row = QHBoxLayout(ghost_box)
        ghost_row.setContentsMargins(10, 10, 10, 10)
        ghost_row.setSpacing(8)

        self.ghost_root = QComboBox()
        self.ghost_root.addItem("None")
        for n in NOTES:
            self.ghost_root.addItem(n)

        self.ghost_scale = QComboBox()
        self.ghost_scale.addItem("None")
        for s in sorted(SCALE_INTERVALS.keys()):
            self.ghost_scale.addItem(s)

        self.ghost_root.currentTextChanged.connect(self.on_ghost_changed)
        self.ghost_scale.currentTextChanged.connect(self.on_ghost_changed)

        ghost_row.addWidget(QLabel("Root"))
        ghost_row.addWidget(self.ghost_root)
        ghost_row.addWidget(QLabel("Scale"))
        ghost_row.addWidget(self.ghost_scale)
        ghost_row.addStretch(1)

        left_layout.addWidget(ghost_box)

        roll_box = QGroupBox("Piano Roll")
        roll_l = QVBoxLayout(roll_box)
        roll_l.setContentsMargins(10, 10, 10, 10)
        roll_l.setSpacing(8)

        self.roll = PianoRollView(self)
        roll_l.addWidget(self.roll, 1)

        left_layout.addWidget(roll_box, 1)

        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_split = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(right_split, 1)

        tracks_box = QGroupBox("Tracks")
        tracks_l = QVBoxLayout(tracks_box)
        tracks_l.setContentsMargins(10, 10, 10, 10)

        self.track_list = QListWidget()
        self.track_list.currentRowChanged.connect(self.on_track_selected)
        tracks_l.addWidget(self.track_list, 1)

        tracks_box.setMinimumHeight(120)
        right_split.addWidget(tracks_box)

        inst_box = QGroupBox("Instruments (layered)")
        inst_layout = QVBoxLayout(inst_box)
        inst_layout.setContentsMargins(10, 10, 10, 10)

        self.inst_picker = QComboBox()
        for n in BLOCKS.names():
            if block_kind(n) == "instrument":
                self.inst_picker.addItem(n)
        inst_layout.addWidget(self.inst_picker)

        inst_btns = QHBoxLayout()
        self.btn_add_inst = QPushButton("Add")
        self.btn_rm_inst = QPushButton("Remove")
        inst_btns.addWidget(self.btn_add_inst)
        inst_btns.addWidget(self.btn_rm_inst)
        inst_layout.addLayout(inst_btns)

        self.inst_list = QListWidget()
        inst_layout.addWidget(self.inst_list, 1)

        self.btn_add_inst.clicked.connect(self.add_instrument)
        self.btn_rm_inst.clicked.connect(self.remove_instrument)
        self.inst_list.currentRowChanged.connect(lambda _: self.select_block_for_edit("instruments"))

        inst_box.setMinimumHeight(150)
        right_split.addWidget(inst_box)

        fx_box = QGroupBox("FX (serial)")
        fx_layout = QVBoxLayout(fx_box)
        fx_layout.setContentsMargins(10, 10, 10, 10)

        self.fx_picker = QComboBox()
        for n in BLOCKS.names():
            if block_kind(n) == "fx":
                self.fx_picker.addItem(n)
        fx_layout.addWidget(self.fx_picker)

        fx_btns = QHBoxLayout()
        self.btn_add_fx = QPushButton("Add")
        self.btn_rm_fx = QPushButton("Remove")
        fx_btns.addWidget(self.btn_add_fx)
        fx_btns.addWidget(self.btn_rm_fx)
        fx_layout.addLayout(fx_btns)

        self.fx_list = QListWidget()
        fx_layout.addWidget(self.fx_list, 1)

        self.btn_add_fx.clicked.connect(self.add_fx)
        self.btn_rm_fx.clicked.connect(self.remove_fx)
        self.fx_list.currentRowChanged.connect(lambda _: self.select_block_for_edit("fx"))

        fx_box.setMinimumHeight(150)
        right_split.addWidget(fx_box)

        self.params_outer = QGroupBox("Block Params")
        outer_v = QVBoxLayout(self.params_outer)
        outer_v.setContentsMargins(10, 10, 10, 10)

        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        outer_v.addWidget(self.params_scroll, 1)

        self.params_inner = QWidget()
        self.params_form = QFormLayout(self.params_inner)
        self.params_scroll.setWidget(self.params_inner)

        self.params_outer.setMinimumHeight(220)
        right_split.addWidget(self.params_outer)

        right_split.setStretchFactor(0, 0)
        right_split.setStretchFactor(1, 1)
        right_split.setStretchFactor(2, 1)
        right_split.setStretchFactor(3, 3)
        right_split.setSizes([150, 220, 220, 380])

        lr_split.addWidget(left_pane)
        lr_split.addWidget(right_pane)
        lr_split.setStretchFactor(0, 3)
        lr_split.setStretchFactor(1, 1)

        self.refresh_tracks()
        self.track_list.setCurrentRow(0)
        self.rebuild_roll()
        self._update_mem_label()

        act = QAction(self)
        act.setShortcut("Space")
        act.triggered.connect(lambda: self.btn_play_toggle.click())
        self.addAction(act)

    def _apply_dark_daw_theme(self):
        self.setStyleSheet("""
        QWidget { background: #151515; color: #e8e8e8; }
        QGroupBox {
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            margin-top: 10px;
            padding: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px 0 6px;
            color: #cfcfcf;
        }
        QPushButton {
            background: #262626;
            border: 1px solid #353535;
            border-radius: 8px;
            padding: 6px 10px;
        }
        QPushButton:hover { background: #2d2d2d; }
        QPushButton:pressed { background: #1f1f1f; }
        QToolButton {
            background: #262626;
            border: 1px solid #353535;
            border-radius: 10px;
            padding: 6px 10px;
            font-size: 16px;
            min-width: 34px;
        }
        QToolButton:checked {
            background: #1e3a57;
            border-color: #2f6aa0;
        }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background: #1d1d1d;
            border: 1px solid #333333;
            border-radius: 6px;
            padding: 4px;
        }
        QListWidget {
            background: #191919;
            border: 1px solid #2f2f2f;
            border-radius: 8px;
        }
        QScrollArea { border: none; }
        QSlider::groove:horizontal {
            background: #252525;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #4a90e2;
            border: 1px solid #75aef0;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        """)

        self.roll.setStyleSheet(
            "background: #111111; border: 1px solid #2f2f2f; border-radius: 10px;"
        )

    # ------------------------------------------------------------------
    # Safe event posting
    # ------------------------------------------------------------------

    def _post_error(self, title: str, body: str):
        app = QApplication.instance()

        if app is None:
            return

        def show():
            QMessageBox.critical(self, title, body)

        QApplication.postEvent(app, _CallableEvent(show))

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _step_samples(self) -> int:
        try:
            return int(self.seq.step_samples())
        except Exception:
            return max(1, int(round(self.seq.step_seconds() * self.seq.sr)))

    def _set_loop(self, on: bool):
        self._loop = bool(on)
        self.audio.set_loop(self._loop)

    def _set_start_step(self, step: int):
        self._start_step = int(max(0, min(self.seq.total_steps() - 1, int(step))))
        self.roll.set_playhead_step(self._start_step)

        if self.audio.is_playing:
            self.audio.seek(self._start_step * self._step_samples())

    def on_ghost_changed(self, *_):
        r = self.ghost_root.currentText().strip()
        s = self.ghost_scale.currentText().strip()

        self._ghost_root = None if r.lower() == "none" else r
        self._ghost_scale = None if s.lower() == "none" else s

        self.rebuild_roll()

    def current_track_index(self) -> int:
        return self.track_list.currentRow()

    def refresh_tracks(self):
        self.track_list.clear()

        for t in self.seq.tracks:
            self.track_list.addItem(t.name)

    def on_track_selected(self, idx: int):
        if idx < 0:
            return

        self.refresh_stacks(idx)
        self.clear_param_editor()
        self.rebuild_roll()

    def refresh_stacks(self, track_i: int):
        if not (0 <= int(track_i) < len(self.seq.tracks)):
            return

        tr = self.seq.tracks[int(track_i)]

        self.inst_list.clear()
        for bi in tr.instruments:
            self.inst_list.addItem(f"{bi.name}")

        self.fx_list.clear()
        for bi in tr.fx:
            self.fx_list.addItem(f"{bi.name}")

    def rebuild_roll(self, *, grid: bool = True, notes: bool = True):
        self.seq.ensure()

        if grid:
            self.roll.rebuild_grid()

        if notes:
            self.roll.rebuild_notes()

        self.roll.set_playhead_step(self._start_step)

    # ------------------------------------------------------------------
    # Fast dirty render scheduling
    # ------------------------------------------------------------------

    def _mark_dirty_and_restart_from_playhead(self, *, full_later: bool = True):
        """
        Called after notes, params, FX, instruments, bars, or MIDI change.

        This does not render in the GUI thread.
        It schedules a fast preview render through AudioEngine.
        """
        try:
            self.seq.touch()
        except Exception:
            pass

        self._pending_dirty_render = True
        self._dirty_timer.start()

        if full_later:
            self._full_render_timer.start()

    def _apply_dirty_restart(self):
        if not self._pending_dirty_render:
            return

        self._pending_dirty_render = False

        if self.audio.is_playing:
            self.audio.request_preview_rerender(immediate=False, full_later=True)

    def _apply_full_idle_render(self):
        if self.audio.is_playing:
            self.audio.request_full_rerender(immediate=False)

    # ------------------------------------------------------------------
    # Stack editing
    # ------------------------------------------------------------------

    def add_instrument(self):
        ti = self.current_track_index()

        if ti < 0:
            return

        name = self.inst_picker.currentText().strip().lower()

        with self._param_lock:
            self.seq.tracks[ti].instruments.append(BlockInstance(name, default_params(name)))

        self.refresh_stacks(ti)
        self._mark_dirty_and_restart_from_playhead()

    def remove_instrument(self):
        ti = self.current_track_index()

        if ti < 0:
            return

        idx = self.inst_list.currentRow()

        if idx < 0:
            return

        with self._param_lock:
            tr = self.seq.tracks[ti]

            if 0 <= idx < len(tr.instruments):
                tr.instruments.pop(idx)

        self.refresh_stacks(ti)
        self.clear_param_editor()
        self._mark_dirty_and_restart_from_playhead()

    def add_fx(self):
        ti = self.current_track_index()

        if ti < 0:
            return

        name = self.fx_picker.currentText().strip().lower()

        with self._param_lock:
            self.seq.tracks[ti].fx.append(BlockInstance(name, default_params(name)))

        self.refresh_stacks(ti)
        self._mark_dirty_and_restart_from_playhead()

    def remove_fx(self):
        ti = self.current_track_index()

        if ti < 0:
            return

        idx = self.fx_list.currentRow()

        if idx < 0:
            return

        with self._param_lock:
            tr = self.seq.tracks[ti]

            if 0 <= idx < len(tr.fx):
                tr.fx.pop(idx)

        self.refresh_stacks(ti)
        self.clear_param_editor()
        self._mark_dirty_and_restart_from_playhead()

    # ------------------------------------------------------------------
    # Ballast controls
    # ------------------------------------------------------------------

    def _ballast_release_safe(self):
        for name in ("release", "clear"):
            fn = getattr(self._ballast, name, None)

            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass

        try:
            self._ballast.set_target_mb(0)
        except Exception:
            pass

    def _ballast_set_safe(self, mb: int):
        try:
            self._ballast.set_target_mb(mb)
            return
        except TypeError:
            pass
        except Exception:
            raise

        try:
            self._ballast.set_target_mb(mb, touch=True, chunk_mb=32)
        except Exception:
            raise

    def _ballast_held_mb_safe(self) -> float:
        try:
            val = getattr(self._ballast, "held_mb")
            return float(val() if callable(val) else val)
        except Exception:
            return float(self._ballast_target_mb)

    def on_cpu_ballast_changed(self, pct: int):
        pct = int(max(0, min(80, int(pct))))

        if pct != self.cpu_spin.value():
            self.cpu_spin.blockSignals(True)
            self.cpu_spin.setValue(pct)
            self.cpu_spin.blockSignals(False)

        self._cpu_ballast_target_pct = pct

        try:
            self._cpu_ballast.set_target_pct(pct)
        except Exception:
            try:
                self._cpu_ballast.stop()
            except Exception:
                pass

            self._cpu_ballast_target_pct = 0
            self.cpu_spin.blockSignals(True)
            self.cpu_spin.setValue(0)
            self.cpu_spin.blockSignals(False)

        self._update_mem_label()

    def on_mem_ballast_changed(self, mb: int):
        mb = int(max(0, int(mb)))

        try:
            if psutil:
                avail_mb = int(psutil.virtual_memory().available / (1024 * 1024))
                safe_max = max(0, avail_mb - 512)
                mb = int(max(0, min(mb, safe_max)))

            if mb != self.mem_spin.value():
                self.mem_spin.blockSignals(True)
                self.mem_spin.setValue(mb)
                self.mem_spin.blockSignals(False)

            self._ballast_target_mb = mb
            self._ballast_set_safe(mb)

        except Exception:
            self._ballast_release_safe()
            self._ballast_target_mb = 0

            self.mem_spin.blockSignals(True)
            self.mem_spin.setValue(0)
            self.mem_spin.blockSignals(False)

        self._update_mem_label()

    def _update_mem_label(self):
        try:
            rss_mb = "?"
            avail_mb = "?"
            held_mb = "?"
            cpu_raw = "?"
            cpu_norm = "?"
            cores = "?"

            if psutil:
                proc = self._process_for_stats or psutil.Process()
                rss_mb = int(proc.memory_info().rss / (1024 * 1024))
                avail_mb = int(psutil.virtual_memory().available / (1024 * 1024))
                cores = int(psutil.cpu_count(logical=True) or 1)

                raw = float(proc.cpu_percent(interval=None))
                cpu_raw = f"{raw:.1f}"
                cpu_norm = f"{raw / max(1, cores):.1f}"

            held_mb = f"{self._ballast_held_mb_safe():.0f}"

            self.mem_label.setText(
                f"RSS: {rss_mb} MB | Ballast: {held_mb} MB | Avail: {avail_mb} MB | "
                f"CPU(proc): {cpu_raw}% raw | {cpu_norm}% norm/{cores} cores"
            )

        except Exception:
            self.mem_label.setText("RSS: ? MB")

    # ------------------------------------------------------------------
    # Param editor
    # ------------------------------------------------------------------

    def _clear_param_rows(self):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)

    def clear_param_editor(self):
        self._clear_param_rows()
        self._editing = None

    def select_block_for_edit(self, stack_name: str):
        ti = self.current_track_index()

        if ti < 0:
            self.clear_param_editor()
            return

        blk_instance: Optional[BlockInstance] = None
        bi_idx = -1

        if stack_name == "instruments":
            bi_idx = self.inst_list.currentRow()

            if 0 <= bi_idx < len(self.seq.tracks[ti].instruments):
                blk_instance = self.seq.tracks[ti].instruments[bi_idx]

        elif stack_name == "fx":
            bi_idx = self.fx_list.currentRow()

            if 0 <= bi_idx < len(self.seq.tracks[ti].fx):
                blk_instance = self.seq.tracks[ti].fx[bi_idx]

        if blk_instance is None:
            self.clear_param_editor()
            return

        self._editing = (ti, stack_name, bi_idx)
        self.build_param_editor(blk_instance.name, blk_instance.params)

    def build_param_editor(self, block_name: str, params: Dict[str, Any]):
        self._clear_param_rows()

        schema = block_params_schema(block_name)

        if not schema:
            self.params_form.addRow(QLabel("(No params)"), QLabel(""))
            return

        with self._param_lock:
            for key, spec in schema.items():
                params.setdefault(key, spec.get("default"))

        for key, spec in schema.items():
            ptype = spec.get("type", "float")
            default = spec.get("default")

            if ptype == "bool":
                w = QCheckBox()
                w.setChecked(bool(params.get(key)))
                w.toggled.connect(lambda checked, k=key: self._set_param(k, bool(checked)))
                self.params_form.addRow(QLabel(key), w)
                continue

            if ptype == "str":
                w = QLineEdit()
                w.setText(str(params.get(key, default) or ""))
                w.textChanged.connect(lambda val, k=key: self._set_param(k, str(val)))
                self.params_form.addRow(QLabel(key), w)
                continue

            if ptype == "choice":
                w = QComboBox()
                choices = list(spec.get("choices", []))

                for c in choices:
                    w.addItem(str(c))

                cur = str(params.get(key, default))

                if cur in [str(x) for x in choices]:
                    w.setCurrentText(cur)

                w.currentTextChanged.connect(lambda val, k=key: self._set_param(k, val))
                self.params_form.addRow(QLabel(key), w)
                continue

            is_int = ptype == "int"
            mn, mx, step, steps_int = _slider_steps_for_spec(spec, is_int=is_int)

            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(8)

            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setMinimum(0)
            sld.setMaximum(steps_int)
            sld.setSingleStep(1)
            sld.setPageStep(max(1, steps_int // 20))

            if is_int:
                spn = QSpinBox()
                spn.setMinimum(int(round(mn)))
                spn.setMaximum(int(round(mx)))
                spn.setSingleStep(max(1, int(round(spec.get("step", 1)))))
                spn.setFixedWidth(90)
            else:
                spn = QDoubleSpinBox()
                spn.setDecimals(int(spec.get("decimals", 6)))
                spn.setMinimum(float(mn))
                spn.setMaximum(float(mx))
                spn.setSingleStep(float(spec.get("step", 0.01)))
                spn.setFixedWidth(110)

            cur_v = params.get(key, default)
            cur_v = float(cur_v if cur_v is not None else (mn + mx) * 0.5)
            cur_v = float(np.clip(cur_v, mn, mx))

            sld.blockSignals(True)
            spn.blockSignals(True)

            sld.setValue(_value_to_slider(cur_v, mn, step, steps_int))

            if is_int:
                spn.setValue(int(round(cur_v)))
            else:
                spn.setValue(cur_v)

            sld.blockSignals(False)
            spn.blockSignals(False)

            sld.sliderPressed.connect(self._begin_param_drag)
            sld.sliderReleased.connect(self._end_param_drag)

            def on_slider(
                val,
                k=key,
                spin=spn,
                mn_=mn,
                step_=step,
                is_int_=is_int,
            ):
                v = _slider_to_value(int(val), mn_, step_)

                if is_int_:
                    v = int(round(v))
                    spin.blockSignals(True)
                    spin.setValue(v)
                    spin.blockSignals(False)
                    self._set_param(k, v)
                else:
                    spin.blockSignals(True)
                    spin.setValue(float(v))
                    spin.blockSignals(False)
                    self._set_param(k, float(v))

            def on_spin(
                val,
                k=key,
                slider=sld,
                mn_=mn,
                mx_=mx,
                step_=step,
                steps_=steps_int,
                is_int_=is_int,
            ):
                v = float(np.clip(float(val), mn_, mx_))

                slider.blockSignals(True)
                slider.setValue(_value_to_slider(v, mn_, step_, steps_))
                slider.blockSignals(False)

                self._set_param(k, int(round(v)) if is_int_ else float(v))

            sld.valueChanged.connect(on_slider)
            spn.valueChanged.connect(on_spin)

            h.addWidget(sld, 1)
            h.addWidget(spn, 0)

            self.params_form.addRow(QLabel(key), row)

    def _begin_param_drag(self):
        self._param_dragging = True

    def _end_param_drag(self):
        self._param_dragging = False
        self._mark_dirty_and_restart_from_playhead(full_later=True)

        if self.audio.is_playing:
            self.audio.request_full_rerender(immediate=False)

    def _set_param(self, key: str, value: Any):
        if not self._editing:
            return

        ti, stack_name, bi = self._editing
        chain_kind = "instrument" if stack_name == "instruments" else "fx"

        self._set_block_param_fast(
            track_i=ti,
            chain_kind=chain_kind,
            block_i=bi,
            param_name=key,
            value=value,
            live=True,
        )

    def _set_block_param_fast(
        self,
        *,
        track_i: int,
        chain_kind: str,
        block_i: int,
        param_name: str,
        value: Any,
        live: bool = True,
    ):
        if not (0 <= int(track_i) < len(self.seq.tracks)):
            return

        track = self.seq.tracks[int(track_i)]

        if chain_kind == "instrument":
            chain = track.instruments
        elif chain_kind == "fx":
            chain = track.fx
        else:
            return

        if not (0 <= int(block_i) < len(chain)):
            return

        with self._param_lock:
            chain[int(block_i)].params[str(param_name)] = value

        if live:
            self._mark_dirty_and_restart_from_playhead(full_later=not self._param_dragging)

    # ------------------------------------------------------------------
    # MIDI import
    # ------------------------------------------------------------------

    def _ticks_to_step_len(
        self,
        start_tick: int,
        end_tick: int,
        ppq: int,
        steps_per_beat: float,
    ) -> Tuple[int, int]:
        start_beats = float(start_tick) / float(ppq)
        end_beats = float(end_tick) / float(ppq)

        s0 = int(round(start_beats * steps_per_beat))
        s1 = int(round(end_beats * steps_per_beat))

        if s1 <= s0:
            s1 = s0 + 1

        return s0, int(s1 - s0)

    def _parse_midi_to_notes_and_bars(
        self,
        path: str,
    ) -> Tuple[Optional[float], int, List[Tuple[int, int, Tuple[str, int]]]]:
        try:
            import mido
        except Exception:
            raise RuntimeError("Missing dependency: mido. Install with: pip install mido")

        mid = mido.MidiFile(path)
        ppq = int(getattr(mid, "ticks_per_beat", 480) or 480)

        bpm: Optional[float] = None
        tempo_us: Optional[int] = None

        for tr in mid.tracks:
            for msg in tr:
                if getattr(msg, "is_meta", False) and msg.type == "set_tempo":
                    tempo_us = int(msg.tempo)
                    break

            if tempo_us is not None:
                break

        if tempo_us is not None and tempo_us > 0:
            bpm = 60_000_000.0 / float(tempo_us)

        steps_per_bar = int(getattr(self.seq, "steps_per_bar", 16) or 16)
        steps_per_beat = float(steps_per_bar) / 4.0

        active: dict[tuple[int, int, int], int] = {}
        out_notes: List[Tuple[int, int, Tuple[str, int]]] = []
        max_end_step = 0

        for track_i, tr in enumerate(mid.tracks):
            abs_tick = 0

            for msg in tr:
                abs_tick += int(getattr(msg, "time", 0) or 0)

                if getattr(msg, "is_meta", False):
                    continue

                mtype = getattr(msg, "type", "")

                if mtype not in ("note_on", "note_off"):
                    continue

                note = int(getattr(msg, "note", 0) or 0)
                ch = int(getattr(msg, "channel", 0) or 0)
                vel = int(getattr(msg, "velocity", 0) or 0)

                key = (track_i, ch, note)

                is_on = mtype == "note_on" and vel > 0
                is_off = mtype == "note_off" or (mtype == "note_on" and vel == 0)

                if is_on:
                    if key in active:
                        start_tick = active.pop(key)
                        s0, ln = self._ticks_to_step_len(start_tick, abs_tick, ppq, steps_per_beat)
                        pitch = midi_to_note_oct(note)
                        out_notes.append((s0, ln, pitch))
                        max_end_step = max(max_end_step, s0 + ln)

                    active[key] = abs_tick

                elif is_off:
                    if key not in active:
                        continue

                    start_tick = active.pop(key)
                    s0, ln = self._ticks_to_step_len(start_tick, abs_tick, ppq, steps_per_beat)
                    pitch = midi_to_note_oct(note)
                    out_notes.append((s0, ln, pitch))
                    max_end_step = max(max_end_step, s0 + ln)

        inferred_end_step = max_end_step if out_notes else 0
        bars_needed = max(1, int(math.ceil(float(max(1, inferred_end_step)) / float(steps_per_bar))))

        total_steps = bars_needed * steps_per_bar
        cleaned: List[Tuple[int, int, Tuple[str, int]]] = []

        for s0, ln, pitch in out_notes:
            s0 = int(max(0, min(total_steps - 1, int(s0))))
            ln = int(max(1, int(ln)))

            if s0 + ln > total_steps:
                ln = max(1, total_steps - s0)

            cleaned.append((s0, ln, pitch))

        cleaned.sort(key=lambda x: (x[0], x[2][1], x[2][0]))
        return bpm, bars_needed, cleaned

    def on_import_midi(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import MIDI",
            "",
            "MIDI Files (*.mid *.midi)",
        )

        if not path:
            return

        ti = self.current_track_index()

        if ti < 0:
            ti = 0

        try:
            bpm, bars_needed, notes = self._parse_midi_to_notes_and_bars(path)
        except Exception as e:
            QMessageBox.critical(self, "MIDI import failed", str(e))
            return

        with self._param_lock:
            if bpm is not None and bpm > 1:
                self.seq.bpm = float(bpm)

            self.seq.bars = int(max(1, bars_needed))
            self.seq.ensure()

            while len(self.seq.notes) <= ti:
                self.seq.notes.append([])

            self.seq.notes[ti] = []

            for s0, ln, pitch in notes:
                self.seq.add_note(ti, int(s0), pitch, int(ln))

            try:
                self.seq.touch()
            except Exception:
                pass

        self.bars_spin.blockSignals(True)
        self.bars_spin.setValue(int(self.seq.bars))
        self.bars_spin.blockSignals(False)

        if self._start_step >= self.seq.total_steps():
            self._start_step = max(0, self.seq.total_steps() - 1)

        self.rebuild_roll(grid=True, notes=True)
        self._mark_dirty_and_restart_from_playhead()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def on_bars_changed(self, bars: int):
        bars = int(max(1, bars))

        with self._param_lock:
            self.seq.bars = bars
            self.seq.ensure()
            new_steps = int(self.seq.total_steps())

            for ti in range(len(self.seq.notes)):
                kept = []

                for ev in self.seq.notes[ti]:
                    if int(ev.start_step) < new_steps:
                        max_len = max(1, new_steps - int(ev.start_step))
                        ev.length_steps = int(min(int(ev.length_steps), max_len))
                        kept.append(ev)

                self.seq.notes[ti] = kept

            try:
                self.seq.touch()
            except Exception:
                pass

        if self._start_step >= self.seq.total_steps():
            self._start_step = max(0, self.seq.total_steps() - 1)

        self.rebuild_roll()
        self._mark_dirty_and_restart_from_playhead()

    def on_play_toggle(self):
        if self.btn_play_toggle.isChecked():
            self.btn_play_toggle.setText("⏸")
            self.on_play()
        else:
            self.btn_play_toggle.setText("▶")
            self.on_pause()

    def on_play(self):
        start_sample = int(self._start_step * self._step_samples())
        self.audio.start(start_sample=start_sample, loop=bool(self._loop))
        self._ui_timer.start()

    def on_pause(self):
        self.audio.pause()
        self._ui_timer.stop()

    def on_stop(self):
        self.on_pause()

        self.btn_play_toggle.blockSignals(True)
        self.btn_play_toggle.setChecked(False)
        self.btn_play_toggle.setText("▶")
        self.btn_play_toggle.blockSignals(False)

        self.audio.stop()
        self._set_start_step(self._start_step)
        self.roll.set_playhead_step(self._start_step)

    def _tick_ui(self):
        if not self.audio.is_playing:
            if self.btn_play_toggle.isChecked():
                self.btn_play_toggle.blockSignals(True)
                self.btn_play_toggle.setChecked(False)
                self.btn_play_toggle.setText("▶")
                self.btn_play_toggle.blockSignals(False)

            return

        step_samples = int(self._step_samples())

        if step_samples <= 0:
            return

        pos = int(self.audio.position_samples)
        step_float = float(pos) / float(step_samples)
        self.roll.set_playhead_step(step_float)
        self._update_mem_label()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export WAV",
            "out.wav",
            "WAV Files (*.wav)",
        )

        if not path:
            return

        try:
            self.audio.export_wav(path)
            QMessageBox.information(self, "Export Successful", f"Audio exported to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        try:
            self.audio.shutdown()
        except Exception:
            pass

        try:
            self._cpu_ballast.stop()
        except Exception:
            pass

        self._ballast_release_safe()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)

    catcher = _EventCatcher()
    app.installEventFilter(catcher)
    app._melody_event_catcher = catcher  # keep reference alive

    w = MelodyGUI()
    w.resize(1450, 850)
    w.show()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()