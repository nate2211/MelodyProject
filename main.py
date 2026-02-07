import sys
import argparse

from pipeline import Sequence, Track, BlockInstance, write_wav
import sounds  # registers blocks

def cli_main():
    parser = argparse.ArgumentParser(description="MelodyProject Modular")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")
    parser.add_argument("--out", default="out.wav", help="Output file for CLI mode")
    args = parser.parse_args()

    if args.gui:
        from PyQt6.QtWidgets import QApplication
        from gui import MelodyGUI
        app = QApplication(sys.argv)
        ex = MelodyGUI()
        ex.show()
        sys.exit(app.exec())

    # CLI example: layer synth + guitar
    seq = Sequence()
    seq.tracks.append(
        Track(
            name="Layered",
            instruments=[
                BlockInstance("synth_keys", {"wave": "saw", "amp": 0.10, "attack": 0.002, "release": 0.03, "pan": -0.2}),
                BlockInstance("guitar_pluck", {"amp": 0.20, "decay": 0.988, "tone": 0.6, "pick_ms": 6.0, "pan": 0.2}),
            ],
            fx=[
                BlockInstance("gain", {"gain_db": -6.0}),
                BlockInstance("delay", {"time_ms": 220.0, "feedback": 0.35, "mix": 0.25}),
                BlockInstance("lowpass", {"cutoff_hz": 9000.0}),
                BlockInstance("softclip", {"drive": 1.2}),
            ],
            volume=1.0,
        )
    )

    seq.ensure_grid()
    # a few notes
    seq.set_note(0, 0, ("C", 4))
    seq.set_note(0, 4, ("E", 4))
    seq.set_note(0, 8, ("G", 4))
    seq.set_note(0, 12, ("C", 5))

    print(f"Rendering to {args.out}...")
    buf = seq.render()
    write_wav(args.out, buf)
    print("Done.")

if __name__ == "__main__":
    cli_main()
