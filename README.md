MelodyProject: Modular DAW & Audio Engine
<img width="1441" height="869" alt="Screenshot 2026-02-16 221235" src="https://github.com/user-attachments/assets/4b837656-ffe3-40d8-9772-132a8461851a" />

MelodyProject is a lightweight, modular digital audio workstation (DAW) and audio processing engine built with Python, NumPy, and PyQt6. It features an "FL-Studio style" interface with a functional piano roll, polyphonic rendering, and a block-based architecture for instruments and effects.
🚀 Features
🎹 Advanced Piano Roll Interface

    Polyphonic Sequencing: Support for complex chords and overlapping notes across multiple octaves.

    Ghost Notes: Intelligent scale-aware shading to help you stay in key (Major, Minor, Pentatonic, Blues, and Modes).

    Smooth Playhead: Real-time playhead movement driven by high-resolution audio sample position.

    Visual Editing: Right-click to delete, drag to resize note length, and Alt+Click to set the start playback position.

🛠 Modular Block Architecture

The engine is built on a "Blocks" system where instruments and effects are interchangeable modules.

    Instruments: Includes synth_keys (PolyBLEP oscillators), guitar_pluck (Karplus-Strong physical modeling), and bell_fm.

    Effects (FX): Serial processing chains including Delay, Reverb (Schroeder-style), Transient Shapers, Biquad Filters (LP/HP/BP), and Soft Clipping.

    Humanization: Dedicated realism layers like piano_key_noise, piano_soundboard resonance, and piano_pedal_bloom to add acoustic life to synthetic sounds.

🎧 High-Fidelity Audio Engine

    NumPy-Powered: All DSP logic is vectorized in NumPy for high performance.

    Stateless Buffer Processing: Designed for stability and predictable rendering.

    Export Capabilities: Render your sequences directly to 16-bit 48kHz WAV files.

    Real-time Monitoring: Low-latency playback using sounddevice.

📦 Installation
Prerequisites

    Python 3.8+

    NumPy

    PyQt6

    sounddevice

Setup

    Clone the repository:
    Bash

    git clone https://github.com/yourusername/MelodyProject.git
    cd MelodyProject

    Install dependencies:
    Bash

    pip install -r requirements.txt

🛠 Usage
Launching the GUI

To start the workstation interface:
Bash

python gui.py

CLI Mode

You can render sequences via the command line for automated workflows:
Bash

python main.py --out my_track.wav

🏗 Project Structure

    gui.py: The main PyQt6 application logic and custom graphics views.

    pipeline.py: The core audio engine, sequencer logic, and track management.

    sounds.py: Implementation of primary instrument oscillators and basic FX.

    humanize.py & realism.py: Specialized DSP blocks for adding acoustic characteristics and "imperfections."

    main.py: Entry point for both CLI and GUI modes.

🧪 DSP Details
Karplus-Strong Synthesis

The guitar_pluck block uses a refined Karplus-Strong algorithm with a pick-transient exciter and a nonlinear string feedback loop to simulate physical string behavior.
PolyBLEP Oscillators

Synth oscillators use Polyphase Band-Limited Step (PolyBLEP) techniques to reduce aliasing, providing a cleaner sound at high frequencies compared to naive waveforms.
📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing

Contributions are welcome! Whether it's adding new DSP blocks, refining the UI, or improving performance, feel free to fork and submit a Pull Request.
