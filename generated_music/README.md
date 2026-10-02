# Generated Music Output Directory (`generated_music/`)

This directory is the output destination for all newly synthesized MIDI tracks created by:
- `generate_music.py` (CLI generation script)
- `app.py` (Streamlit Web Interface)

---

## 🎵 File Naming Convention
Generated files follow a descriptive timestamped naming structure:
```
music_gen_YYYYMMDD_HHMMSS.mid
```
Example: `music_gen_20261001_235900.mid`

---

## 🎧 How to Play Generated MIDI Files
Standard MIDI files contain event-based musical performance data rather than raw audio waveforms. You can listen to or inspect them using:
1. **The Streamlit App (`app.py`)**: Has a built-in WebAudio HTML5 player for direct browser playback.
2. **Media Players**: VLC Media Player, Windows Media Player, QuickTime.
3. **Digital Audio Workstations (DAWs)**: FL Studio, Ableton Live, Logic Pro, GarageBand, Reaper, Cakewalk.
4. **Notation Software**: MuseScore, Sibelius, Finale.
5. **Online MIDI Visualizers**: [Online Sequencer](https://onlinesequencer.net/) or [Visual MIDI](https://midivisualizer.github.io/).
