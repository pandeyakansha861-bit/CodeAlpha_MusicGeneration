# Training Data Directory (`data/`)

This directory is designated for storing raw MIDI (`.mid` or `.midi`) files used to train the LSTM neural network.

---

## 📁 How to Place Your Data
1. Place standard `.mid` or `.midi` files directly inside this `data/` directory.
2. Example supported file formats:
   - `classical_sonata.mid`
   - `chopin_nocturne.midi`
   - `folk_melody.mid`

---

## 🎼 Best Practices for Training Data
- **Genre Consistency**: For optimal harmonic coherence, use music files of a similar genre (e.g., solo piano pieces, classical melodies, or chiptune/8-bit tunes).
- **Polyphony & Instrumentation**: Single-instrument tracks (like solo piano) produce cleaner note-prediction patterns and train much faster.
- **Dataset Size**:
  - Minimum: At least 1-3 MIDI files containing a combined total of > 100 notes.
  - Recommended: 10-30 MIDI files for richer vocabulary and melodic transitions.

---

## 🌐 Where to Find Free & Public Domain MIDI Files
- **Free Classical MIDI**: [Classical Archives](https://www.classicalarchives.com/) / [Kunstderfuge](https://www.kunstderfuge.com/)
- **MAESTRO Dataset**: [Magenta MAESTRO](https://magenta.tensorflow.org/datasets/maestro)
- **Video Game Music (VGMusic)**: [VGMusic Archive](https://www.vgmusic.com/)
- **MuseScore / Free MIDI collections**: Public domain piano pieces (Bach, Chopin, Mozart, Beethoven).

---

## ⚡ Quick Starter Datasets
If you don't have MIDI files immediately available, `train_model.py` can generate synthetic starter classical/folk MIDI tracks directly into this directory using the command:
```bash
python train_model.py --create-samples
```
This enables end-to-end testing of the training pipeline out of the box!
