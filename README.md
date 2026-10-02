# 🎵 CodeAlpha AI Music Generation (Task 3)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64-red.svg)](https://streamlit.io/)
[![Music21](https://img.shields.io/badge/Music21-9.9-green.svg)](https://web.mit.edu/music21/)
[![CodeAlpha Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship-purple.svg)](https://codealpha.tech/)

An end-to-end Artificial Intelligence Music Generation application developed for the **CodeAlpha Artificial Intelligence Internship (Task 3)**. The project implements a deep learning sequence model using **Long Short-Term Memory (LSTM)** recurrent neural networks to learn melodic patterns from MIDI datasets and compose original musical sequences.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Objective](#-objective)
3. [How AI Music Generation Works](#-how-ai-music-generation-works)
4. [Deep Learning Architecture](#-deep-learning-architecture)
5. [Technologies & Libraries Used](#-technologies--libraries-used)
6. [Project Structure](#-project-structure)
7. [Dataset Requirements & Preprocessing](#-dataset-requirements--preprocessing)
8. [Installation & Setup](#-installation--setup)
9. [Training the Model](#-training-the-model)
10. [Generating Music (CLI)](#-generating-music-cli)
11. [Running the Streamlit Web Application](#-running-the-streamlit-web-application)
12. [Example Workflow](#-example-workflow)
13. [Limitations](#-limitations)
14. [Future Improvements](#-future-improvements)
15. [Author & Acknowledgements](#-author--acknowledgements)

---

## 🌟 Project Overview

Music is fundamentally sequential and structural: notes, chords, and rests unfold over time following harmonic rules and melodic motifs. This project builds a standalone, locally runnable AI composer that learns musical syntax directly from raw Musical Instrument Digital Interface (MIDI) files.

The system features:
- **Zero Paid APIs**: Fully local execution without requiring OpenAI, Gemini, or external API keys.
- **Robust Data Pipeline**: Uses `music21` to extract polyphonic notes and chords into symbolic token streams.
- **Stacked LSTM Neural Network**: Models long-range temporal dependencies and harmonic transitions.
- **Temperature-Controlled Sampling**: Enables flexible exploration of creativity versus strict harmonic fidelity.
- **Interactive Streamlit Web UI**: Includes real-time model status detection, hyperparameter controls, piano-roll visualizations, an in-browser WebAudio synthesizer, and one-click MIDI downloads.

---

## 🎯 Objective

- Build a machine learning pipeline capable of parsing, tokenizing, and encoding symbolic musical scores.
- Formulate music generation as an autoregressive next-token prediction task.
- Train an LSTM recurrent neural network on sequence pairs.
- Invert generated discrete tokens back into valid, standard MIDI files playable in any media player or DAW.
- Deliver a beginner-friendly, clean, and locally verifiable codebase suitable for internship evaluation.

---

## 🔬 How AI Music Generation Works

```
   Raw MIDI Files (.mid / .midi)
                │
                ▼
   [ music21 Parsing Engine ] ────► Extract Notes ('C4') & Chords ('C4.E4.G4')
                │
                ▼
   [ Categorical Tokenization ] ──► Map Unique Tokens to Discrete Integer IDs
                │
                ▼
   [ Sliding Window Generator ] ──► Window of N Notes (Input X) ──► Next Note (Target y)
                │
                ▼
   [ Neural Network Training ] ───► Embedding ──► Stacked LSTMs ──► Softmax Output
                │
                ▼
   [ Autoregressive Inference ] ──► Prime with Seed ──► Temperature Softmax Sampling
                │
                ▼
   [ MIDI Synthesis Engine ] ─────► Convert Tokens to Durations & Offsets ──► .mid Output
```

1. **Extraction**: A MIDI file contains events (note on/off, pitch, velocity, duration). `music21` extracts sequential notes (`note.Note`) and harmonic clusters (`chord.Chord`).
2. **Representation**:
   - Single notes are represented by pitch and octave: e.g., `"C4"`, `"F#5"`.
   - Simultaneous notes (chords) are represented by period-delimited pitches: e.g., `"C4.E4.G4"`.
3. **Sequence Construction**: A sliding window of context (e.g., 32 notes) is mapped to the single subsequent note.
4. **Learning**: The network optimizes cross-entropy loss to assign high probabilities to notes that naturally follow given melodic context.
5. **Generation**: Given an initial seed sequence, the model predicts the probability distribution over all possible notes, samples one using temperature scaling, appends it, slides the window, and repeats.

---

## 🧠 Deep Learning Architecture

The model is built using TensorFlow/Keras with the following layers:

| Layer | Type | Configuration | Purpose |
|---|---|---|---|
| 1 | **Input** | `(sequence_length,)` = 32 | Accepts integer indices of preceding musical notes |
| 2 | **Embedding** | `input_dim=V`, `output_dim=64` | Embeds discrete pitch categories into continuous vector space |
| 3 | **LSTM 1** | `units=128`, `return_sequences=True` | Extracts temporal features and sequential motifs |
| 4 | **Dropout 1** | `rate=0.2` | Regularization to prevent overfitting |
| 5 | **LSTM 2** | `units=128`, `return_sequences=False` | Encodes overall temporal context into a dense latent state |
| 6 | **Dropout 2** | `rate=0.2` | Secondary regularization |
| 7 | **Dense** | `units=128`, `activation='relu'` | Non-linear feature representation |
| 8 | **Dense Output** | `units=vocab_size`, `activation='softmax'` | Probability distribution over all possible musical tokens |

### Temperature-Scaled Sampling Formula
Rather than always selecting the highest-probability note (which leads to repetitive loops), we scale logits by temperature $T$:

$$P(x_i) = \frac{\exp(\log(p_i) / T)}{\sum_j \exp(\log(p_j) / T)}$$

- **$T = 0.4$ (Conservative):** Strong adherence to dominant training melodies.
- **$T = 0.8$ (Balanced):** Harmonious combination of familiarity and novelty.
- **$T = 1.2$ (Experimental):** High entropy; discovers unexpected note intervals.

---

## 💻 Technologies & Libraries Used

- **Programming Language**: Python 3.10+
- **Deep Learning**: TensorFlow 2.x / Keras
- **Symbolic Musicology**: Music21
- **Numerical Processing**: NumPy
- **Interactive Web App**: Streamlit
- **Audio Synthesis**: WebAudio API (HTML5 Oscillator Synthesis with ADSR envelopes)

---

## 📁 Project Structure

```
CodeAlpha_MusicGeneration/
│
├── app.py                  # Interactive Streamlit Web Application
├── train_model.py          # Data pipeline, preprocessing, & LSTM training routine
├── generate_music.py       # Inference script & MIDI reconstruction engine
├── requirements.txt        # Minimal, verified project dependencies
├── README.md               # Complete project documentation
├── .gitignore              # Ignored files (virtualenvs, cache, checkpoints)
│
├── data/                   # Directory for training MIDI files (.mid)
│   └── README.md           # Dataset formatting guide & public sources
│
├── models/                 # Directory where trained weights & vocab are saved
│   └── README.md           # Model artifacts description
│
└── generated_music/        # Destination for newly composed MIDI files
    └── README.md           # Playback instructions & file format details
```

---

## 🎼 Dataset Requirements & Preprocessing

- **Format**: Standard MIDI (`.mid` or `.midi`).
- **Recommendation**: Clean, solo-instrument files (such as piano sonatas, folk tunes, or video game melodies) yield the cleanest learned structures.
- **Location**: Place your `.mid` files directly into the `data/` directory.

### Quick Starter Dataset
If you do not have MIDI files immediately downloaded, the project includes an automated starter generator:
```bash
python train_model.py --create-samples
```
This generates 3 classical pieces (*Ode to Joy theme*, *Canon in D progression*, and *Twinkle theme*) directly into `data/`.

---

## 🚀 Installation & Setup

### 1. Clone or Open the Workspace
```bash
cd CodeAlpha_MusicGeneration
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

1. Ensure training MIDI files are present in `data/`.
2. Run the training script:
```bash
python train_model.py --epochs 25 --batch_size 32
```

### Optional Training Arguments:
- `--epochs`: Number of training epochs (default: `25`).
- `--batch_size`: Batch size (default: `32`).
- `--seq_len`: Note window context length (default: `32`).
- `--data_dir`: Custom directory for MIDI files (default: `data`).
- `--create-samples`: Auto-generates starter classical tracks if `data/` is empty.

Upon successful completion, two files are created:
- `models/music_lstm_model.keras`: Trained weights and neural network graph.
- `models/notes_vocab.json`: Note dictionary, integer mappings, and training metadata.

---

## 🎹 Generating Music (CLI)

Once the model is trained, generate a new MIDI composition using:
```bash
python generate_music.py --notes 50 --temperature 0.8 --tempo 120 --instrument Piano
```

### Command-Line Options:
- `--notes`: Number of musical notes to synthesize (e.g. `50`).
- `--temperature`: Creativity level from `0.2` to `1.4` (default: `0.8`).
- `--tempo`: Beats per minute (BPM) (default: `120`).
- `--instrument`: Timbre (`Piano`, `Acoustic Guitar`, `Violin`, `Flute`, `Harpsichord`, etc.).
- `--output_dir`: Output directory (default: `generated_music`).

The output is saved as a timestamped MIDI file (e.g. `generated_music/music_gen_20261002_000500.mid`).

---

## 🌐 Running the Streamlit Web Application

Launch the interactive dashboard with:
```bash
streamlit run app.py
```

### Web Application Features:
- **Model Status Indicator**: Detects whether the model is trained. If not trained, provides clear step-by-step guidance without crashing.
- **Real-Time Generation**: Customize note count, temperature, tempo, and instrument with interactive sliders.
- **In-Browser WebAudio Synthesizer**: Listen to the generated track directly in Chrome/Edge/Firefox with zero external dependencies.
- **Interactive Piano-Roll**: Scatter chart displaying note pitch numbers across sequential steps.
- **MIDI Downloader**: Instant download button to save the `.mid` file locally.
- **Model Architecture Inspector**: Detailed interactive breakdown of layers, activation functions, and formulas.
- **Generated Library Explorer**: Browse and download all past compositions stored in `generated_music/`.

---

## 🔁 Example Workflow

```bash
# Step 1: Install packages
pip install -r requirements.txt

# Step 2: Generate starter MIDI samples into data/
python train_model.py --create-samples

# Step 3: Train the LSTM neural network
python train_model.py --epochs 25

# Step 4: Generate a composition via command line
python generate_music.py --notes 60 --temperature 0.85

# Step 5: Launch the Streamlit Web Application
streamlit run app.py
```

---

## ⚠️ Limitations

1. **Monophonic & Simplified Polyphony**: The model tokenizes single notes and simultaneous chords. Complex contrapuntal tracks with independent polyphonic voices on different rhythms are simplified into sequential events.
2. **Fixed Duration Quantization**: Note lengths are set to standardized beat subdivisions (e.g., eighth or quarter notes) rather than expressive human rubato timing.
3. **Dataset Dependency**: The musical quality of generated pieces directly mirrors the diversity and coherence of the training data provided in `data/`.

---

## 🔮 Future Improvements

- **Polyphonic Multi-Track Architecture**: Extend the model with separate streams for melody, bassline, and percussion.
- **Attention Mechanisms (Transformers / Music Transformer)**: Implement Relative Multi-Head Attention to capture minute structure across hundreds of measures.
- **Variable Duration & Velocity Encoding**: Tokenize dynamic volume (MIDI velocity) and flexible note durations alongside pitch.
- **Direct SoundFont FluidSynth Audio Rendering**: Integrate WAV/MP3 rendering alongside MIDI.

---

## 👤 Author & Acknowledgements

- **Developer**: Artificial Intelligence Intern
- **Internship**: [CodeAlpha](https://codealpha.tech/)
- **Domain**: Artificial Intelligence Internship
- **Task**: Task 3 — AI Music Generation

*Special thanks to the CodeAlpha mentorship team for the project guidelines and the open-source creators of `music21`, `TensorFlow`, and `Streamlit`.*
