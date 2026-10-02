"""
=============================================================================
CodeAlpha Artificial Intelligence Internship - Task 3: AI Music Generation
File: app.py
Description: Interactive Streamlit Web Application for AI Music Generation.
             Features model status detection, hyperparameter tuning,
             autoregressive generation, interactive piano-roll visualization,
             in-browser WebAudio playback, and MIDI file downloads.
=============================================================================
"""

import os
import glob
import json
import base64
import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import music21 as m21

# Import generation and training utilities
from generate_music import (
    load_model_and_vocab,
    generate_notes_sequence,
    convert_notes_to_midi,
    INSTRUMENT_MAP
)
from train_model import create_sample_midis


# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Music Generator | CodeAlpha",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom CSS for Sleek Modern Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* Header Gradient Banner */
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.9) 0%, rgba(49, 46, 129, 0.8) 50%, rgba(67, 56, 202, 0.7) 100%);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF, #C7D2FE, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .hero-subtitle {
        color: #E0E7FF;
        font-size: 1.05rem;
        font-weight: 500;
        margin-bottom: 12px;
    }
    
    .badge-internship {
        display: inline-block;
        background: rgba(99, 102, 241, 0.25);
        border: 1px solid rgba(165, 180, 252, 0.4);
        color: #C7D2FE;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 0.5px;
    }
    
    /* Status Cards */
    .status-card-ready {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    
    .status-card-warning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    
    /* Note Badges */
    .note-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #E0E7FF;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 8px;
        margin: 3px;
        border-radius: 6px;
    }
    
    .chord-badge {
        display: inline-block;
        background: rgba(236, 72, 153, 0.18);
        border: 1px solid rgba(236, 72, 153, 0.4);
        color: #FCE7F3;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 3px 8px;
        margin: 3px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper Functions: State & Audio Synthesis
# ---------------------------------------------------------------------------
def check_model_availability(models_dir: str = "models") -> tuple:
    """Checks if the trained model and vocabulary exist."""
    model_path = os.path.join(models_dir, "music_lstm_model.keras")
    vocab_path = os.path.join(models_dir, "notes_vocab.json")
    if os.path.exists(model_path) and os.path.exists(vocab_path):
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            return True, vocab_data
        except Exception:
            return False, None
    return False, None


def get_note_frequencies(note_str: str) -> list:
    """Calculates frequency in Hz for a note or chord."""
    frequencies = []
    if "." in note_str or note_str.isdigit():
        for p in note_str.split("."):
            try:
                n = m21.note.Note(int(p)) if p.isdigit() else m21.note.Note(p)
                frequencies.append(round(n.pitch.frequency, 2))
            except Exception:
                pass
    else:
        try:
            n = m21.note.Note(note_str)
            frequencies.append(round(n.pitch.frequency, 2))
        except Exception:
            frequencies.append(261.63) # Default C4
    return frequencies


def render_webaudio_player(notes_list: list, tempo_bpm: int, note_duration: float):
    """
    Renders a self-contained WebAudio player that plays the generated musical
    notes directly in the user's browser using oscillator synthesis and ADSR envelopes.
    Requires no internet connection or external soundfont files!
    """
    # Build schedule payload
    events = []
    step_duration_sec = (60.0 / tempo_bpm) * (note_duration * 2.0)

    for idx, item in enumerate(notes_list):
        freqs = get_note_frequencies(item)
        events.append({
            "step": idx,
            "name": item,
            "freqs": freqs,
            "duration": round(step_duration_sec, 3)
        })

    events_json = json.dumps(events)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        .player-box {{
          background: #111827;
          border: 1px solid #374151;
          border-radius: 12px;
          padding: 16px 20px;
          color: #F3F4F6;
          font-family: system-ui, -apple-system, sans-serif;
          margin-top: 10px;
          margin-bottom: 12px;
        }}
        .controls {{
          display: flex;
          align-items: center;
          gap: 12px;
        }}
        .btn {{
          background: #4F46E5;
          color: white;
          border: none;
          padding: 8px 18px;
          border-radius: 8px;
          font-weight: 600;
          font-size: 0.9rem;
          cursor: pointer;
          transition: background 0.2s;
        }}
        .btn:hover {{
          background: #4338CA;
        }}
        .btn-stop {{
          background: #374151;
        }}
        .btn-stop:hover {{
          background: #4B5563;
        }}
        .status-text {{
          font-size: 0.85rem;
          color: #9CA3AF;
          font-family: monospace;
          margin-left: auto;
        }}
        .progress-bar-bg {{
          width: 100%;
          height: 6px;
          background: #1F2937;
          border-radius: 3px;
          margin-top: 12px;
          overflow: hidden;
        }}
        .progress-bar-fill {{
          height: 100%;
          width: 0%;
          background: #6366F1;
          transition: width 0.1s linear;
        }}
      </style>
    </head>
    <body>
      <div class="player-box">
        <div class="controls">
          <button id="playBtn" class="btn" onclick="togglePlay()">▶ Play Composition</button>
          <button id="stopBtn" class="btn btn-stop" onclick="stopMusic()">⏹ Stop</button>
          <span id="statusLabel" class="status-text">Ready to play</span>
        </div>
        <div class="progress-bar-bg">
          <div id="progressFill" class="progress-bar-fill"></div>
        </div>
      </div>

      <script>
        const events = {events_json};
        let audioCtx = null;
        let isPlaying = false;
        let timeouts = [];

        function initAudio() {{
          if (!audioCtx) {{
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            audioCtx = new AudioContext();
          }}
          if (audioCtx.state === 'suspended') {{
            audioCtx.resume();
          }}
        }}

        function playTone(freq, startTime, duration) {{
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();

          osc.type = 'triangle'; // Warm acoustic-like timbre
          osc.frequency.setValueAtTime(freq, startTime);

          // ADSR envelope
          const attack = 0.03;
          const release = 0.08;
          const stopTime = startTime + duration;

          gain.gain.setValueAtTime(0.001, startTime);
          gain.gain.linearRampToValueAtTime(0.2, startTime + attack);
          gain.gain.setValueAtTime(0.18, stopTime - release);
          gain.gain.exponentialRampToValueAtTime(0.0001, stopTime);

          osc.connect(gain);
          gain.connect(audioCtx.destination);

          osc.start(startTime);
          osc.stop(stopTime);
        }}

        function togglePlay() {{
          if (isPlaying) {{
            stopMusic();
            return;
          }}
          initAudio();
          isPlaying = true;
          document.getElementById('playBtn').textContent = '⏸ Pause';
          
          let currentTime = audioCtx.currentTime + 0.05;
          let totalDuration = 0;
          events.forEach(e => totalDuration += e.duration);

          const startTimeReal = Date.now();

          // Schedule WebAudio notes
          events.forEach((ev, i) => {{
            ev.freqs.forEach(freq => {{
              playTone(freq, currentTime, ev.duration);
            }});
            
            // Progress UI updates
            const delayMs = (currentTime - audioCtx.currentTime) * 1000;
            const t = setTimeout(() => {{
              if (!isPlaying) return;
              document.getElementById('statusLabel').textContent = 
                'Playing: ' + ev.name + ' (' + (i + 1) + '/' + events.length + ')';
              const pct = ((i + 1) / events.length) * 100;
              document.getElementById('progressFill').style.width = pct + '%';
            }}, Math.max(0, delayMs));
            
            timeouts.push(t);
            currentTime += ev.duration;
          }});

          // End of playback
          const totalMs = totalDuration * 1000;
          const endTimeout = setTimeout(() => {{
            stopMusic();
            document.getElementById('statusLabel').textContent = 'Finished';
            document.getElementById('progressFill').style.width = '100%';
          }}, totalMs + 200);
          timeouts.push(endTimeout);
        }}

        function stopMusic() {{
          isPlaying = false;
          timeouts.forEach(t => clearTimeout(t));
          timeouts = [];
          document.getElementById('playBtn').textContent = '▶ Play Composition';
          document.getElementById('statusLabel').textContent = 'Stopped';
          document.getElementById('progressFill').style.width = '0%';
        }}
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=95)


def render_piano_roll_chart(notes_list: list):
    """
    Visualizes the generated musical notes across time steps.
    """
    try:
        import pandas as pd
        
        data_points = []
        for step, token in enumerate(notes_list):
            if "." in token or token.isdigit():
                parts = token.split(".")
                for p in parts:
                    try:
                        n = m21.note.Note(int(p)) if p.isdigit() else m21.note.Note(p)
                        data_points.append({
                            "Step": step + 1,
                            "Pitch": n.pitch.nameWithOctave,
                            "MIDI Number": n.pitch.midi,
                            "Type": "Chord Note"
                        })
                    except Exception:
                        pass
            else:
                try:
                    n = m21.note.Note(token)
                    data_points.append({
                        "Step": step + 1,
                        "Pitch": n.pitch.nameWithOctave,
                        "MIDI Number": n.pitch.midi,
                        "Type": "Melody Note"
                    })
                except Exception:
                    pass

        if data_points:
            df = pd.DataFrame(data_points)
            st.scatter_chart(
                df,
                x="Step",
                y="MIDI Number",
                color="Type",
                height=260,
                use_container_width=True
            )
    except Exception as e:
        st.info("Piano roll chart not available.")


# ---------------------------------------------------------------------------
# Application Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="badge-internship">CodeAlpha Artificial Intelligence Internship • Task 3</div>
    <div class="hero-title">🎵 AI Music Generator</div>
    <div class="hero-subtitle">
        Algorithmic Music Composition using Long Short-Term Memory (LSTM) Recurrent Neural Networks
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Check Model Availability Status
# ---------------------------------------------------------------------------
model_ready, vocab_metadata = check_model_availability("models")


# ---------------------------------------------------------------------------
# Sidebar Configuration & Tools
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Generation Settings")

    if model_ready:
        st.success("🟢 Model: Loaded & Ready")
        num_notes = st.slider(
            "Length (Number of Notes)",
            min_value=20,
            max_value=150,
            value=50,
            step=5,
            help="Controls how many sequential musical notes/chords to generate."
        )

        temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.2,
            max_value=1.4,
            value=0.8,
            step=0.05,
            help="Low (<0.6) = conservative & repetitive; High (>1.0) = adventurous & varied."
        )

        tempo_bpm = st.slider(
            "Tempo (BPM)",
            min_value=60,
            max_value=180,
            value=120,
            step=5,
            help="Beats per minute for playback and MIDI export."
        )

        instrument = st.selectbox(
            "Instrument Timbre",
            options=list(INSTRUMENT_MAP.keys()),
            index=0,
            help="Sets the target MIDI instrument channel."
        )

        note_duration_map = {
            "Sixteenth Note (0.25)": 0.25,
            "Eighth Note (0.5)": 0.5,
            "Quarter Note (1.0)": 1.0,
            "Half Note (2.0)": 2.0
        }
        note_duration_label = st.selectbox(
            "Note Duration",
            options=list(note_duration_map.keys()),
            index=1,
            help="Base length per generated note element."
        )
        note_duration = note_duration_map[note_duration_label]

        seed_mode = st.radio(
            "Seed Initialization",
            options=["Random from Vocabulary", "First Training Motif"],
            index=0,
            help="Defines how the initial LSTM input window is primed."
        )
    else:
        st.warning("🟠 Model Not Trained")
        st.info("Train the model using `python train_model.py` to enable generation controls.")

    st.markdown("---")
    st.markdown("### 🛠️ Quick Dataset Tools")

    if st.button("✨ Generate Starter MIDI in data/"):
        with st.spinner("Generating classical starter MIDI tracks..."):
            created = create_sample_midis("data")
            st.success(f"Created {len(created)} sample files in 'data/'!")
            for c in created:
                st.write(f"- `{os.path.basename(c)}`")
            st.info("Now run `python train_model.py` in your terminal to train.")

    st.markdown("---")
    st.markdown("### 📖 About Project")
    st.caption(
        "**CodeAlpha AI Internship - Task 3**\n\n"
        "This project implements an end-to-end deep learning pipeline for symbolic music generation. "
        "It uses `music21` to parse MIDI pitches and chords, normalizes them into token sequences, "
        "and trains a multi-layer LSTM neural network to predict harmonic musical progressions."
    )


# ---------------------------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------------------------
tab_generate, tab_architecture, tab_dataset, tab_history = st.tabs([
    "🎹 Generate Music",
    "🧠 Model Architecture",
    "📂 Training Data & Setup",
    "🎶 Generated Library"
])


# ===========================================================================
# TAB 1: GENERATE MUSIC
# ===========================================================================
with tab_generate:
    if not model_ready:
        st.markdown("""
        <div class="status-card-warning">
            <h3 style="margin-top:0; color:#F59E0B;">⚠️ No Trained Model Found</h3>
            <p>
                In compliance with project specifications, this system <b>does not fabricate pre-trained weights</b>.
                Before generating music, the LSTM model must be trained on MIDI data.
            </p>
            <h4 style="margin-bottom:8px; color:#F3F4F6;">Quick 3-Step Setup:</h4>
            <ol style="margin-top:4px; line-height: 1.8;">
                <li><b>Add MIDI files:</b> Place <code>.mid</code> files into the <code>data/</code> folder (or click <i>'Generate Starter MIDI in data/'</i> in the sidebar).</li>
                <li><b>Train the neural network:</b> Run the following command in your terminal:
                    <pre style="background:#1F2937; padding:10px 14px; border-radius:8px; margin: 8px 0; color:#A7F3D0;">python train_model.py</pre>
                </li>
                <li><b>Refresh this page:</b> Once training completes, the model will be automatically detected and ready for generation!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Display Model Metrics Overview
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Model Architecture", "Dual-LSTM + Embedding")
        with col_m2:
            st.metric("Vocabulary Size", f"{vocab_metadata.get('vocab_size', 'N/A')} notes")
        with col_m3:
            st.metric("Context Window", f"{vocab_metadata.get('sequence_length', 32)} tokens")
        with col_m4:
            trained_date = vocab_metadata.get('trained_at', 'Recently')
            if "T" in trained_date:
                trained_date = trained_date.split("T")[0]
            st.metric("Training Status", "Trained", trained_date)

        st.markdown("---")

        # Action Button
        generate_col, info_col = st.columns([1, 2])
        with generate_col:
            generate_clicked = st.button("🎼 Generate Musical Composition", type="primary", use_container_width=True)

        if generate_clicked:
            try:
                with st.spinner("🤖 Neural network composing new melody..."):
                    # Load model
                    model, meta = load_model_and_vocab("models")

                    # Handle seed
                    seed_seq = None
                    if seed_mode == "First Training Motif" and meta.get("vocab"):
                        seed_seq = meta["vocab"][:meta.get("sequence_length", 32)]

                    # Generate notes
                    generated_notes = generate_notes_sequence(
                        model=model,
                        vocab_meta=meta,
                        num_notes=num_notes,
                        temperature=temperature,
                        seed_sequence=seed_seq
                    )

                    # Output filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_filename = f"music_gen_{timestamp}.mid"
                    os.makedirs("generated_music", exist_ok=True)
                    out_filepath = os.path.join("generated_music", out_filename)

                    # Convert to MIDI
                    convert_notes_to_midi(
                        notes_sequence=generated_notes,
                        output_path=out_filepath,
                        tempo_bpm=tempo_bpm,
                        note_duration=note_duration,
                        instrument_name=instrument
                    )

                    # Store in session state
                    st.session_state["latest_notes"] = generated_notes
                    st.session_state["latest_file"] = out_filepath
                    st.session_state["latest_bpm"] = tempo_bpm
                    st.session_state["latest_duration"] = note_duration

                st.success(f"🎉 Composition generated successfully! ({len(generated_notes)} notes)")

            except Exception as e:
                st.error(f"Generation error: {str(e)}")

        # Display Results if Available
        if "latest_notes" in st.session_state and "latest_file" in st.session_state:
            latest_notes = st.session_state["latest_notes"]
            latest_file = st.session_state["latest_file"]
            latest_bpm = st.session_state.get("latest_bpm", 120)
            latest_duration = st.session_state.get("latest_duration", 0.5)

            st.markdown("#### 🎧 Listen to Composition (In-Browser WebAudio)")
            render_webaudio_player(latest_notes, latest_bpm, latest_duration)

            # Download Button
            with open(latest_file, "rb") as f:
                midi_bytes = f.read()

            st.download_button(
                label=f"⬇️ Download MIDI File ({os.path.basename(latest_file)})",
                data=midi_bytes,
                file_name=os.path.basename(latest_file),
                mime="audio/midi",
                type="secondary"
            )

            st.markdown("#### 📈 Note Roll Visualization")
            render_piano_roll_chart(latest_notes)

            # Note Sequence Badges
            with st.expander("🔍 Inspect Generated Sequence Tokens", expanded=False):
                html_badges = []
                for n in latest_notes:
                    cls = "chord-badge" if ("." in n or n.isdigit()) else "note-badge"
                    html_badges.append(f"<span class='{cls}'>{n}</span>")
                st.markdown("".join(html_badges), unsafe_allow_html=True)


# ===========================================================================
# TAB 2: MODEL ARCHITECTURE
# ===========================================================================
with tab_architecture:
    st.markdown("### 🧠 Deep Learning Architecture & Mathematical Pipeline")
    st.markdown("""
    The sequence generator utilizes a recurrent neural network architecture designed
    specifically for autoregressive symbolic music modeling.
    """)

    col_arch1, col_arch2 = st.columns([1, 1])

    with col_arch1:
        st.markdown("""
        #### 🏗️ Layer-by-Layer Architecture
        
        1. **Input Layer**:
           - Accepts integer token sequences of shape `(batch_size, sequence_length)`.
           - Default context window: `32 notes`.
        
        2. **Embedding Layer (`dim=64`)**:
           - Maps discrete categorical musical pitch classes into a continuous 64-dimensional latent space.
           - Allows the network to learn harmonic affinities (e.g., fifths, octaves, minor/major third relationships).
        
        3. **LSTM Layer 1 (`units=128`, `return_sequences=True`)**:
           - Captures temporal dependencies across musical time steps.
           - Gating mechanisms (Forget, Input, Cell, Output) prevent the vanishing gradient problem.
        
        4. **Dropout Regularization (`rate=0.2`)**:
           - Randomly drops 20% of activations to prevent co-adaptation and overfitting on smaller datasets.
        
        5. **LSTM Layer 2 (`units=128`, `return_sequences=False`)**:
           - Consolidates recurrence into a high-level representation vector of the current musical context.
        
        6. **Dropout Regularization (`rate=0.2`)**:
           - Secondary regularization layer.
        
        7. **Dense Layer (`units=128`, `activation='relu'`)**:
           - Non-linear feature transformation before the final projection.
        
        8. **Output Layer (`units=vocab_size`, `activation='softmax'`)**:
           - Outputs categorical probability distribution across all known musical pitches and chords.
        """)

    with col_arch2:
        st.markdown("""
        #### 🎲 Sampling With Temperature
        
        During inference, selecting the maximum probability token greedily (`argmax`) leads to repetitive loops. 
        Instead, we apply **Boltzmann temperature scaling**:
        """)
        st.latex(r"P(x_i) = \frac{\exp(\log(p_i) / T)}{\sum_j \exp(\log(p_j) / T)}")
        st.markdown("""
        - **$T < 0.6$ (Cold / Conservative):** Sharpen probabilities towards the most frequent notes. Great for strict melodies.
        - **$T = 0.8$ (Balanced):** Balances melodic coherence with harmonic exploration.
        - **$T > 1.0$ (Warm / Experimental):** Flattens the distribution, allowing rare transitions and avant-garde note choices.
        """)

        st.markdown("""
        #### 🎵 Tokenization & Representation
        - **Single Notes:** Encoded as scientific pitch notation strings (e.g., `C4`, `D#5`, `F3`).
        - **Chords:** Encoded as period-separated pitch names (e.g., `C4.E4.G4`).
        - **Inversion:** When exporting, tokens are mapped back to `music21.note.Note` and `music21.chord.Chord` objects.
        """)


# ===========================================================================
# TAB 3: TRAINING DATA & PIPELINE
# ===========================================================================
with tab_dataset:
    st.markdown("### 📂 Training Data Management")
    st.markdown("""
    Training files are stored in the local `data/` directory. The preprocessing pipeline
    automatically parses all valid `.mid` and `.midi` files.
    """)

    # List current files in data/
    existing_midi = glob.glob(os.path.join("data", "*.mid")) + glob.glob(os.path.join("data", "*.midi"))
    
    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("#### 📁 Files Currently in `data/`")
        if existing_midi:
            st.success(f"Found {len(existing_midi)} MIDI file(s) ready for training:")
            for f in existing_midi:
                size_kb = os.path.getsize(f) / 1024
                st.write(f"- 📄 `{os.path.basename(f)}` ({size_kb:.1f} KB)")
        else:
            st.warning("No MIDI files currently found in `data/`.")
            st.info("You can click the button below to generate starter tracks.")

        if st.button("✨ Create Starter Classical MIDIs", key="tab_create_btn"):
            created = create_sample_midis("data")
            st.success(f"Created {len(created)} sample files!")
            st.rerun()

    with col_d2:
        st.markdown("#### 🚀 How to Run Training")
        st.markdown("""
        Open your terminal in the project directory and execute:
        ```bash
        python train_model.py --epochs 25 --batch_size 32
        ```
        
        **Available Command-Line Options:**
        - `--data_dir`: Custom path to MIDI data (default: `data`)
        - `--epochs`: Number of iterations over the dataset (default: `25`)
        - `--batch_size`: Number of sequence pairs per gradient step (default: `32`)
        - `--seq_len`: Sliding window sequence length (default: `32`)
        - `--create-samples`: Auto-generates starter classical tracks
        """)


# ===========================================================================
# TAB 4: GENERATED MUSIC LIBRARY
# ===========================================================================
with tab_history:
    st.markdown("### 🎶 Generated Compositions Library")
    
    gen_files = sorted(
        glob.glob(os.path.join("generated_music", "*.mid")),
        key=os.path.getmtime,
        reverse=True
    )

    if gen_files:
        st.info(f"Found {len(gen_files)} generated track(s) in `generated_music/`:")
        for g in gen_files:
            fname = os.path.basename(g)
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(g)).strftime("%Y-%m-%d %H:%M:%S")
            fsize = os.path.getsize(g) / 1024

            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.write(f"🎵 **{fname}** — `{mtime}` ({fsize:.1f} KB)")
            with col_h2:
                with open(g, "rb") as bf:
                    data = bf.read()
                st.download_button(
                    label="⬇️ Download",
                    data=data,
                    file_name=fname,
                    mime="audio/midi",
                    key=f"dl_{fname}"
                )
    else:
        st.write("No compositions generated yet. Use the **Generate Music** tab to create one!")
