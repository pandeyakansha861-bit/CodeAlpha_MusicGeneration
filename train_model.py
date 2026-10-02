"""
=============================================================================
CodeAlpha Artificial Intelligence Internship - Task 3: AI Music Generation
File: train_model.py
Description: Data preprocessing pipeline, LSTM model construction, and
             training routine for algorithmic music composition using MIDI.
=============================================================================
"""

import os
import glob
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import music21 as m21
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def create_sample_midis(target_dir: str = "data") -> list:
    """
    Synthesize starter classical and melodic MIDI pieces into the target directory.
    This enables immediate local training and testing even without downloading
    external datasets.
    """
    os.makedirs(target_dir, exist_ok=True)
    created_files = []

    # 1. Classical Theme: Beethoven Ode to Joy motif
    theme_1_notes = [
        "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
        "C4", "C4", "D4", "E4", "E4", "D4", "D4",
        "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
        "C4", "C4", "D4", "E4", "D4", "C4", "C4",
        "D4", "D4", "E4", "C4", "D4", "E4", "F4", "E4", "C4",
        "D4", "E4", "F4", "E4", "D4", "C4", "D4", "G3",
        "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
        "C4", "C4", "D4", "E4", "D4", "C4", "C4"
    ]
    stream1 = m21.stream.Stream()
    stream1.append(m21.tempo.MetronomeMark(number=120))
    stream1.append(m21.instrument.Piano())
    for n in theme_1_notes:
        stream1.append(m21.note.Note(n, quarterLength=0.5))
    path1 = os.path.join(target_dir, "starter_melody_ode_to_joy.mid")
    stream1.write("midi", fp=path1)
    created_files.append(path1)

    # 2. Pachelbel Canon in D Melodic Progression (Notes & Chords)
    theme_2_elements = [
        # Measure 1: D Major
        "D4.F#4.A4", "A3", "F#4", "D4",
        # Measure 2: B Minor
        "B3.D4.F#4", "F#3", "D4", "B3",
        # Measure 3: G Major
        "G3.B3.D4", "D3", "B3", "G3",
        # Measure 4: A Major
        "A3.C#4.E4", "E3", "C#4", "A3",
        # Repeat with melodic variations
        "F#4", "E4", "D4", "C#4", "B3", "A3", "B3", "C#4",
        "D4", "C#4", "B3", "A3", "G3", "F#3", "G3", "E3",
        "D3.F#3.A3", "F#4", "A4", "D5", "C#5", "B4", "A4", "G4",
        "F#4", "E4", "D4", "E4", "F#4", "G4", "A4", "D4.F#4.A4"
    ]
    stream2 = m21.stream.Stream()
    stream2.append(m21.tempo.MetronomeMark(number=100))
    stream2.append(m21.instrument.Piano())
    for item in theme_2_elements:
        if "." in item:
            pitches = item.split(".")
            chord_obj = m21.chord.Chord(pitches, quarterLength=1.0)
            stream2.append(chord_obj)
        else:
            stream2.append(m21.note.Note(item, quarterLength=0.5))
    path2 = os.path.join(target_dir, "starter_harmony_canon.mid")
    stream2.write("midi", fp=path2)
    created_files.append(path2)

    # 3. Folk / Lullaby Melodic Sequence: Twinkle Star Theme
    theme_3_notes = [
        "C4", "C4", "G4", "G4", "A4", "A4", "G4",
        "F4", "F4", "E4", "E4", "D4", "D4", "C4",
        "G4", "G4", "F4", "F4", "E4", "E4", "D4",
        "G4", "G4", "F4", "F4", "E4", "E4", "D4",
        "C4", "C4", "G4", "G4", "A4", "A4", "G4",
        "F4", "F4", "E4", "E4", "D4", "D4", "C4",
        # Added octave variation
        "C5", "C5", "G4", "G4", "A4", "A4", "G4",
        "F4", "F4", "E4", "E4", "D4", "D4", "C4"
    ]
    stream3 = m21.stream.Stream()
    stream3.append(m21.tempo.MetronomeMark(number=110))
    stream3.append(m21.instrument.Piano())
    for n in theme_3_notes:
        stream3.append(m21.note.Note(n, quarterLength=0.5))
    path3 = os.path.join(target_dir, "starter_melody_twinkle.mid")
    stream3.write("midi", fp=path3)
    created_files.append(path3)

    return created_files


def extract_notes_from_midi(file_path: str) -> list:
    """
    Parses a single MIDI file using music21 and extracts sequential note
    pitch names (e.g., 'C4') and chords (e.g., 'C4.E4.G4').
    """
    notes = []
    try:
        midi_data = m21.converter.parse(file_path)
        # Flatten the score to inspect all notes and chords sequentially
        elements = midi_data.recurse().notes

        for element in elements:
            if isinstance(element, m21.note.Note):
                notes.append(str(element.pitch.nameWithOctave))
            elif isinstance(element, m21.chord.Chord):
                chord_str = ".".join(str(p.nameWithOctave) for p in element.pitches)
                notes.append(chord_str)
    except Exception as exc:
        print(f"  [Warning] Skipping malformed file '{os.path.basename(file_path)}': {exc}")
    return notes


def load_and_preprocess_dataset(data_dir: str = "data") -> list:
    """
    Searches for MIDI files in the given data directory and extracts notes
    across all valid files into a single training sequence.
    """
    print(f"\n[1/4] Scanning data directory: '{data_dir}'...")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Search for .mid and .midi files (case-insensitive)
    midi_files = glob.glob(os.path.join(data_dir, "*.mid")) + \
                 glob.glob(os.path.join(data_dir, "*.midi")) + \
                 glob.glob(os.path.join(data_dir, "*.MID"))

    if not midi_files:
        print("\n" + "=" * 70)
        print(" [!] No MIDI training files found in the 'data/' directory.")
        print(" To train the model, you can:")
        print("   1. Place your own .mid or .midi files into the 'data/' folder.")
        print("   2. Run `python train_model.py --create-samples` to auto-generate")
        print("      starter classical MIDI tracks for immediate training.")
        print("=" * 70 + "\n")
        return []

    print(f"  Found {len(midi_files)} MIDI file(s). Parsing musical elements...")
    all_notes = []
    for idx, file_path in enumerate(midi_files, start=1):
        extracted = extract_notes_from_midi(file_path)
        print(f"  ({idx}/{len(midi_files)}) {os.path.basename(file_path)}: {len(extracted)} notes/chords")
        all_notes.extend(extracted)

    print(f"  Total musical tokens extracted: {len(all_notes)}")
    return all_notes


def prepare_sequences(notes: list, sequence_length: int = 32):
    """
    Prepares input sequences (X) and target outputs (y) for supervised
    next-step prediction. Also creates categorical index mappings.
    """
    print("\n[2/4] Encoding musical tokens and building sequences...")

    # Unique sorted vocabulary of musical events (pitches & chords)
    vocab = sorted(list(set(notes)))
    vocab_size = len(vocab)
    print(f"  Unique vocabulary size: {vocab_size} distinct pitch/chord patterns")

    # Safeguard: check whether dataset has enough notes to create sequences
    if len(notes) <= sequence_length:
        raise ValueError(
            f"Insufficient notes in dataset ({len(notes)} notes). "
            f"Model requires at least {sequence_length + 1} notes to build training pairs. "
            f"Please add more MIDI files to 'data/'."
        )

    # Bidirectional mappings between note strings and integer indices
    note_to_int = {note: i for i, note in enumerate(vocab)}
    int_to_note = {i: note for i, note in enumerate(vocab)}

    network_input = []
    network_output = []

    # Slide a window of length `sequence_length` across the token sequence
    for i in range(len(notes) - sequence_length):
        seq_in = notes[i : i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    X = np.array(network_input, dtype=np.int32)
    y = np.array(network_output, dtype=np.int32)

    print(f"  Input sequences (X) shape : {X.shape}")
    print(f"  Target outputs (y) shape  : {y.shape}")

    vocab_metadata = {
        "vocab": vocab,
        "note_to_int": note_to_int,
        "int_to_note": {str(k): v for k, v in int_to_note.items()},
        "vocab_size": vocab_size,
        "sequence_length": sequence_length,
        "total_notes_trained": len(notes)
    }

    return X, y, vocab_metadata


def build_lstm_model(vocab_size: int, sequence_length: int = 32) -> tf.keras.Model:
    """
    Constructs a sequential deep learning architecture:
      - Embedding Layer: Maps discrete musical pitches into a continuous vector space
      - 2 Stacked LSTM Layers: Captures temporal melodic progressions and transitions
      - Dropout Layers: Mitigates overfitting on smaller musical corpora
      - Dense Output Layer: Predicts categorical probability distribution over vocabulary
    """
    print(f"\n[3/4] Building LSTM Neural Network Architecture...")
    embedding_dim = 64
    lstm_units = 128

    model = Sequential([
        Input(shape=(sequence_length,), name="input_note_sequence"),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="note_embedding"),
        LSTM(lstm_units, return_sequences=True, name="lstm_layer_1"),
        Dropout(0.2, name="dropout_1"),
        LSTM(lstm_units, return_sequences=False, name="lstm_layer_2"),
        Dropout(0.2, name="dropout_2"),
        Dense(128, activation="relu", name="dense_features"),
        Dense(vocab_size, activation="softmax", name="output_probabilities")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train(data_dir: str = "data",
          models_dir: str = "models",
          sequence_length: int = 32,
          epochs: int = 25,
          batch_size: int = 32,
          create_samples_if_empty: bool = False):
    """
    Main training workflow:
      1. Loads MIDI files from `data_dir`
      2. Extracts notes and encodes training pairs
      3. Compiles and trains the LSTM model
      4. Saves model weights and vocabulary metadata into `models_dir`
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Check if data directory is empty
    midi_files = glob.glob(os.path.join(data_dir, "*.mid")) + \
                 glob.glob(os.path.join(data_dir, "*.midi"))

    if not midi_files and create_samples_if_empty:
        print("  Notice: Data directory is empty. Generating starter sample MIDIs...")
        create_sample_midis(data_dir)

    # Step 1: Preprocess dataset
    notes = load_and_preprocess_dataset(data_dir)
    if not notes:
        return None

    # Step 2: Prepare sequence pairs
    # Adjust sequence length if total notes is small
    effective_seq_len = sequence_length
    if len(notes) < sequence_length + 1:
        if len(notes) > 16:
            effective_seq_len = 16
            print(f"  Note count ({len(notes)}) is smaller than default sequence length ({sequence_length}).")
            print(f"  Dynamically adjusting sequence length to {effective_seq_len}...")
        else:
            print(f"  [Error] Only {len(notes)} notes extracted. Please provide more MIDI files.")
            return None

    X, y, vocab_meta = prepare_sequences(notes, sequence_length=effective_seq_len)

    # Step 3: Build LSTM Neural Network
    model = build_lstm_model(vocab_size=vocab_meta["vocab_size"], sequence_length=effective_seq_len)

    # Step 4: Fit Model
    print(f"\n[4/4] Training LSTM for {epochs} epochs (Batch Size: {batch_size})...")

    # Early stopping to prevent overfitting if loss plateaus
    early_stopping = EarlyStopping(
        monitor="loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Step 5: Save Model & Vocabulary Artifacts
    model_path = os.path.join(models_dir, "music_lstm_model.keras")
    vocab_path = os.path.join(models_dir, "notes_vocab.json")

    model.save(model_path)
    print(f"\n  [Success] Trained model saved to: '{model_path}'")

    final_loss = float(history.history["loss"][-1])
    final_acc = float(history.history["accuracy"][-1]) if "accuracy" in history.history else 0.0

    vocab_meta["trained_at"] = datetime.datetime.now().isoformat()
    vocab_meta["final_loss"] = round(final_loss, 4)
    vocab_meta["final_accuracy"] = round(final_acc, 4)
    vocab_meta["epochs_trained"] = len(history.history["loss"])

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_meta, f, indent=2)
    print(f"  [Success] Vocabulary metadata saved to: '{vocab_path}'")

    print("\n" + "=" * 70)
    print(" [SUCCESS] Model training complete!")
    print(f"    - Final Loss     : {final_loss:.4f}")
    print(f"    - Final Accuracy : {final_acc * 100:.2f}%")
    print(f"    - Distinct Notes : {vocab_meta['vocab_size']}")
    print("\n You can now:")
    print("   1. Generate music via CLI: python generate_music.py")
    print("   2. Run the web UI        : streamlit run app.py")
    print("=" * 70 + "\n")

    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an LSTM neural network on MIDI note sequences."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Directory containing training .mid or .midi files (default: data)"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models",
        help="Directory to save trained model & vocab artifacts (default: models)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Number of training epochs (default: 25)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--seq_len", type=int, default=32,
        help="Sequence length window for context (default: 32)"
    )
    parser.add_argument(
        "--create-samples", action="store_true",
        help="Generate starter sample classical MIDI files into the data directory"
    )

    args = parser.parse_args()

    if args.create_samples:
        print("[*] Generating starter MIDI samples in 'data/' directory...")
        samples = create_sample_midis(args.data_dir)
        print(f"[*] Successfully created {len(samples)} sample files:")
        for s in samples:
            print(f"    - {s}")
        print("[*] You can now train the model using: python train_model.py\n")
    else:
        train(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            sequence_length=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            create_samples_if_empty=False
        )
