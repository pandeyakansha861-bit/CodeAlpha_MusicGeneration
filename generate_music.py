"""
=============================================================================
CodeAlpha Artificial Intelligence Internship - Task 3: AI Music Generation
File: generate_music.py
Description: Music inference module. Loads a trained LSTM model, generates
             a sequence of musical tokens using temperature-controlled
             probabilistic sampling, and converts them into a valid MIDI file.
=============================================================================
"""

import os
import json
import argparse
import datetime
from pathlib import Path

import numpy as np
import music21 as m21
import tensorflow as tf


INSTRUMENT_MAP = {
    "Piano": m21.instrument.Piano,
    "Electric Piano": m21.instrument.ElectricPiano,
    "Acoustic Guitar": m21.instrument.AcousticGuitar,
    "Electric Guitar": m21.instrument.ElectricGuitar,
    "Violin": m21.instrument.Violin,
    "Flute": m21.instrument.Flute,
    "Clarinet": m21.instrument.Clarinet,
    "Trumpet": m21.instrument.Trumpet,
    "Saxophone": m21.instrument.AltoSaxophone,
    "Harpsichord": m21.instrument.Harpsichord,
    "Church Bells": m21.instrument.TubularBells
}


def load_model_and_vocab(models_dir: str = "models"):
    """
    Loads the trained Keras LSTM model and associated vocabulary metadata.
    Raises FileNotFoundError with friendly instructions if not trained yet.
    """
    model_path = os.path.join(models_dir, "music_lstm_model.keras")
    vocab_path = os.path.join(models_dir, "notes_vocab.json")

    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Trained model or vocabulary metadata not found in '{models_dir}'.\n"
            f"Please train the model first by running:\n"
            f"   python train_model.py\n"
            f"or verify that '{model_path}' exists."
        )

    print(f"Loading trained neural network from '{model_path}'...")
    model = tf.keras.models.load_model(model_path)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_meta = json.load(f)

    # Note: JSON keys are always strings, so convert keys in int_to_note back to integers
    int_to_note = {int(k): v for k, v in vocab_meta["int_to_note"].items()}
    vocab_meta["int_to_note"] = int_to_note

    return model, vocab_meta


def sample_with_temperature(probabilities: np.ndarray, temperature: float = 0.8) -> int:
    """
    Applies temperature-scaled softmax sampling to control prediction entropy:
      - Low temperature (< 0.6): High confidence, coherent, repeats dominant motifs.
      - Medium temperature (0.7 - 0.9): Ideal balance of harmonic coherence and creativity.
      - High temperature (> 1.0): High entropy, adventurous, diverse transitions.
    """
    temperature = max(0.05, float(temperature))
    predictions = np.asarray(probabilities, dtype=np.float64)

    # Avoid log(0)
    predictions = np.clip(predictions, 1e-12, 1.0)
    log_preds = np.log(predictions) / temperature
    exp_preds = np.exp(log_preds - np.max(log_preds))  # Numerical stability
    probs = exp_preds / np.sum(exp_preds)

    # Ensure sum is strictly 1.0 for np.random.choice
    probs = probs / np.sum(probs)
    return int(np.random.choice(len(probs), p=probs))


def generate_notes_sequence(model: tf.keras.Model,
                            vocab_meta: dict,
                            num_notes: int = 50,
                            temperature: float = 0.8,
                            seed_sequence: list = None) -> list:
    """
    Generates a sequential list of musical notes/chords using autoregressive
    inference with the trained LSTM.
    """
    vocab = vocab_meta["vocab"]
    note_to_int = vocab_meta["note_to_int"]
    int_to_note = vocab_meta["int_to_note"]
    seq_length = vocab_meta["sequence_length"]
    vocab_size = vocab_meta["vocab_size"]

    # 1. Initialize Seed Sequence
    if seed_sequence is not None and len(seed_sequence) >= seq_length:
        pattern = [note_to_int[n] if n in note_to_int else 0 for n in seed_sequence[-seq_length:]]
    else:
        # Pick a random starting window from vocabulary
        start_indices = np.random.choice(vocab_size, size=seq_length, replace=True)
        pattern = list(start_indices)

    generated_tokens = []
    print(f"Generating {num_notes} musical notes (Temperature: {temperature})...")

    # 2. Autoregressive Loop
    for step in range(num_notes):
        # Shape: (1, seq_length)
        input_tensor = np.array([pattern], dtype=np.int32)

        # Forward pass: predict probabilities over next note
        probabilities = model.predict(input_tensor, verbose=0)[0]

        # Sample next index using temperature
        predicted_idx = sample_with_temperature(probabilities, temperature=temperature)
        predicted_note = int_to_note[predicted_idx]

        generated_tokens.append(predicted_note)

        # Slide the window forward
        pattern.append(predicted_idx)
        pattern = pattern[1:]

    return generated_tokens


def convert_notes_to_midi(notes_sequence: list,
                          output_path: str,
                          tempo_bpm: int = 120,
                          note_duration: float = 0.5,
                          instrument_name: str = "Piano") -> str:
    """
    Transforms a list of note/chord string representations into a valid MIDI file.
    
    Parameters:
      - notes_sequence: list of strings (e.g., ['C4', 'E4.G4', 'D4', ...])
      - output_path: destination file path for .mid
      - tempo_bpm: beats per minute (e.g., 120)
      - note_duration: quarterLength per note (0.5 = eighth note, 1.0 = quarter note)
      - instrument_name: key from INSTRUMENT_MAP
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stream = m21.stream.Stream()

    # Set Tempo
    stream.append(m21.tempo.MetronomeMark(number=tempo_bpm))

    # Set Instrument
    instrument_cls = INSTRUMENT_MAP.get(instrument_name, m21.instrument.Piano)
    stream.append(instrument_cls())

    current_offset = 0.0

    for item in notes_sequence:
        # Pattern is a chord (contains '.' or is numeric group)
        if ("." in item) or item.isdigit():
            chord_parts = item.split(".")
            chord_notes = []
            for p in chord_parts:
                try:
                    # Integer MIDI pitch number (e.g. 60)
                    new_n = m21.note.Note(int(p))
                except ValueError:
                    # String pitch name with octave (e.g. 'C4')
                    new_n = m21.note.Note(p)
                chord_notes.append(new_n)

            chord_element = m21.chord.Chord(chord_notes)
            chord_element.quarterLength = note_duration
            chord_element.offset = current_offset
            stream.append(chord_element)

        # Pattern is a single note
        else:
            try:
                note_element = m21.note.Note(item)
            except Exception:
                # Fallback to Middle C if an unrecognized token occurs
                note_element = m21.note.Note("C4")
            note_element.quarterLength = note_duration
            note_element.offset = current_offset
            stream.append(note_element)

        current_offset += note_duration

    # Write out valid MIDI file
    stream.write("midi", fp=output_path)
    return output_path


def generate_music(num_notes: int = 50,
                   temperature: float = 0.8,
                   tempo_bpm: int = 120,
                   instrument_name: str = "Piano",
                   note_duration: float = 0.5,
                   output_dir: str = "generated_music",
                   models_dir: str = "models",
                   custom_filename: str = None) -> tuple:
    """
    End-to-end music synthesis routine:
      1. Loads trained model and vocabulary
      2. Generates sequence of notes
      3. Compiles and saves MIDI file
    Returns: (output_file_path, generated_notes_list)
    """
    model, vocab_meta = load_model_and_vocab(models_dir=models_dir)

    # Autoregressively generate notes
    generated_notes = generate_notes_sequence(
        model=model,
        vocab_meta=vocab_meta,
        num_notes=num_notes,
        temperature=temperature
    )

    # Create timestamped filename
    if not custom_filename:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_filename = f"music_gen_{timestamp}.mid"

    if not custom_filename.endswith(".mid") and not custom_filename.endswith(".midi"):
        custom_filename += ".mid"

    output_path = os.path.join(output_dir, custom_filename)

    # Convert to standard MIDI
    saved_path = convert_notes_to_midi(
        notes_sequence=generated_notes,
        output_path=output_path,
        tempo_bpm=tempo_bpm,
        note_duration=note_duration,
        instrument_name=instrument_name
    )

    print(f"\n[Success] Music generated successfully!")
    print(f"  - Notes generated  : {len(generated_notes)}")
    print(f"  - Saved to         : {saved_path}")
    print(f"  - Tempo            : {tempo_bpm} BPM")
    print(f"  - Instrument       : {instrument_name}\n")

    return saved_path, generated_notes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate algorithmic MIDI music using the trained LSTM model."
    )
    parser.add_argument(
        "--notes", type=int, default=50,
        help="Number of musical notes to generate (default: 50)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (0.2=conservative, 1.2=creative, default: 0.8)"
    )
    parser.add_argument(
        "--tempo", type=int, default=120,
        help="Playback tempo in BPM (default: 120)"
    )
    parser.add_argument(
        "--instrument", type=str, default="Piano",
        choices=list(INSTRUMENT_MAP.keys()),
        help="Instrument timbre to assign in the MIDI stream (default: Piano)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="generated_music",
        help="Directory to save generated MIDI files (default: generated_music)"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models",
        help="Directory where trained model and vocab are stored (default: models)"
    )

    args = parser.parse_args()

    try:
        generate_music(
            num_notes=args.notes,
            temperature=args.temperature,
            tempo_bpm=args.tempo,
            instrument_name=args.instrument,
            output_dir=args.output_dir,
            models_dir=args.models_dir
        )
    except FileNotFoundError as err:
        print(f"\n[Error] {err}")
