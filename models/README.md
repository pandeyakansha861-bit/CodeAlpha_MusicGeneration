# Trained Models Directory (`models/`)

This directory stores the artifacts produced by `train_model.py`.

---

## 📦 Stored Artifacts

When you run `python train_model.py`, the following files are saved here:

1. **`music_lstm_model.keras`**:
   - The compiled TensorFlow/Keras neural network containing model weights, architecture, and optimizer state.
   - Built with an Embedding layer, dual stacked LSTM recurrent layers, dropout regularization, and Dense classification layers.

2. **`notes_vocab.json`**:
   - The categorical vocabulary mapping between musical pitch tokens (notes and chords) and numerical integer indices.
   - Contains metadata such as sequence length, total note count, vocabulary size, and training timestamp.

---

## ⚠️ Model Availability Note
By default, this repository does not include pre-trained model weights to adhere to the requirement:
> *"Do not pretend that a model was trained if training data is unavailable."*

To train the model:
1. Ensure MIDI files exist in `../data/`.
2. Run:
   ```bash
   python train_model.py
   ```
3. Once completed, `music_lstm_model.keras` and `notes_vocab.json` will appear in this directory.
