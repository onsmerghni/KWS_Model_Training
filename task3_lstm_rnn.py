"""
task3_lstm_rnn.py
-----------------
Task 3 — Upgrade: RNN / LSTM
The Concept: Treat the MFCCs as a sequence of features where
past context matters. Each time frame (row of the MFCC matrix)
is one step in the sequence, carrying 13 frequency features.

Reshape: (51, 13, 1)  ->  (51, 13)   [51 time steps, 13 features each]

Pipeline A — LSTM:
    MFCC (51, 13) -> LSTM(64, return_seq=True) -> LSTM(32) -> Dense(32) -> Softmax(2)

Pipeline B — GRU (lighter alternative):
    MFCC (51, 13) -> GRU(64, return_seq=True)  -> GRU(32)  -> Dense(32) -> Softmax(2)

Run:
    python task3_lstm_rnn.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from data_loader import prepare_dataset

plt.style.use("seaborn-v0_8-whitegrid")
tf.get_logger().setLevel("ERROR")

print("=" * 55)
print("  TASK 3 — RNN / LSTM")
print("=" * 55)


# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/4] Loading & preprocessing data...")
(
    X_train, X_val,
    y_train, y_val,
    class_weight_dict,
    INPUT_SHAPE, NUM_CLASSES,
) = prepare_dataset()
# INPUT_SHAPE = (51, 13, 1)

# For RNN we need (51, 13) — squeeze the channel dimension
X_train_seq = X_train.squeeze(-1)   # (N, 51, 13)
X_val_seq   = X_val.squeeze(-1)     # (N, 51, 13)
SEQ_SHAPE   = X_train_seq.shape[1:] # (51, 13)

print(f"  Sequence input shape: {SEQ_SHAPE}  (time_steps=51, features=13)")


# ── 2. Build models ───────────────────────────────────────────
print("\n[2/4] Building LSTM and GRU models...")

def build_lstm(seq_shape, num_classes):
    """
    Two-layer stacked LSTM.
    seq_shape: (51, 13)
    """
    model = models.Sequential([
        layers.Input(shape=seq_shape, name="mfcc_sequence"),

        # First LSTM — return full sequence for stacking
        layers.LSTM(64, return_sequences=True, name="lstm_1"),
        layers.Dropout(0.3, name="drop_1"),

        # Second LSTM — return only final hidden state
        layers.LSTM(32, return_sequences=False, name="lstm_2"),
        layers.Dropout(0.3, name="drop_2"),

        # Classification head
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="LSTM_Model")
    return model


def build_gru(seq_shape, num_classes):
    """
    Two-layer stacked GRU (lighter than LSTM, similar performance).
    seq_shape: (51, 13)
    """
    model = models.Sequential([
        layers.Input(shape=seq_shape, name="mfcc_sequence"),

        layers.GRU(64, return_sequences=True, name="gru_1"),
        layers.Dropout(0.3, name="drop_1"),

        layers.GRU(32, return_sequences=False, name="gru_2"),
        layers.Dropout(0.3, name="drop_2"),

        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="GRU_Model")
    return model


lstm_model = build_lstm(SEQ_SHAPE, NUM_CLASSES)
gru_model  = build_gru(SEQ_SHAPE, NUM_CLASSES)

for m in [lstm_model, gru_model]:
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

print("\n── LSTM ──")
lstm_model.summary()
print("\n── GRU ──")
gru_model.summary()


# ── 3. Train LSTM ─────────────────────────────────────────────
print("\n[3/4] Training LSTM...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

t0 = time.time()
history_lstm = lstm_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
lstm_time_per_epoch = (time.time() - t0) / len(history_lstm.history["loss"])

print("\n[3b/4] Training GRU...")
t0 = time.time()
history_gru = gru_model.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
gru_time_per_epoch = (time.time() - t0) / len(history_gru.history["loss"])


# Training curves — LSTM vs GRU
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history_lstm.history["val_accuracy"], label="LSTM val acc", marker="o", linewidth=2)
axes[0].plot(history_gru.history["val_accuracy"],  label="GRU val acc",  marker="s", linewidth=2)
axes[0].set_title("Task 3 — LSTM vs GRU: Val Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history_lstm.history["val_loss"], label="LSTM val loss", marker="o", linewidth=2)
axes[1].plot(history_gru.history["val_loss"],  label="GRU val loss",  marker="s", linewidth=2)
axes[1].set_title("Task 3 — LSTM vs GRU: Val Loss")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Task 3 — RNN / LSTM Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig("task3_training_curves.png", dpi=150)
plt.show()
print("  Saved: task3_training_curves.png")


# ── 4. Export best model to TFLite ───────────────────────────
best_model = lstm_model if (
    max(history_lstm.history["val_accuracy"]) >=
    max(history_gru.history["val_accuracy"])
) else gru_model

best_name = "LSTM" if best_model is lstm_model else "GRU"
print(f"\n[4/4] Saving best model ({best_name})...")

# NOTE: LSTM/GRU cannot be quantized to int8 TFLite with the standard converter
# because they use TensorList ops (Flex delegate) which the calibrator doesn't support.
# We save in two formats instead:

# Format 1: Keras native format (full precision, for fine-tuning / inference in Python)
keras_path = f"kws_task3_{best_name.lower()}.keras"
best_model.save(keras_path)
print(f"  Saved Keras model : {keras_path}  ({os.path.getsize(keras_path)/1024:.2f} KB)")

# Format 2: TFLite float32 (no quantization — avoids the Flex op calibration issue)
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    # No int8 quantization — float32 TFLite works fine for LSTM
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    tflite_path  = "kws_task3_lstm.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    tflite_size = os.path.getsize(tflite_path) / 1024
    print(f"  Saved TFLite (float32): {tflite_path}  ({tflite_size:.2f} KB)")
except Exception as e:
    tflite_size = None
    print(f"  TFLite export skipped (LSTM Flex ops not supported by calibrator): {e}")


# ── Results summary ───────────────────────────────────────────
lstm_acc = max(history_lstm.history["val_accuracy"])
gru_acc  = max(history_gru.history["val_accuracy"])

print("\n" + "=" * 55)
print("  TASK 3 RESULTS")
print("=" * 55)
print(f"  LSTM")
print(f"    Parameters   : {lstm_model.count_params():,}")
print(f"    Val Accuracy : {lstm_acc*100:.2f}%")
print(f"    Time / Epoch : {lstm_time_per_epoch:.1f} s")
print(f"  GRU")
print(f"    Parameters   : {gru_model.count_params():,}")
print(f"    Val Accuracy : {gru_acc*100:.2f}%")
print(f"    Time / Epoch : {gru_time_per_epoch:.1f} s")
print(f"  Best model     : {best_name}")
print(f"  TFLite Size    : {os.path.getsize(tflite_path)/1024:.2f} KB")
print("=" * 55)
print("\nWhy LSTM works well for audio:")
print("  - Processes MFCC frame by frame (left to right in time)")
print("  - Hidden state carries memory of previous frames")
print("  - Handles variable-length sequences naturally")
print("\n→ Next: Task 4 — DS-CNN (same accuracy, 10x fewer params)")
