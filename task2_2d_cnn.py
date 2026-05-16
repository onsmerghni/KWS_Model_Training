"""
task2_2d_cnn.py
---------------
Task 2 — Upgrade: 2D CNN
The Concept: Treat the MFCC matrix as a grayscale 2D image where
  X-axis = time, Y-axis = frequency.
No flattening — Conv2D slides a small filter across both axes,
learning local time-frequency patterns.

Pipeline:
    MFCC (51, 13, 1)
        -> Conv2D(32) + BN + ReLU + MaxPool2D
        -> Conv2D(64) + BN + ReLU + MaxPool2D
        -> Flatten -> Dense(64) -> Dropout -> Softmax(2)

Run:
    python task2_2d_cnn.py
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
print("  TASK 2 — 2D CNN")
print("=" * 55)


# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/4] Loading & preprocessing data...")
(
    X_train, X_val,
    y_train, y_val,
    class_weight_dict,
    INPUT_SHAPE, NUM_CLASSES,
) = prepare_dataset()
# INPUT_SHAPE = (51, 13, 1)  — already has the channel dim, no extra reshape needed


# ── 2. Build model ────────────────────────────────────────────
print("\n[2/4] Building 2D CNN...")

def build_2d_cnn(input_shape, num_classes):
    """
    Standard 2D CNN treating the MFCC as a single-channel image.
    input_shape: (51, 13, 1)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="mfcc_input"),

        # ── Conv block 1 ──────────────────────────────────────
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.ReLU(name="relu1"),
        layers.MaxPooling2D((2, 2), name="pool1"),         # (25, 6, 32)
        layers.Dropout(0.2, name="drop1"),

        # ── Conv block 2 ──────────────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.ReLU(name="relu2"),
        layers.MaxPooling2D((2, 2), name="pool2"),         # (12, 3, 64)
        layers.Dropout(0.2, name="drop2"),

        # ── Classification head ───────────────────────────────
        layers.Flatten(name="flatten"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="drop3"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="2D_CNN")
    return model

cnn_model = build_2d_cnn(INPUT_SHAPE, NUM_CLASSES)
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
cnn_model.summary()


# ── 3. Train ──────────────────────────────────────────────────
print("\n[3/4] Training 2D CNN...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

t0 = time.time()
history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
time_per_epoch = (time.time() - t0) / len(history.history["loss"])

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train Acc", marker="o", linewidth=2)
axes[0].plot(history.history["val_accuracy"], label="Val Acc",   marker="s", linewidth=2)
axes[0].set_title("Task 2 — 2D CNN Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"],     label="Train Loss", marker="o", linewidth=2)
axes[1].plot(history.history["val_loss"], label="Val Loss",   marker="s", linewidth=2)
axes[1].set_title("Task 2 — 2D CNN Loss")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Task 2 — 2D CNN Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig("task2_training_curves.png", dpi=150)
plt.show()
print("  Saved: task2_training_curves.png")


# ── 4. Visualize learned filters (first conv layer) ───────────
print("\n[3b/4] Visualizing learned filters...")
filters = cnn_model.get_layer("conv1").get_weights()[0]   # (3, 3, 1, 32)
n_filters = min(16, filters.shape[-1])
fig, axes = plt.subplots(2, n_filters // 2, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    f = filters[:, :, 0, i]
    im = ax.imshow(f, cmap="viridis", aspect="auto")
    ax.set_title(f"F{i+1}", fontsize=8)
    ax.axis("off")
plt.suptitle("Task 2 — Learned Conv1 Filters (Time × Freq)", fontsize=12)
plt.tight_layout()
plt.savefig("task2_filters.png", dpi=150)
plt.show()
print("  Saved: task2_filters.png")


# ── 5. Export to TFLite ───────────────────────────────────────
print("\n[4/4] Exporting to TFLite (int8)...")

def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path  = "kws_task2_2d_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"  Saved: {tflite_path}  ({os.path.getsize(tflite_path)/1024:.2f} KB)")


# ── Results summary ───────────────────────────────────────────
best_val_acc = max(history.history["val_accuracy"])

print("\n" + "=" * 55)
print("  TASK 2 RESULTS")
print("=" * 55)
print(f"  Architecture     : 2D CNN")
print(f"  Total Parameters : {cnn_model.count_params():,}")
print(f"  Val Accuracy     : {best_val_acc*100:.2f}%")
print(f"  Avg Time / Epoch : {time_per_epoch:.1f} s")
print(f"  TFLite Size      : {os.path.getsize(tflite_path)/1024:.2f} KB")
print("=" * 55)
print("\nWhy 2D CNN is better than MLP:")
print("  - Preserves time-frequency structure (no flattening)")
print("  - Weight sharing: same filter scans the whole MFCC map")
print("  - Translation invariance via pooling")
print("\n→ Next: Task 3 — treat MFCC as a sequence with LSTM/GRU")
