import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"   # silence TF info/warning logs

"""
task1_baseline_mlp.py
---------------------
Task 1 — Baseline MLP
The Concept: Flatten the 2D MFCC feature map (Time x Frequency)
into a 1D array and pass it through standard Dense layers.

Pipeline:
    Raw Audio -> MFCC (51, 13, 1) -> Flatten (663,) -> Dense(32) -> Dense(16) -> Softmax(2)

Run:
    python task1_baseline_mlp.py
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
print("  TASK 1 — BASELINE MLP")
print("=" * 55)

# ── 1. Load data ──────────────────────────────────────────────
print("\n[1/4] Loading & preprocessing data...")
(
    X_train, X_val,
    y_train, y_val,
    class_weight_dict,
    INPUT_SHAPE, NUM_CLASSES,
) = prepare_dataset()


# ── 2. Build model ────────────────────────────────────────────
print("\n[2/4] Building Baseline MLP...")

def build_baseline_mlp(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape, name="mfcc_input"),
        # *** MLP-specific step: FLATTEN the 2D MFCC to 1D ***
        layers.Flatten(name="flatten_mfcc"),              # (663,)
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dense(16, activation="relu", name="dense_2"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="Baseline_MLP")
    return model

mlp_model = build_baseline_mlp(INPUT_SHAPE, NUM_CLASSES)
mlp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
mlp_model.summary()


# ── 3. Train ──────────────────────────────────────────────────
print("\n[3/4] Training Baseline MLP...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5
    ),
]

t0 = time.time()
history = mlp_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
time_per_epoch = (time.time() - t0) / len(history.history["loss"])

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"],     label="Train Acc", marker="o")
axes[0].plot(history.history["val_accuracy"], label="Val Acc",   marker="s")
axes[0].set_title("Task 1 — MLP Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid()

axes[1].plot(history.history["loss"],     label="Train Loss", marker="o")
axes[1].plot(history.history["val_loss"], label="Val Loss",   marker="s")
axes[1].set_title("Task 1 — MLP Loss")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid()

plt.tight_layout()
plt.savefig("task1_training_curves.png", dpi=150)
plt.show()
print("  Saved: task1_training_curves.png")


# ── 4. Variant: MLP + Dropout ─────────────────────────────────
print("\n[3b/4] Training MLP + Dropout variant...")

def build_mlp_dropout(input_shape, num_classes):
    return models.Sequential([
        layers.Input(shape=input_shape, name="mfcc_input"),
        layers.Flatten(name="flatten"),
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="drop_1"),
        layers.Dense(16, activation="relu", name="dense_2"),
        layers.Dropout(0.3, name="drop_2"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="MLP_Dropout")

model_drop = build_mlp_dropout(INPUT_SHAPE, NUM_CLASSES)
model_drop.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_drop = model_drop.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        )
    ],
    verbose=1,
)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
axes2[0].plot(history.history["val_accuracy"],      label="Baseline MLP", marker="o", linewidth=2)
axes2[0].plot(history_drop.history["val_accuracy"], label="MLP + Dropout", marker="s", linewidth=2)
axes2[0].set_title("Validation Accuracy Comparison")
axes2[0].set_xlabel("Epoch"); axes2[0].legend(); axes2[0].grid(alpha=0.3)

axes2[1].plot(history.history["val_loss"],      label="Baseline MLP", marker="o", linewidth=2)
axes2[1].plot(history_drop.history["val_loss"], label="MLP + Dropout", marker="s", linewidth=2)
axes2[1].set_title("Validation Loss Comparison")
axes2[1].set_xlabel("Epoch"); axes2[1].legend(); axes2[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("task1_comparison.png", dpi=150)
plt.show()
print("  Saved: task1_comparison.png")


# ── 5. Export best model to TFLite ───────────────────────────
print("\n[4/4] Exporting to TFLite (int8)...")

def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(mlp_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path  = "kws_task1_baseline_mlp.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"  Saved: {tflite_path}  ({os.path.getsize(tflite_path)/1024:.2f} KB)")


# ── Results summary ───────────────────────────────────────────
best_val_acc = max(history.history["val_accuracy"])
best_drop_acc = max(history_drop.history["val_accuracy"])

print("\n" + "=" * 55)
print("  TASK 1 RESULTS")
print("=" * 55)
print(f"  Architecture      : Baseline MLP")
print(f"  Total Parameters  : {mlp_model.count_params():,}")
print(f"  Val Accuracy (MLP): {best_val_acc*100:.2f}%")
print(f"  Val Acc (Dropout) : {best_drop_acc*100:.2f}%")
print(f"  Avg Time / Epoch  : {time_per_epoch:.1f} s")
print(f"  TFLite Size       : {os.path.getsize(tflite_path)/1024:.2f} KB")
print("=" * 55)
print("\nWhy MLP struggles with audio:")
print("  - Flattening destroys the 2D time-frequency structure")
print("  - Shift-sensitive: same word at different time = different vector")
print("  - No weight sharing between positions")
print("\n→ Next: Task 2 — treat MFCC as 2D image with Conv2D layers")
