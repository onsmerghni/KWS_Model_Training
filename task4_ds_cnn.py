"""
task4_ds_cnn.py
---------------
Task 4 — Upgrade: Depthwise Separable CNN (DS-CNN)
The Concept: The accuracy of a CNN but with ~10x fewer parameters.
Replace standard Conv2D with SeparableConv2D which splits convolution into:
  1. Depthwise conv  — applies one filter per input channel (spatial filtering)
  2. Pointwise conv  — 1x1 conv that mixes channels (channel mixing)

Standard Conv2D:   (3x3x1x32) =   288 params  per conv layer
SeparableConv2D:   (3x3x1) + (1x1x1x32) = 9 + 32 = 41 params  per layer
=> ~7x fewer parameters per layer

Pipeline:
    MFCC (51, 13, 1)
        -> SeparableConv2D(32) + BN + ReLU + MaxPool2D
        -> SeparableConv2D(64) + BN + ReLU + MaxPool2D
        -> SeparableConv2D(64) + BN + ReLU + GlobalAvgPool
        -> Dense(32) -> Dropout -> Softmax(2)

Run:
    python task4_ds_cnn.py
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
print("  TASK 4 — DEPTHWISE SEPARABLE CNN (DS-CNN)")
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


# ── 2. Build models ───────────────────────────────────────────
print("\n[2/4] Building DS-CNN and standard CNN for comparison...")

def build_ds_cnn(input_shape, num_classes):
    """
    DS-CNN using SeparableConv2D layers.
    Much lighter than standard Conv2D.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="mfcc_input"),

        # ── DS-Conv block 1 ───────────────────────────────────
        layers.SeparableConv2D(
            32, (3, 3), padding="same", use_bias=False, name="sep_conv1"
        ),
        layers.BatchNormalization(name="bn1"),
        layers.ReLU(name="relu1"),
        layers.MaxPooling2D((2, 2), name="pool1"),    # (25, 6, 32)
        layers.Dropout(0.2, name="drop1"),

        # ── DS-Conv block 2 ───────────────────────────────────
        layers.SeparableConv2D(
            64, (3, 3), padding="same", use_bias=False, name="sep_conv2"
        ),
        layers.BatchNormalization(name="bn2"),
        layers.ReLU(name="relu2"),
        layers.MaxPooling2D((2, 2), name="pool2"),    # (12, 3, 64)
        layers.Dropout(0.2, name="drop2"),

        # ── DS-Conv block 3 ───────────────────────────────────
        layers.SeparableConv2D(
            64, (3, 3), padding="same", use_bias=False, name="sep_conv3"
        ),
        layers.BatchNormalization(name="bn3"),
        layers.ReLU(name="relu3"),
        # GlobalAvgPooling instead of Flatten — keeps it tiny
        layers.GlobalAveragePooling2D(name="gap"),

        # ── Classification head ───────────────────────────────
        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="drop3"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="DS_CNN")
    return model


def build_standard_cnn(input_shape, num_classes):
    """Same architecture but with standard Conv2D — for fair comparison."""
    model = models.Sequential([
        layers.Input(shape=input_shape, name="mfcc_input"),

        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1"),
        layers.BatchNormalization(name="bn1"),
        layers.ReLU(name="relu1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(0.2, name="drop1"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2"),
        layers.BatchNormalization(name="bn2"),
        layers.ReLU(name="relu2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(0.2, name="drop2"),

        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv3"),
        layers.BatchNormalization(name="bn3"),
        layers.ReLU(name="relu3"),
        layers.GlobalAveragePooling2D(name="gap"),

        layers.Dense(32, activation="relu", name="dense_1"),
        layers.Dropout(0.3, name="drop3"),
        layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="Standard_CNN")
    return model


ds_cnn_model  = build_ds_cnn(INPUT_SHAPE, NUM_CLASSES)
std_cnn_model = build_standard_cnn(INPUT_SHAPE, NUM_CLASSES)

for m in [ds_cnn_model, std_cnn_model]:
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

print("\n── DS-CNN (SeparableConv2D) ──")
ds_cnn_model.summary()
print("\n── Standard CNN (Conv2D) ──")
std_cnn_model.summary()

ds_params  = ds_cnn_model.count_params()
std_params = std_cnn_model.count_params()
print(f"\nParam reduction: {std_params:,} → {ds_params:,}")
print(f"DS-CNN is {std_params/ds_params:.1f}x smaller than Standard CNN")


# ── 3. Train ──────────────────────────────────────────────────
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

print("\n[3/4] Training DS-CNN...")
t0 = time.time()
history_ds = ds_cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
ds_time_per_epoch = (time.time() - t0) / len(history_ds.history["loss"])

print("\n[3b/4] Training Standard CNN (for comparison)...")
t0 = time.time()
history_std = std_cnn_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)
std_time_per_epoch = (time.time() - t0) / len(history_std.history["loss"])


# Training curves comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(history_ds.history["val_accuracy"],  label="DS-CNN",        marker="o", linewidth=2)
axes[0].plot(history_std.history["val_accuracy"], label="Standard CNN",  marker="s", linewidth=2)
axes[0].set_title("Task 4 — DS-CNN vs Standard CNN: Val Accuracy")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(history_ds.history["val_loss"],  label="DS-CNN",       marker="o", linewidth=2)
axes[1].plot(history_std.history["val_loss"], label="Standard CNN", marker="s", linewidth=2)
axes[1].set_title("Task 4 — DS-CNN vs Standard CNN: Val Loss")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Task 4 — DS-CNN Training Curves", fontsize=13)
plt.tight_layout()
plt.savefig("task4_training_curves.png", dpi=150)
plt.show()
print("  Saved: task4_training_curves.png")

# Parameter vs Accuracy bar chart
fig2, ax = plt.subplots(figsize=(8, 5))
models_names = ["Standard CNN", "DS-CNN"]
params_k     = [std_params / 1000, ds_params / 1000]
accs         = [
    max(history_std.history["val_accuracy"]) * 100,
    max(history_ds.history["val_accuracy"])  * 100,
]
colors = ["#4ecdc4", "#ff6b6b"]
bars = ax.bar(models_names, params_k, color=colors, edgecolor="black", alpha=0.85)
for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"Val Acc: {acc:.1f}%",
        ha="center", fontsize=11, fontweight="bold"
    )
ax.set_ylabel("Parameters (thousands)")
ax.set_title("DS-CNN vs Standard CNN — Parameters vs Accuracy")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("task4_params_vs_accuracy.png", dpi=150)
plt.show()
print("  Saved: task4_params_vs_accuracy.png")


# ── 4. Export DS-CNN to TFLite ────────────────────────────────
print("\n[4/4] Exporting DS-CNN to TFLite (int8)...")

def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(ds_cnn_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.float32
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_path  = "kws_task4_ds_cnn.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"  Saved: {tflite_path}  ({os.path.getsize(tflite_path)/1024:.2f} KB)")


# ── Results summary ───────────────────────────────────────────
ds_acc  = max(history_ds.history["val_accuracy"])
std_acc = max(history_std.history["val_accuracy"])

print("\n" + "=" * 55)
print("  TASK 4 RESULTS")
print("=" * 55)
print(f"  DS-CNN (SeparableConv2D)")
print(f"    Parameters   : {ds_params:,}")
print(f"    Val Accuracy : {ds_acc*100:.2f}%")
print(f"    Time / Epoch : {ds_time_per_epoch:.1f} s")
print(f"    TFLite Size  : {os.path.getsize(tflite_path)/1024:.2f} KB")
print(f"  Standard CNN (Conv2D)")
print(f"    Parameters   : {std_params:,}")
print(f"    Val Accuracy : {std_acc*100:.2f}%")
print(f"    Time / Epoch : {std_time_per_epoch:.1f} s")
print(f"  Param reduction: {std_params/ds_params:.1f}x smaller")
print(f"  Accuracy delta : {(ds_acc - std_acc)*100:+.2f}%")
print("=" * 55)
print("\nWhy DS-CNN matters for embedded / edge devices:")
print("  - SeparableConv2D = depthwise + pointwise = far fewer MACs")
print("  - Smaller model fits in MCU flash (e.g. STM32 B-U585I)")
print("  - Comparable accuracy to full Conv2D")
print("\n→ Next: benchmark.py — compare all 4 architectures")
