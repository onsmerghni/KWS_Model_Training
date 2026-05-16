"""
benchmark.py
------------
Final Benchmark — Train all 4 KWS architectures and produce
the comparison table required by the project contract.

Architecture    | Total Params | Time/Epoch | Val Accuracy
----------------|--------------|------------|-------------
Baseline MLP    |              |            |
2D CNN          |              |            |
LSTM            |              |            |
DS-CNN          |              |            |

Run:
    python benchmark.py
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

print("=" * 60)
print("  KWS BENCHMARK — All 4 Architectures")
print("=" * 60)


# ── 1. Shared data loading ────────────────────────────────────
print("\n[STEP 1] Loading & preprocessing data (shared)...")
(
    X_train, X_val,
    y_train, y_val,
    class_weight_dict,
    INPUT_SHAPE, NUM_CLASSES,
) = prepare_dataset()

# Sequence version for LSTM/GRU
X_train_seq = X_train.squeeze(-1)   # (N, 51, 13)
X_val_seq   = X_val.squeeze(-1)
SEQ_SHAPE   = X_train_seq.shape[1:] # (51, 13)

EPOCHS     = 20
BATCH_SIZE = 32
CALLBACKS  = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5
    ),
]

results = {}


# ─────────────────────────────────────────────────────────────
# TASK 1 — Baseline MLP
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("  [TASK 1] Baseline MLP")
print("-" * 55)

mlp = models.Sequential([
    layers.Input(shape=INPUT_SHAPE, name="mfcc_input"),
    layers.Flatten(name="flatten"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="Baseline_MLP")

mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
h_mlp = mlp.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=CALLBACKS, verbose=1,
)
t_mlp = (time.time() - t0) / len(h_mlp.history["loss"])

results["Baseline MLP"] = {
    "params"     : mlp.count_params(),
    "time_epoch" : round(t_mlp, 1),
    "val_acc"    : round(max(h_mlp.history["val_accuracy"]) * 100, 2),
    "history"    : h_mlp,
}
print(f"  Val Accuracy: {results['Baseline MLP']['val_acc']}%")


# ─────────────────────────────────────────────────────────────
# TASK 2 — 2D CNN
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("  [TASK 2] 2D CNN")
print("-" * 55)

cnn = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),
    layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(), layers.ReLU(),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),
    layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(), layers.ReLU(),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"), layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="2D_CNN")

cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
h_cnn = cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=CALLBACKS, verbose=1,
)
t_cnn = (time.time() - t0) / len(h_cnn.history["loss"])

results["2D CNN"] = {
    "params"     : cnn.count_params(),
    "time_epoch" : round(t_cnn, 1),
    "val_acc"    : round(max(h_cnn.history["val_accuracy"]) * 100, 2),
    "history"    : h_cnn,
}
print(f"  Val Accuracy: {results['2D CNN']['val_acc']}%")


# ─────────────────────────────────────────────────────────────
# TASK 3 — LSTM
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("  [TASK 3] LSTM")
print("-" * 55)

lstm = models.Sequential([
    layers.Input(shape=SEQ_SHAPE),
    layers.LSTM(64, return_sequences=True), layers.Dropout(0.3),
    layers.LSTM(32, return_sequences=False), layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="LSTM")

lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
h_lstm = lstm.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=CALLBACKS, verbose=1,
)
t_lstm = (time.time() - t0) / len(h_lstm.history["loss"])

results["LSTM"] = {
    "params"     : lstm.count_params(),
    "time_epoch" : round(t_lstm, 1),
    "val_acc"    : round(max(h_lstm.history["val_accuracy"]) * 100, 2),
    "history"    : h_lstm,
}
print(f"  Val Accuracy: {results['LSTM']['val_acc']}%")


# ─────────────────────────────────────────────────────────────
# TASK 4 — DS-CNN
# ─────────────────────────────────────────────────────────────
print("\n" + "-" * 55)
print("  [TASK 4] DS-CNN (SeparableConv2D)")
print("-" * 55)

ds_cnn = models.Sequential([
    layers.Input(shape=INPUT_SHAPE),
    layers.SeparableConv2D(32, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(), layers.ReLU(),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),
    layers.SeparableConv2D(64, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(), layers.ReLU(),
    layers.MaxPooling2D((2, 2)), layers.Dropout(0.2),
    layers.SeparableConv2D(64, (3, 3), padding="same", use_bias=False),
    layers.BatchNormalization(), layers.ReLU(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(32, activation="relu"), layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax"),
], name="DS_CNN")

ds_cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

t0 = time.time()
h_ds = ds_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    class_weight=class_weight_dict,
    callbacks=CALLBACKS, verbose=1,
)
t_ds = (time.time() - t0) / len(h_ds.history["loss"])

results["DS-CNN"] = {
    "params"     : ds_cnn.count_params(),
    "time_epoch" : round(t_ds, 1),
    "val_acc"    : round(max(h_ds.history["val_accuracy"]) * 100, 2),
    "history"    : h_ds,
}
print(f"  Val Accuracy: {results['DS-CNN']['val_acc']}%")


# ─────────────────────────────────────────────────────────────
# BENCHMARK TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL BENCHMARK TABLE")
print("=" * 60)
header = f"{'Architecture':<18} {'Total Params':>14} {'Time/Epoch (s)':>16} {'Val Accuracy':>14}"
print(header)
print("-" * 60)
for arch, r in results.items():
    print(
        f"{arch:<18} {r['params']:>14,} {r['time_epoch']:>16.1f} {r['val_acc']:>13.2f}%"
    )
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# VISUALIZATION — Benchmark charts
# ─────────────────────────────────────────────────────────────
arch_names = list(results.keys())
val_accs   = [r["val_acc"]    for r in results.values()]
params_k   = [r["params"]/1e3 for r in results.values()]
times      = [r["time_epoch"] for r in results.values()]
colors     = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Val Accuracy
bars0 = axes[0].bar(arch_names, val_accs, color=colors, edgecolor="black", alpha=0.85)
for b, v in zip(bars0, val_accs):
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Val Accuracy (%)"); axes[0].set_title("Validation Accuracy")
axes[0].set_ylim(0, 105); axes[0].grid(axis="y", alpha=0.3)

# 2. Parameters (K)
bars1 = axes[1].bar(arch_names, params_k, color=colors, edgecolor="black", alpha=0.85)
for b, v in zip(bars1, params_k):
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                 f"{v:.1f}K", ha="center", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Parameters (thousands)"); axes[1].set_title("Model Size (Parameters)")
axes[1].grid(axis="y", alpha=0.3)

# 3. Time per epoch
bars2 = axes[2].bar(arch_names, times, color=colors, edgecolor="black", alpha=0.85)
for b, v in zip(bars2, times):
    axes[2].text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                 f"{v:.1f}s", ha="center", fontsize=10, fontweight="bold")
axes[2].set_ylabel("Seconds"); axes[2].set_title("Training Time per Epoch")
axes[2].grid(axis="y", alpha=0.3)

plt.suptitle("KWS Benchmark — All 4 Architectures", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("benchmark_chart.png", dpi=150)
plt.show()
print("Saved: benchmark_chart.png")

# Learning curves — all on one plot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
for (arch, r), c in zip(results.items(), colors):
    axes2[0].plot(r["history"].history["val_accuracy"],
                  label=arch, color=c, linewidth=2)
    axes2[1].plot(r["history"].history["val_loss"],
                  label=arch, color=c, linewidth=2)
axes2[0].set_title("Validation Accuracy — All Models")
axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("Accuracy")
axes2[0].legend(); axes2[0].grid(alpha=0.3)
axes2[1].set_title("Validation Loss — All Models")
axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("Loss")
axes2[1].legend(); axes2[1].grid(alpha=0.3)
plt.suptitle("KWS — Learning Curves Comparison", fontsize=13)
plt.tight_layout()
plt.savefig("benchmark_learning_curves.png", dpi=150)
plt.show()
print("Saved: benchmark_learning_curves.png")

print("\nDone! All outputs saved.")
