"""
data_loader.py
--------------
Shared data loading and MFCC preprocessing for all KWS tasks.
Used by task1, task2, task3, task4.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
N_MFCC       = 13
N_FFT        = 512
HOP_LENGTH   = 320
# Resulting MFCC shape after librosa: (51, 13, 1)  [Time × Freq × Channel]


def load_speech_commands_tfds(sample_rate=SAMPLE_RATE, max_duration=1.0):
    """
    Load Yes / No / Silence samples from Speech Commands v0.0.3 via TFDS.
    Returns raw audio arrays X (N, 16000) and binary labels y (N,).
      yes  → 1
      no / silence → 0
    """
    print("Downloading / loading Speech Commands v0.0.3 (cached after first run ~2 GB)...")
    ds, _ = tfds.load(
        "speech_commands:0.0.3",
        split="train",
        with_info=True,
        shuffle_files=True,
    )

    # Keep only silence(0), yes(2), no(3)
    def filter_yes_no_silence(example):
        label = example["label"]
        return tf.logical_or(
            tf.logical_or(label == 0, label == 2),
            label == 3,
        )

    def preprocess(example):
        audio = tf.cast(example["audio"], tf.float32) / 32768.0
        target = int(sample_rate * max_duration)
        audio = audio[:target]
        audio = tf.pad(audio, [[0, tf.maximum(target - tf.shape(audio)[0], 0)]])
        label_binary = tf.cond(
            example["label"] == 2,
            lambda: tf.constant(1, dtype=tf.int32),   # yes  → 1
            lambda: tf.constant(0, dtype=tf.int32),   # no/silence → 0
        )
        return audio, label_binary

    ds_proc = ds.filter(filter_yes_no_silence).map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE
    )

    X_list, y_list = [], []
    for audio, label in ds_proc.as_numpy_iterator():
        X_list.append(audio)
        y_list.append(label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def audio_to_mfcc(audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    """
    Convert a raw audio array to MFCC feature map.
    Output shape: (51, 13, 1)  — Time × Freq × Channel
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mfcc = mfcc.T                           # (frames, n_mfcc)
    return np.expand_dims(mfcc, axis=-1)    # (frames, n_mfcc, 1)


def prepare_dataset(test_size=0.2, random_state=42):
    """
    Full pipeline: load → split → extract MFCCs → normalize.
    Returns:
        X_train_mfcc, X_val_mfcc  : shape (N, 51, 13, 1)
        y_train, y_val            : shape (N,)
        class_weight_dict         : dict for imbalance handling
        INPUT_SHAPE               : (51, 13, 1)
        NUM_CLASSES               : 2
    """
    # 1. Raw audio
    X, y = load_speech_commands_tfds()
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    print(f"  Total samples : {len(y)}")
    print(f"  Yes (1)       : {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
    print(f"  No/Silence (0): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")

    # 2. Train / val split
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. MFCC extraction
    print("  Extracting MFCC features...")
    X_train_mfcc = np.array([audio_to_mfcc(a) for a in X_train_raw])
    X_val_mfcc   = np.array([audio_to_mfcc(a) for a in X_val_raw])

    # 4. Normalize
    scaler = StandardScaler()
    N_tr, T, F, C = X_train_mfcc.shape
    X_train_mfcc = scaler.fit_transform(
        X_train_mfcc.reshape(N_tr, -1)
    ).reshape(N_tr, T, F, C)
    N_v = X_val_mfcc.shape[0]
    X_val_mfcc = scaler.transform(
        X_val_mfcc.reshape(N_v, -1)
    ).reshape(N_v, T, F, C)

    print(f"  Train : {X_train_mfcc.shape} | Val : {X_val_mfcc.shape}")

    # 5. Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(cw))
    print(f"  Class weights : {class_weight_dict}")

    INPUT_SHAPE = X_train_mfcc.shape[1:]   # (51, 13, 1)
    NUM_CLASSES = 2

    return (
        X_train_mfcc, X_val_mfcc,
        y_train, y_val,
        class_weight_dict,
        INPUT_SHAPE, NUM_CLASSES,
    )


if __name__ == "__main__":
    # Quick sanity check
    X_train, X_val, y_train, y_val, cw, shape, nc = prepare_dataset()
    print(f"\nInput shape : {shape}")
    print(f"Num classes : {nc}")
