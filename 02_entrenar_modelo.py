"""02_entrenar_modelo.py

Entrena un modelo LSTM para reconocer señas usando los keypoints extraídos.

Requiere haber corrido antes:
  python 01_extraer_keypoints.py

Salida:
  ai/sign_language/action.h5

Ejemplo:
  python 02_entrenar_modelo.py
"""

import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parent
KEYPOINTS_DIR = ROOT / "data" / "keypoints"
MODEL_DIR = ROOT / "ai" / "sign_language"
MODEL_FILE = MODEL_DIR / "action.h5"
LABELS_FILE = MODEL_DIR / "labels.json"

SEQUENCE_LENGTH = 30
BATCH_SIZE = 16
EPOCHS = 200
SEED = 42


def load_data():
    if not KEYPOINTS_DIR.exists():
        raise RuntimeError(f"No existe la carpeta de keypoints: {KEYPOINTS_DIR}. Ejecuta 01_extraer_keypoints.py primero.")

    labels = sorted([d.name for d in KEYPOINTS_DIR.iterdir() if d.is_dir()])
    if not labels:
        raise RuntimeError(f"No se encontraron clases en {KEYPOINTS_DIR}.")

    X, y = [], []
    for idx, label in enumerate(labels):
        folder = KEYPOINTS_DIR / label
        files = sorted(folder.glob("*.npy"))
        if not files:
            continue
        for f in files:
            seq = np.load(f)
            if seq.shape[0] != SEQUENCE_LENGTH:
                continue
            X.append(seq)
            y.append(idx)

    if not X:
        raise RuntimeError("No se encontro ninguna secuencia valida. Asegurate de haber ejecutado 01_extraer_keypoints.py y que haya .npy de 30 frames.")

    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y, num_classes=len(labels))
    return X, y, labels


def build_model(input_shape, n_classes):
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),       # tanh por defecto, estable
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1.0),  # gradient clipping
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    X, y, labels = load_data()
    print("Clases:", labels)
    print("X shape:", X.shape, "y shape:", y.shape)

    model = build_model((SEQUENCE_LENGTH, X.shape[-1]), n_classes=len(labels))
    model.summary()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar labels.json sincronizado con el modelo (fix A8)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print("labels.json guardado:", labels)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        str(MODEL_FILE), monitor="val_accuracy", save_best_only=True, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=30, restore_best_weights=True, verbose=1
    )

    model.fit(
        X,
        y,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stop],
        shuffle=True,
    )

    print("Entrenamiento completado. Modelo guardado en:", MODEL_FILE)


if __name__ == "__main__":
    main()
