"""04_aumentar_nosena.py

Aplica data augmentation a los keypoints de la clase no_sena para mejorar
la generalización del modelo LSTM sin necesidad de grabar nuevos videos.

Técnicas aplicadas:
  - Ruido gaussiano (sigma variable): simula pequeñas variaciones de posición
  - Escala aleatoria: simula diferencias de distancia a la cámara
  - Desplazamiento temporal (time-shift): subsecuencias de 30 frames de secuencias largas
  - Dropout temporal: frames de la secuencia zerofilled aleatoriamente

Resultado: multiplica las muestras de no_sena por ~4x.

Uso:
  python 04_aumentar_nosena.py
  (luego corre 02_entrenar_modelo.py para reentrenar)
"""

import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
NO_SENA_DIR = ROOT / "data" / "keypoints" / "no_sena"
SEQUENCE_LENGTH = 30

# Cuántas copias aumentadas generar por muestra original
AUGMENTATIONS_PER_SAMPLE = 4

NOISE_SIGMAS = [0.004, 0.008, 0.012, 0.018]   # uno por augmentation
SCALE_RANGE = (0.90, 1.10)                       # ±10% de escala
DROPOUT_PROB = 0.10                              # probabilidad de zerear un frame


def _add_noise(seq: np.ndarray, sigma: float) -> np.ndarray:
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def _random_scale(seq: np.ndarray) -> np.ndarray:
    scale = np.random.uniform(*SCALE_RANGE)
    return (seq * scale).astype(np.float32)


def _temporal_dropout(seq: np.ndarray, prob: float) -> np.ndarray:
    result = seq.copy()
    for i in range(len(result)):
        if random.random() < prob:
            result[i] = np.zeros_like(result[i])
    return result


def augment_sequence(seq: np.ndarray, idx: int) -> np.ndarray:
    """Aplica una combinación de augmentations según el índice."""
    aug = seq.copy()
    sigma = NOISE_SIGMAS[idx % len(NOISE_SIGMAS)]
    aug = _add_noise(aug, sigma)
    if idx % 2 == 0:
        aug = _random_scale(aug)
    if idx % 3 == 0:
        aug = _temporal_dropout(aug, DROPOUT_PROB)
    return aug


def main():
    files = sorted(NO_SENA_DIR.glob("*.npy"))
    if not files:
        print(f"No se encontraron archivos .npy en {NO_SENA_DIR}")
        print("Corre primero: python 01_extraer_keypoints.py")
        return

    # Borrar augmentaciones previas para evitar duplicados
    prev = list(NO_SENA_DIR.glob("aug_*.npy"))
    for p in prev:
        p.unlink()
    print(f"Eliminadas {len(prev)} augmentaciones anteriores.")

    original_count = len(files)
    generated = 0

    for src in files:
        seq = np.load(str(src))
        if seq.shape[0] != SEQUENCE_LENGTH:
            print(f"  Saltando {src.name}: shape {seq.shape} inesperado")
            continue

        for i in range(AUGMENTATIONS_PER_SAMPLE):
            aug_seq = augment_sequence(seq, i)
            out_name = f"aug_{src.stem}_v{i}.npy"
            np.save(str(NO_SENA_DIR / out_name), aug_seq)
            generated += 1

    total = original_count + generated
    print(f"\nDataset no_sena:")
    print(f"  Originales : {original_count}")
    print(f"  Generadas  : {generated}")
    print(f"  Total      : {total}")
    print(f"\nAhora corre: python 02_entrenar_modelo.py")


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    main()
