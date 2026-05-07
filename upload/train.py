"""
train.py - Script de entrenamiento del modelo YOLO para detección de señas.

Este módulo entrena un modelo YOLOv11s de segmentación con un dataset
personalizado de lenguaje de señas. Detecta automáticamente si hay una
GPU disponible y ajusta los parámetros de entrenamiento en consecuencia.

Dependencias:
    - ultralytics: Framework YOLO para entrenamiento y predicción.
    - torch: Backend de deep learning para detección de GPU/CPU.
    - multiprocessing: Manejo seguro de procesos en Windows.

Uso:
    Ejecutar directamente desde la terminal::

        python train.py

    El entrenamiento guardará resultados en::

        HORUS/runs/segment/vocales-2/

Note:
    En Windows es necesario usar ``mp.set_start_method("spawn")`` para
    evitar errores de multiprocessing con PyTorch/CUDA.

Autor:
    Equipo HORUS - Semillero de Investigación.
"""

from ultralytics import YOLO
import torch
import multiprocessing as mp


def train_yolo_model():
    """Entrena un modelo YOLO11s-seg con un dataset personalizado de señas.

    El flujo de entrenamiento es el siguiente:

    1. Carga el modelo preentrenado ``yolo11s-seg.pt``.
    2. Detecta automáticamente si hay una GPU CUDA disponible.
    3. Configura los hiperparámetros de entrenamiento:
       - 100 épocas.
       - Tamaño de imagen 640×640.
       - Batch size de 16 (optimizado para GPUs con 4GB VRAM).
       - 2 workers de multiprocessing (seguro en Windows).
    4. Inicia el entrenamiento y guarda los pesos en el directorio del proyecto.

    Returns:
        ultralytics.engine.results.Results: Objeto con métricas y resultados
        del entrenamiento, incluyendo mAP, pérdidas y curvas de aprendizaje.

    Raises:
        FileNotFoundError: Si el archivo ``yolo11s-seg.pt`` o el dataset
            ``data.yaml`` no se encuentran en las rutas especificadas.

    Example:
        >>> results = train_yolo_model()
        >>> print(results.results_dict)

    Note:
        Las rutas del dataset y directorio de salida están configuradas
        de forma absoluta para la máquina de desarrollo. Modificar según
        sea necesario para otros entornos.
    """
    # Cargar modelo preentrenado
    model = YOLO("yolo11s-seg.pt")

    # Detectar GPU o CPU automáticamente
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"\n🔍 Dispositivo detectado: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # Configurar y ejecutar entrenamiento
    results = model.train(
        data=r"C:\Users\Asus\Desktop\U\Semillero\Dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=r"C:\Users\Asus\Desktop\U\Semillero\HORUS\runs\segment",
        name="vocales-2",
        pretrained=True
    )

    return results


# Punto de entrada principal (necesario en Windows para multiprocessing)
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    train_yolo_model()
