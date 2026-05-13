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

def train_yolo_model():
    """
    Trains a YOLO model using the YOLOv8 medium pre-trained weights and a custom dataset.

    This function performs the following steps:
    1. Loads the YOLOv8 medium pre-trained model.
    2. Trains the model using the custom dataset specified in the "config.yaml" file.
    3. Sets the training parameters including the number of epochs, image size, and batch size.

    Parameters:
    None

    Returns:
    results: Training results containing metrics and information about the training process.
    """
    # Load the YOLOv8 medium pre-trained model
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

# Call the function to train the YOLO model
train_yolo_model()