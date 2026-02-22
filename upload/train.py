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

    # Train the model with custom data and specified training parameters
    results = model.train(data="/home/srm/Documents/hands-pose-estimation/own_dataset/signRobot.v5i.yolov11/data.yaml", epochs=10, imgsz=640, batch=14)

    return results

# Call the function to train the YOLO model
train_yolo_model()