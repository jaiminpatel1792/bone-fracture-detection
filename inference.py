import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE = 224
MODEL_PATH = "models/best_model_densenet.keras"

# Your class mapping (as you requested: fractured=1)
CLASS_NAMES = {0: "not fractured", 1: "fractured"}

DEFAULT_THRESHOLD = 0.50


def load_model():
    """Load the trained Keras model from disk."""
    return tf.keras.models.load_model(MODEL_PATH)


def prepare_image(pil_img: Image.Image) -> np.ndarray:
    """
    Prepare image for DenseNet model:
    - RGB
    - Resize to 224x224
    - Convert to float32
    - Apply DenseNet preprocess_input
    """
    img = pil_img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    arr = preprocess_input(arr)
    return arr


def predict_fracture(model, pil_img: Image.Image, threshold: float = DEFAULT_THRESHOLD):
    """
    Predict fracture.
    Returns:
    - prob_fractured: float
    - predicted_label: 0/1
    - predicted_name: str
    """
    x = prepare_image(pil_img)

    pred = model(x, training=False)
    if isinstance(pred, (list, tuple)):
        pred = pred[0]

    prob = float(pred.numpy().ravel()[0])
    label = 1 if prob >= threshold else 0
    return prob, label, CLASS_NAMES[label]
