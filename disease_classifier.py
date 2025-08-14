import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = './model-checkpoints/efficientNetB3_best.keras'


def _preprocess_leaf_tf(leaf_img, target_size=(224, 224)):
    img_resized = cv2.resize(leaf_img, target_size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_array = img_rgb.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


class DiseaseClassifier:
    def __init__(self, image_size=(224, 224)):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.image_size = image_size
        self.classes = [
            "Apple - Scab", "Apple - Black Rot", "Apple - Cedar Apple Rust", "Apple - Healthy",
            "Blueberry - Healthy", "Cherry - Powdery Mildew", "Cherry - Healthy",
            "Corn - Gray Leaf Spot", "Corn - Common Rust", "Corn - Northern Leaf Blight",
            "Corn - Healthy", "Grape - Black Rot", "Grape - Esca Black Measles", "Grape - Leaf Blight",
            "Grape - Healthy", "Orange - Haunglongbing Citrus Greening", "Peach - Bacterial Spot",
            "Peach - Healthy", "Bell Pepper - Bacterial Spot", "Bell Pepper - Healthy",
            "Potato - Early Blight", "Potato - Late Blight", "Potato - Healthy", "Raspberry - Healthy",
            "Soybean - Healthy", "Squash - Powdery Mildew", "Strawberry - Leaf Scorch", "Strawberry - Healthy",
            "Tomato - Bacterial Spot", "Tomato - Early Blight", "Tomato - Late Blight",
            "Tomato - Leaf Mold", "Tomato - Septoria Leaf Spot", "Tomato - Spider Mites",
            "Tomato - Target Spot", "Tomato - Yellow Leaf Curl Virus", "Tomato - Mosaic Virus",
            "Tomato - Healthy"
        ]

        # Warm-up: run a fake prediction once
        dummy_input = preprocess_input(np.random.randint(0, 256, size=(3, *self.image_size, 3), dtype=np.uint8))
        _ = self.model.predict(dummy_input, verbose=0)

    def classify(self, batch_images_bgr):
        """
        Classifies a batch of images.

        Args:
            batch_images_bgr: List or NumPy array of images in BGR format.

        Returns:
            List of tuples: (class_name, confidence) for each image.
        """
        preprocessed_batch = []
        for img_bgr in batch_images_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, self.image_size)
            img_preprocessed = preprocess_input(img_resized)
            preprocessed_batch.append(img_preprocessed)

        preprocessed_batch = np.array(preprocessed_batch, dtype=np.float32)

        preds = self.model.predict(preprocessed_batch, verbose=0)

        results = []
        for pred in preds:
            class_idx = np.argmax(pred)
            confidence = pred[class_idx]
            results.append((self.classes[class_idx], float(confidence)))

        return results
