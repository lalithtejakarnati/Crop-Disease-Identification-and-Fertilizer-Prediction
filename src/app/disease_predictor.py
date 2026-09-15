"""
Disease prediction logic for Crop AI Assistant.
"""

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config.settings import (
    DISEASE_CLASSES_PATH,
    DISEASE_MODEL_PATH,
)


class DiseasePredictor:
    """Load and run the trained crop disease classification model."""

    def __init__(self):
        self.model = tf.keras.models.load_model(
            DISEASE_MODEL_PATH
        )

        with open(
            DISEASE_CLASSES_PATH,
            "r",
        ) as file:
            self.classes = [
                line.strip()
                for line in file
                if line.strip()
            ]

    def predict(self, image):
        """
        Predict the disease/class for a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
            Uploaded crop leaf image.

        Returns
        -------
        dict
            Prediction, confidence, and all class probabilities.
        """

        image = image.convert("RGB")
        image = image.resize((160, 160))

        image_array = np.array(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        # MobileNetV2 preprocessing is already included
        # in the trained Keras model.
        predictions = self.model.predict(
            image_array,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = self.classes[
            predicted_index
        ]

        confidence = float(
            predictions[predicted_index]
        )

        probabilities = []

        for index, class_name in enumerate(
            self.classes
        ):
            probabilities.append(
                {
                    "class": class_name,
                    "probability": round(
                        float(
                            predictions[index]
                        ) * 100,
                        2,
                    ),
                }
            )

        probabilities.sort(
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "prediction": predicted_class,
            "confidence": round(
                confidence * 100,
                2,
            ),
            "probabilities": probabilities,
        }
