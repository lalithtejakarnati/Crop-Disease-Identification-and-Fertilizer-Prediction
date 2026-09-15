"""
Fertilizer prediction logic for Crop AI Assistant.
"""

import joblib
import numpy as np

from src.config.settings import FERTILIZER_MODEL_PATH
from src.utils.fertilizer_utils import prepare_fertilizer_input


class FertilizerPredictor:
    """Load and run the trained fertilizer recommendation model."""

    def __init__(self):
        self.model = joblib.load(
            FERTILIZER_MODEL_PATH
        )

    def predict(
        self,
        temperature,
        humidity,
        moisture,
        soil_type,
        crop_type,
        nitrogen,
        potassium,
        phosphorous,
    ):
        """
        Predict a fertilizer from soil and environmental inputs.
        Preserves original signature and return type (str).
        """

        fertilizer_input = prepare_fertilizer_input(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

        prediction = self.model.predict(
            fertilizer_input
        )

        return str(prediction[0])

    def predict_proba_dict(
        self,
        temperature,
        humidity,
        moisture,
        soil_type,
        crop_type,
        nitrogen,
        potassium,
        phosphorous,
    ):
        """
        Return the probability distribution across all fertilizer classes.
        """
        fertilizer_input = prepare_fertilizer_input(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

        probs = self.model.predict_proba(fertilizer_input)[0]
        classes = self.model.classes_

        return {
            str(cls): float(prob)
            for cls, prob in zip(classes, probs)
        }

    def predict_top_k(
        self,
        temperature,
        humidity,
        moisture,
        soil_type,
        crop_type,
        nitrogen,
        potassium,
        phosphorous,
        k=3,
    ):
        """
        Return the top-k ranked fertilizer candidates with probabilities.
        """
        proba_dict = self.predict_proba_dict(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

        ranked = sorted(
            proba_dict.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            {"fertilizer": fert, "probability": prob}
            for fert, prob in ranked[:k]
        ]

    def predict_with_confidence(
        self,
        temperature,
        humidity,
        moisture,
        soil_type,
        crop_type,
        nitrogen,
        potassium,
        phosphorous,
        confidence_threshold=0.45,
        margin_threshold=0.15,
    ):
        """
        Generate a prediction alongside calibrated uncertainty metrics.
        Communicates when a recommendation is uncertain and requires soil testing.
        """
        top_k = self.predict_top_k(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
            k=3,
        )

        top_1 = top_k[0]
        top_2 = top_k[1] if len(top_k) > 1 else {"probability": 0.0}

        margin_gap = float(top_1["probability"] - top_2["probability"])
        is_confident = (
            top_1["probability"] >= confidence_threshold
            and margin_gap >= margin_threshold
        )

        uncertainty_status = (
            "confident"
            if is_confident
            else "uncertain_verify_with_soil_testing"
        )

        guidance_note = (
            "Recommendation supported by high model confidence."
            if is_confident
            else "Low-confidence recommendation — verify with soil testing/agronomic guidance."
        )

        return {
            "prediction": top_1["fertilizer"],
            "confidence": float(top_1["probability"]),
            "margin_gap": margin_gap,
            "uncertainty_status": uncertainty_status,
            "guidance_note": guidance_note,
            "top_candidates": top_k,
        }
