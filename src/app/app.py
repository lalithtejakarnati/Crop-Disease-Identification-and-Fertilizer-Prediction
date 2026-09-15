"""
Main Flask application for Crop AI Assistant.
"""

import json
import os
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request
from PIL import Image

from src.app.disease_fertilizer_recommender import (
    DiseaseFertilizerRecommender,
)
from src.app.disease_predictor import DiseasePredictor
from src.app.fertilizer_predictor import FertilizerPredictor


app = Flask(
    __name__,
    template_folder="../../frontend/templates",
    static_folder="../../frontend/static",
)


print("Loading AI models...")

disease_predictor = DiseasePredictor()
fertilizer_predictor = FertilizerPredictor()

disease_fertilizer_recommender = DiseaseFertilizerRecommender(
    fertilizer_predictor=fertilizer_predictor,
)


DISEASE_INFO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config",
    "disease_info",
    "disease_info.json",
)

with open(
    DISEASE_INFO_PATH,
    "r",
    encoding="utf-8",
) as file:
    disease_info = json.load(file)


FERTILIZER_INFO_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "config",
    "fertilizer_info",
    "fertilizer_info.json",
)

with open(
    FERTILIZER_INFO_PATH,
    "r",
    encoding="utf-8",
) as file:
    fertilizer_info = json.load(file)


CROP_BY_DISEASE_PREFIX = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Maize",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Bell Pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawberry",
    "Tomato": "Tomato",
}


def crop_from_disease_class(disease_class):
    """Return the fertilizer model crop name for a disease class."""
    prefix = str(disease_class).split("___", 1)[0].strip()

    crop_type = CROP_BY_DISEASE_PREFIX.get(prefix)

    if crop_type is None:
        raise ValueError(
            f"Unable to determine crop from disease class: "
            f"{disease_class}"
        )

    return crop_type


print("AI models loaded successfully.")
print(
    f"Loaded disease information for "
    f"{len(disease_info)} classes."
)
print(
    f"Loaded fertilizer information for "
    f"{len(fertilizer_info)} fertilizers."
)


@app.route("/")
def home():
    """Render the Crop AI web application."""
    return render_template("index.html")


@app.route(
    "/api/predict-disease",
    methods=["POST"],
)
def predict_disease():
    """Predict crop disease from an uploaded leaf image."""

    try:
        if "image" not in request.files:
            return jsonify(
                {
                    "success": False,
                    "error": "No image uploaded.",
                }
            ), 400

        uploaded_file = request.files["image"]

        if uploaded_file.filename == "":
            return jsonify(
                {
                    "success": False,
                    "error": "No image selected.",
                }
            ), 400

        image = Image.open(uploaded_file)

        result = disease_predictor.predict(image)

        prediction_class = result.get("prediction")

        info = disease_info.get(
            prediction_class,
            {},
        )

        return jsonify(
            {
                "success": True,
                **result,
                "disease_info": info,
            }
        )

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


@app.route(
    "/api/predict-fertilizer",
    methods=["POST"],
)
def predict_fertilizer_api():
    """Predict fertilizer from soil and environmental inputs."""

    try:
        data = request.get_json()

        if not data:
            return jsonify(
                {
                    "success": False,
                    "error": "No input data received.",
                }
            ), 400

        prediction = fertilizer_predictor.predict(
            temperature=float(data["temperature"]),
            humidity=float(data["humidity"]),
            moisture=float(data["moisture"]),
            soil_type=data["soil_type"],
            crop_type=data["crop_type"],
            nitrogen=float(data["nitrogen"]),
            potassium=float(data["potassium"]),
            phosphorous=float(data["phosphorous"]),
        )

        info = fertilizer_info.get(
            prediction,
            {},
        )

        return jsonify(
            {
                "success": True,
                "fertilizer": prediction,
                "fertilizer_info": info,
            }
        )

    except KeyError as error:
        return jsonify(
            {
                "success": False,
                "error": f"Missing input field: {error}",
            }
        ), 400

    except ValueError as error:
        return jsonify(
            {
                "success": False,
                "error": f"Invalid input value: {error}",
            }
        ), 400

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


@app.route(
    "/api/predict-disease-and-fertilizer",
    methods=["POST"],
)
def predict_disease_and_fertilizer():
    """
    Run the complete Crop AI pipeline.

    Workflow:

        Leaf image
            ↓
        Disease CNN
            ↓
        Disease class + confidence
            ↓
        Disease-aware fertilizer recommender
            ↓
        Final fertilizer recommendation
    """

    try:
        if "image" not in request.files:
            return jsonify(
                {
                    "success": False,
                    "error": "No image uploaded.",
                }
            ), 400

        uploaded_file = request.files["image"]

        if uploaded_file.filename == "":
            return jsonify(
                {
                    "success": False,
                    "error": "No image selected.",
                }
            ), 400

        image = Image.open(uploaded_file)

        disease_result = disease_predictor.predict(
            image
        )

        disease_class = disease_result.get(
            "prediction"
        )

        if not disease_class:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Disease prediction did not "
                        "return a disease class."
                    ),
                }
            ), 500

        crop_type = crop_from_disease_class(
            disease_class
        )

        recommendation = (
            disease_fertilizer_recommender.recommend_automatic(
                disease_class=disease_class,
                crop_type=crop_type,
            )
        )

        predicted_fertilizer = recommendation[
            "fertilizer"
        ]

        return jsonify(
            {
                "success": True,
                "disease": {
                    "prediction": disease_result[
                        "prediction"
                    ],
                    "confidence": disease_result[
                        "confidence"
                    ],
                    "probabilities": disease_result[
                        "probabilities"
                    ],
                    "disease_info": disease_info.get(
                        disease_class,
                        {},
                    ),
                },
                "fertilizer": {
                    **recommendation,
                    "fertilizer_info": fertilizer_info.get(
                        predicted_fertilizer,
                        {},
                    ),
                },
            }
        )

    except KeyError as error:
        return jsonify(
            {
                "success": False,
                "error": f"Missing input field: {error}",
            }
        ), 400

    except ValueError as error:
        return jsonify(
            {
                "success": False,
                "error": f"Invalid input value: {error}",
            }
        ), 400

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500


@app.route("/api/health")
def health_check():
    """Return application and model health status."""

    return jsonify(
        {
            "status": "ok",
            "disease_model": True,
            "fertilizer_model": True,
            "disease_fertilizer_recommender": True,
        }
    )


if __name__ == "__main__":
    if os.environ.get(
        "WERKZEUG_RUN_MAIN"
    ) == "true":
        threading.Timer(
            1.0,
            lambda: webbrowser.open(
                "http://127.0.0.1:5001/"
            ),
        ).start()

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
    )
