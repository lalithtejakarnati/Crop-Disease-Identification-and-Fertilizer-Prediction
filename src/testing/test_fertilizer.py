"""
Tests and evaluation for the fertilizer recommendation model.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from src.app.fertilizer_predictor import FertilizerPredictor
from src.config.settings import (
    FERTILIZER_DATA_PATH,
    FERTILIZER_MODEL_PATH,
)


@pytest.fixture(scope="module")
def fertilizer_data():
    data = pd.read_csv(FERTILIZER_DATA_PATH)
    data.columns = data.columns.str.strip()
    feature_columns = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
    ]
    target_column = "Fertilizer Name"
    X = data[feature_columns]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_model():
    return joblib.load(FERTILIZER_MODEL_PATH)


@pytest.fixture(scope="module")
def predictor():
    return FertilizerPredictor()


def test_fertilizer_model_loads(trained_model):
    assert trained_model is not None
    assert hasattr(trained_model, "predict")
    assert hasattr(trained_model, "predict_proba")
    assert hasattr(trained_model, "classes_")
    assert len(trained_model.classes_) == 7


def test_fertilizer_model_evaluation(trained_model, fertilizer_data):
    _, X_test, _, y_test = fertilizer_data
    predictions = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

    # Performance must exceed baseline standards
    assert accuracy >= 0.55
    assert macro_f1 >= 0.50


def test_fertilizer_top_k_evaluation(trained_model, fertilizer_data):
    _, X_test, _, y_test = fertilizer_data
    probs = trained_model.predict_proba(X_test)
    classes = trained_model.classes_

    top2_hits = [
        true in classes[np.argsort(prob)[-2:]]
        for prob, true in zip(probs, y_test.values)
    ]
    top3_hits = [
        true in classes[np.argsort(prob)[-3:]]
        for prob, true in zip(probs, y_test.values)
    ]

    top2_acc = np.mean(top2_hits)
    top3_acc = np.mean(top3_hits)

    assert top2_acc >= 0.80
    assert top3_acc >= 0.90


def test_fertilizer_predictor_api(predictor):
    prediction = predictor.predict(
        temperature=26.0,
        humidity=60.0,
        moisture=20.0,
        soil_type="Loamy",
        crop_type="Tomato",
        nitrogen=25.0,
        potassium=120.0,
        phosphorous=30.0,
    )
    assert isinstance(prediction, str)
    assert prediction in predictor.model.classes_


def test_fertilizer_predictor_confidence_and_top_k(predictor):
    result = predictor.predict_with_confidence(
        temperature=26.0,
        humidity=60.0,
        moisture=20.0,
        soil_type="Loamy",
        crop_type="Tomato",
        nitrogen=25.0,
        potassium=120.0,
        phosphorous=30.0,
    )

    assert "prediction" in result
    assert "confidence" in result
    assert "top_candidates" in result
    assert len(result["top_candidates"]) == 3
    assert result["confidence"] <= 1.0


def evaluate_cli():
    print("=" * 60)
    print("FERTILIZER MODEL TEST EVALUATION")
    print("=" * 60)
    print(f"Dataset: {FERTILIZER_DATA_PATH}")
    print(f"Model:   {FERTILIZER_MODEL_PATH}")
    print()

    data = pd.read_csv(FERTILIZER_DATA_PATH)
    data.columns = data.columns.str.strip()

    feature_columns = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
    ]
    target_column = "Fertilizer Name"

    X = data[feature_columns]
    y = data[target_column]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(FERTILIZER_MODEL_PATH)
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)
    classes = model.classes_

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)
    top2 = np.mean([
        true in classes[np.argsort(prob)[-2:]]
        for prob, true in zip(probs, y_test.values)
    ])
    top3 = np.mean([
        true in classes[np.argsort(prob)[-3:]]
        for prob, true in zip(probs, y_test.values)
    ])

    print(f"Test samples:          {len(X_test)}")
    print(f"Test Accuracy (Top-1): {accuracy:.2%}")
    print(f"Test Top-2 Accuracy:   {top2:.2%}")
    print(f"Test Top-3 Accuracy:   {top3:.2%}")
    print(f"Test Macro F1:         {macro_f1:.2%}")
    print()
    print("Classification Report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("=" * 60)


if __name__ == "__main__":
    evaluate_cli()