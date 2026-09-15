from pathlib import Path

import pytest

from src.app.disease_fertilizer_recommender import (
    DiseaseFertilizerRecommender,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISEASE_CLASSES_PATH = PROJECT_ROOT / "src" / "models" / "disease_classes.txt"


@pytest.fixture(scope="module")
def recommender():
    return DiseaseFertilizerRecommender()


def base_inputs():
    return {
        "crop_type": "Tomato",
        "temperature": 30,
        "humidity": 60,
        "moisture": 40,
        "soil_type": "Loamy",
        "nitrogen": 20,
        "potassium": 5,
        "phosphorous": 20,
    }


def test_validated_disease_guidance_is_applied(recommender):
    result = recommender.recommend(
        disease_class="Tomato___Early_blight",
        **base_inputs(),
    )

    assert result["evidence_status"] == "validated"
    assert result["crop_match"] is True
    assert result["disease_guidance_applied"] is True
    assert result["nutrient_guidance"]["potassium"] == "avoid_excess"


def test_validated_guidance_respects_potassium_constraint(recommender):
    result = recommender.recommend(
        disease_class="Tomato___Early_blight",
        **base_inputs(),
    )

    fertilizer_npk = result["fertilizer_npk"]

    potassium = int(fertilizer_npk.split("-")[2])

    assert potassium < 25


def test_limited_guidance_preserves_compatible_baseline(recommender):
    inputs = base_inputs()
    inputs["crop_type"] = "Squash"

    result = recommender.recommend(
        disease_class="Squash___Powdery_mildew",
        **inputs,
    )

    assert result["evidence_status"] == "limited"
    assert result["crop_match"] is True
    assert result["disease_guidance_applied"] is True
    assert result["fertilizer"] == result["baseline_fertilizer"]


def test_limited_powdery_mildew_does_not_select_excessive_n_fertilizer(
    recommender,
):
    inputs = base_inputs()
    inputs["crop_type"] = "Squash"

    result = recommender.recommend(
        disease_class="Squash___Powdery_mildew",
        **inputs,
    )

    nitrogen = int(result["fertilizer_npk"].split("-")[0])

    assert nitrogen < 25


def test_insufficient_evidence_uses_baseline(recommender):
    result = recommender.recommend(
        disease_class="Tomato___healthy",
        **base_inputs(),
    )

    assert result["evidence_status"] == "insufficient"
    assert result["disease_guidance_applied"] is False
    assert result["fertilizer"] == result["baseline_fertilizer"]


def test_crop_mismatch_does_not_apply_disease_guidance(recommender):
    result = recommender.recommend(
        disease_class="Tomato___Early_blight",
        crop_type="Potato",
        temperature=30,
        humidity=60,
        moisture=40,
        soil_type="Loamy",
        nitrogen=20,
        potassium=5,
        phosphorous=20,
    )

    assert result["crop_match"] is False
    assert result["disease_guidance_applied"] is False
    assert result["fertilizer"] == result["baseline_fertilizer"]


def test_all_cnn_disease_classes_have_guidance_entries(recommender):
    with DISEASE_CLASSES_PATH.open("r", encoding="utf-8") as file:
        disease_classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    assert len(disease_classes) == 38
    assert set(disease_classes) == set(recommender.guidance["diseases"])


def test_unknown_disease_class_is_rejected(recommender):
    with pytest.raises(KeyError):
        recommender.recommend(
            disease_class="Unknown___Disease",
            **base_inputs(),
        )


def test_recommendation_does_not_expose_model_probability_as_confidence(
    recommender,
):
    result = recommender.recommend(
        disease_class="Tomato___Early_blight",
        **base_inputs(),
    )

    assert "confidence" not in result
    assert "probability" not in result
    assert "probabilities" not in result