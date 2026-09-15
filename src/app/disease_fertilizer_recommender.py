"""
Disease-aware fertilizer recommendation logic for Crop AI.

This module combines:

1. The existing soil/crop/environment fertilizer predictor.
2. Disease-specific nutrient guidance from the disease-fertilizer
   knowledge base.
3. The trained fertilizer model's probability distribution as an
   internal candidate-ranking signal.

Disease detection does not itself determine a fertilizer. Disease-specific
adjustments are applied only when the knowledge base contains sufficient
evidence for doing so.

Important:
- predict_proba() is used only for internal candidate ranking.
- It is NOT exposed as user-facing confidence.
- Explicit disease nutrient constraints can exclude conflicting
  fertilizer candidates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.app.fertilizer_predictor import FertilizerPredictor
from src.utils.fertilizer_utils import prepare_fertilizer_input


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GUIDANCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "config"
    / "disease_fertilizer"
    / "disease_nutrient_guidance.json"
)


class DiseaseFertilizerRecommender:
    """
    Disease-aware fertilizer recommendation engine.

    The trained fertilizer model remains the primary predictor.
    Disease-specific knowledge acts as a conservative constraint layer.
    """

    def __init__(
        self,
        fertilizer_predictor: FertilizerPredictor | None = None,
        guidance_path: Path | str = GUIDANCE_PATH,
    ) -> None:
        self.fertilizer_predictor = (
            fertilizer_predictor
            if fertilizer_predictor is not None
            else FertilizerPredictor()
        )

        self.guidance_path = Path(guidance_path)
        self.guidance = self._load_guidance()

        self.diseases: dict[str, dict[str, Any]] = self.guidance.get(
            "diseases",
            {}
        )

        self.fertilizers: dict[str, dict[str, Any]] = self.guidance.get(
            "fertilizers",
            {}
        )

        self.policy: dict[str, Any] = self.guidance.get(
            "policy",
            {}
        )

    def _load_guidance(self) -> dict[str, Any]:
        """Load and validate the disease-fertilizer knowledge base."""

        if not self.guidance_path.exists():
            raise FileNotFoundError(
                f"Disease nutrient guidance file not found: "
                f"{self.guidance_path}"
            )

        with self.guidance_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                "Disease nutrient guidance must contain a JSON object."
            )

        if "diseases" not in data:
            raise ValueError(
                "Disease nutrient guidance is missing the 'diseases' "
                "section."
            )

        if "fertilizers" not in data:
            raise ValueError(
                "Disease nutrient guidance is missing the 'fertilizers' "
                "section."
            )

        return data

    def _get_disease_guidance(
        self,
        disease_class: str,
    ) -> dict[str, Any]:
        """
        Return guidance for an exact CNN disease class.

        The disease class must match disease_classes.txt exactly.
        """

        if disease_class not in self.diseases:
            raise KeyError(
                f"Unknown disease class: {disease_class!r}"
            )

        return self.diseases[disease_class]

    def _fertilizer_npk(
        self,
        fertilizer_name: str,
    ) -> tuple[float, float, float]:
        """Return N, P, K percentages for a known fertilizer."""

        fertilizer = self.fertilizers.get(fertilizer_name)

        if fertilizer is None:
            raise KeyError(
                f"Unknown fertilizer in guidance database: "
                f"{fertilizer_name!r}"
            )

        return (
            float(fertilizer["nitrogen"]),
            float(fertilizer["phosphorous"]),
            float(fertilizer["potassium"]),
        )

    def _candidate_fertilizers(self) -> list[str]:
        """Return all fertilizer classes known to the system."""

        return list(self.fertilizers.keys())

    def _violates_guidance(
        self,
        fertilizer_name: str,
        nutrient_guidance: dict[str, Any],
    ) -> bool:
        """
        Determine whether a fertilizer conflicts with explicit
        disease-specific nutrient constraints.

        Only explicit 'avoid_excess' guidance creates a constraint.

        The current fertilizer products are treated as heavy contributors
        to a nutrient when that nutrient's product percentage is >= 25%.
        This is a project-level heuristic for candidate filtering, not a
        universal agronomic threshold.
        """

        nitrogen, phosphorous, potassium = self._fertilizer_npk(
            fertilizer_name
        )

        nutrient_values = {
            "nitrogen": nitrogen,
            "phosphorous": phosphorous,
            "potassium": potassium,
        }

        for nutrient, guidance_value in nutrient_guidance.items():
            if guidance_value != "avoid_excess":
                continue

            nutrient_value = nutrient_values.get(nutrient)

            if nutrient_value is None:
                continue

            if nutrient_value >= 25:
                return True

        return False

    def _prepare_model_input(
        self,
        temperature: float,
        humidity: float,
        moisture: float,
        soil_type: str,
        crop_type: str,
        nitrogen: float,
        potassium: float,
        phosphorous: float,
    ) -> pd.DataFrame:
        """Prepare exactly the same input format used by FertilizerPredictor."""

        return prepare_fertilizer_input(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

    def _get_model_scores(
        self,
        fertilizer_input: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Get internal Random Forest candidate scores.

        These scores are used only for ranking candidates. They must not
        be interpreted or returned as calibrated prediction confidence.
        """

        model = self.fertilizer_predictor.model

        if not hasattr(model, "predict_proba"):
            raise TypeError(
                "The serialized fertilizer model must provide "
                "predict_proba() for disease-aware candidate ranking."
            )

        if not hasattr(model, "classes_"):
            raise TypeError(
                "The serialized fertilizer model must expose classes_."
            )

        probabilities = model.predict_proba(fertilizer_input)

        if len(probabilities) != 1:
            raise ValueError(
                "Expected exactly one fertilizer input row."
            )

        row = probabilities[0]
        classes = model.classes_

        return {
            str(class_name): float(probability)
            for class_name, probability in zip(classes, row)
        }

    def _rank_candidates(
        self,
        model_scores: dict[str, float],
        nutrient_guidance: dict[str, Any],
    ) -> list[str]:
        """
        Rank candidates by model score after applying disease constraints.

        Compatible candidates are ranked first by their model score.

        Conflicting candidates remain at the end only as a diagnostic
        fallback. They are never selected while a compatible candidate
        exists.
        """

        candidates = self._candidate_fertilizers()

        scored_candidates = [
            (
                fertilizer,
                model_scores.get(fertilizer, 0.0),
            )
            for fertilizer in candidates
        ]

        compatible = [
            item
            for item in scored_candidates
            if not self._violates_guidance(
                item[0],
                nutrient_guidance,
            )
        ]

        incompatible = [
            item
            for item in scored_candidates
            if self._violates_guidance(
                item[0],
                nutrient_guidance,
            )
        ]

        compatible.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        incompatible.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            fertilizer
            for fertilizer, _ in compatible + incompatible
        ]

    def _build_explanation(
        self,
        disease_class: str,
        guidance: dict[str, Any],
        selected_fertilizer: str,
        baseline_fertilizer: str,
        crop_type: str,
    ) -> str:
        """Build a transparent recommendation explanation."""

        evidence_status = guidance.get(
            "evidence_status",
            "insufficient",
        )

        if evidence_status == "insufficient":
            return (
                "No disease-specific fertilizer adjustment is supported "
                "for this disease. The recommendation follows the "
                "soil-, crop-, nutrient-, and environment-based "
                "fertilizer model."
            )

        if evidence_status == "limited":
            if selected_fertilizer == baseline_fertilizer:
                return (
                    f"Disease guidance for {disease_class} is limited. "
                    f"The fertilizer model selected {baseline_fertilizer}, "
                    "and it does not conflict with the explicit disease "
                    "nutrient constraint."
                )

            return (
                f"Disease guidance for {disease_class} is limited. "
                f"The fertilizer model selected {baseline_fertilizer}, "
                f"but that candidate conflicts with the disease nutrient "
                f"constraint. {selected_fertilizer} was selected as the "
                "highest-scoring compatible candidate."
            )

        if evidence_status == "validated":
            if selected_fertilizer == baseline_fertilizer:
                return (
                    f"Validated disease-specific nutrient guidance was "
                    f"considered for {crop_type}. The fertilizer model's "
                    f"{baseline_fertilizer} recommendation does not "
                    "conflict with the explicit nutrient constraint."
                )

            return (
                f"Validated disease-specific nutrient guidance was "
                f"applied for {crop_type}. The fertilizer model selected "
                f"{baseline_fertilizer}, but that candidate conflicts "
                f"with the disease nutrient constraint. "
                f"{selected_fertilizer} was selected as the "
                "highest-scoring compatible candidate."
            )

        return (
            "Disease-aware nutrient guidance was considered together "
            "with the baseline fertilizer prediction."
        )

    def recommend(
        self,
        disease_class: str,
        crop_type: str,
        temperature: float,
        humidity: float,
        moisture: float,
        soil_type: str,
        nitrogen: float,
        potassium: float,
        phosphorous: float,
    ) -> dict[str, Any]:
        """
        Generate a disease-aware fertilizer recommendation.

        Returns a structured result containing the baseline prediction,
        final recommendation, disease evidence status, nutrient guidance,
        and explanation.

        Model probability scores are intentionally not returned as
        user-facing confidence.
        """

        guidance = self._get_disease_guidance(disease_class)

        guidance_crop = str(
            guidance.get("crop", "")
        ).strip()

        crop_matches = (
            not guidance_crop
            or guidance_crop.lower()
            == str(crop_type).strip().lower()
        )

        fertilizer_input = self._prepare_model_input(
            temperature=temperature,
            humidity=humidity,
            moisture=moisture,
            soil_type=soil_type,
            crop_type=crop_type,
            nitrogen=nitrogen,
            potassium=potassium,
            phosphorous=phosphorous,
        )

        baseline_fertilizer = str(
            self.fertilizer_predictor.predict(
                temperature=temperature,
                humidity=humidity,
                moisture=moisture,
                soil_type=soil_type,
                crop_type=crop_type,
                nitrogen=nitrogen,
                potassium=potassium,
                phosphorous=phosphorous,
            )
        )

        evidence_status = str(
            guidance.get(
                "evidence_status",
                "insufficient",
            )
        )

        nutrient_guidance = dict(
            guidance.get(
                "nutrient_guidance",
                {},
            )
        )

        model_scores = self._get_model_scores(
            fertilizer_input
        )

        if not crop_matches:
            selected_fertilizer = baseline_fertilizer
            applied_disease_guidance = False
            explanation = (
                "The detected disease belongs to a different crop than "
                "the supplied crop input. Disease-specific nutrient "
                "guidance was therefore not applied."
            )

        elif evidence_status == "insufficient":
            selected_fertilizer = baseline_fertilizer
            applied_disease_guidance = False
            explanation = self._build_explanation(
                disease_class=disease_class,
                guidance=guidance,
                selected_fertilizer=selected_fertilizer,
                baseline_fertilizer=baseline_fertilizer,
                crop_type=crop_type,
            )

        else:
            ranked_candidates = self._rank_candidates(
                model_scores=model_scores,
                nutrient_guidance=nutrient_guidance,
            )

            if not ranked_candidates:
                selected_fertilizer = baseline_fertilizer
            else:
                selected_fertilizer = ranked_candidates[0]

            applied_disease_guidance = (
                selected_fertilizer != baseline_fertilizer
                or any(
                    value == "avoid_excess"
                    for value in nutrient_guidance.values()
                )
            )

            explanation = self._build_explanation(
                disease_class=disease_class,
                guidance=guidance,
                selected_fertilizer=selected_fertilizer,
                baseline_fertilizer=baseline_fertilizer,
                crop_type=crop_type,
            )

        fertilizer_details = self.fertilizers.get(
            selected_fertilizer,
            {},
        )

        return {
            "fertilizer": selected_fertilizer,
            "baseline_fertilizer": baseline_fertilizer,
            "disease_class": disease_class,
            "crop": crop_type,
            "evidence_status": evidence_status,
            "crop_match": crop_matches,
            "disease_guidance_applied": applied_disease_guidance,
            "nutrient_guidance": nutrient_guidance,
            "fertilizer_npk": fertilizer_details.get("npk"),
            "explanation": explanation,
            "disease_is_not_fertilizer_cure": bool(
                self.policy.get(
                    "disease_is_not_fertilizer_cure",
                    True,
                )
            ),
        }
    def recommend_automatic(
        self,
        disease_class: str,
        crop_type: str,
    ) -> dict[str, Any]:
        """
        Generate an automatic fertilizer recommendation using only the
        detected disease and crop.

        This mode is intentionally separate from the soil/environment
        fertilizer predictor because those inputs are not available in
        the primary leaf-upload workflow.

        17-17-17 is used as the general-purpose baseline because it
        provides balanced N-P-K nutrition. Explicit disease nutrient
        constraints can replace it with the closest compatible fertilizer.
        """

        guidance = self._get_disease_guidance(disease_class)

        guidance_crop = str(
            guidance.get("crop", "")
        ).strip()

        crop_matches = (
            not guidance_crop
            or guidance_crop.lower()
            == str(crop_type).strip().lower()
        )

        evidence_status = str(
            guidance.get(
                "evidence_status",
                "insufficient",
            )
        )

        nutrient_guidance = dict(
            guidance.get(
                "nutrient_guidance",
                {},
            )
        )

        baseline_fertilizer = "17-17-17"

        if not crop_matches:
            selected_fertilizer = baseline_fertilizer
            applied_disease_guidance = False
            explanation = (
                "The detected disease does not match the detected crop "
                "guidance entry. A general balanced fertilizer baseline "
                "was therefore used."
            )
        else:
            compatible = [
                fertilizer
                for fertilizer in self._candidate_fertilizers()
                if not self._violates_guidance(
                    fertilizer,
                    nutrient_guidance,
                )
            ]

            if baseline_fertilizer in compatible:
                selected_fertilizer = baseline_fertilizer
            elif compatible:
                selected_fertilizer = min(
                    compatible,
                    key=lambda fertilizer: sum(
                        abs(
                            self._fertilizer_npk(fertilizer)[index]
                            - self._fertilizer_npk(
                                baseline_fertilizer
                            )[index]
                        )
                        for index in range(3)
                    ),
                )
            else:
                selected_fertilizer = baseline_fertilizer

            has_explicit_constraint = any(
                value == "avoid_excess"
                for value in nutrient_guidance.values()
            )

            applied_disease_guidance = (
                has_explicit_constraint
                and selected_fertilizer != baseline_fertilizer
            )

            if evidence_status == "insufficient":
                explanation = (
                    "No disease-specific fertilizer adjustment is "
                    "supported for this disease. A general balanced "
                    "17-17-17 fertilizer baseline was selected. "
                    "Exact fertilizer selection should be refined using "
                    "soil testing and crop nutrient requirements."
                )
            elif evidence_status == "limited":
                explanation = (
                    f"Disease nutrient guidance for {disease_class} "
                    "is limited. A general balanced fertilizer baseline "
                    "was selected while respecting the available "
                    "nutrient constraints. Soil testing should be used "
                    "to refine the fertilizer choice."
                )
            elif evidence_status == "validated":
                explanation = (
                    f"Validated nutrient guidance for {disease_class} "
                    "was considered. The automatic recommendation uses "
                    "a balanced fertilizer baseline while respecting "
                    "explicit nutrient constraints. Soil and crop "
                    "information should be used for a precise fertilizer "
                    "selection."
                )
            else:
                explanation = (
                    "Disease-aware nutrient guidance was considered "
                    "with the automatic balanced fertilizer baseline."
                )

        fertilizer_details = self.fertilizers.get(
            selected_fertilizer,
            {},
        )

        return {
            "fertilizer": selected_fertilizer,
            "baseline_fertilizer": baseline_fertilizer,
            "disease_class": disease_class,
            "crop": crop_type,
            "evidence_status": evidence_status,
            "crop_match": crop_matches,
            "disease_guidance_applied": applied_disease_guidance,
            "nutrient_guidance": nutrient_guidance,
            "fertilizer_npk": fertilizer_details.get("npk"),
            "explanation": explanation,
            "automatic": True,
            "disease_is_not_fertilizer_cure": bool(
                self.policy.get(
                    "disease_is_not_fertilizer_cure",
                    True,
                )
            ),
        }
