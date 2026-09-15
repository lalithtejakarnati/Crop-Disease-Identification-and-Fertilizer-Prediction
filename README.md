# Crop AI Assistant

Crop AI is an AI-powered crop disease identification and fertilizer recommendation application.

## Features

- Crop leaf disease identification using a MobileNetV2 CNN.
- 38 disease classes across 14 crops.
- Disease information lookup.
- Disease-specific nutrient guidance.
- Fertilizer recommendation using a machine-learning model.
- Disease-aware fertilizer recommendation.
- Combined disease and fertilizer prediction through Flask.
- Web-based frontend for image upload and prediction.

## Supported Crops

1. Apple
2. Bell Pepper
3. Blueberry
4. Cherry
5. Grape
6. Maize
7. Orange
8. Peach
9. Potato
10. Raspberry
11. Soybean
12. Squash
13. Strawberry
14. Tomato

## Disease Prediction

The disease prediction pipeline uses a MobileNetV2 convolutional neural network trained for 38 disease classes.

The model achieved approximately 96% validation and independent test accuracy during evaluation.

## Fertilizer Recommendation

The fertilizer recommendation model uses:

- Temperature
- Humidity
- Moisture
- Soil Type
- Crop Type
- Nitrogen
- Potassium
- Phosphorous

The production fertilizer dataset contains 350 reference scenarios covering all 14 supported crops.

The current Random Forest model achieved approximately:

- 54.29% holdout test accuracy
- 55.14% mean 5-fold cross-validation accuracy
- 44.36% mean 5-fold macro F1

The fertilizer model is therefore treated as a recommendation-ranking component rather than an authoritative agronomic prescription system.

## Disease-Aware Recommendation

The recommendation flow is:

```text
Leaf Image
    |
    v
Disease CNN
    |
    v
Detected Disease + Crop
    |
    v
Disease Nutrient Guidance
    |
    v
Fertilizer ML Model
    |
    v
Disease-Aware Candidate Ranking
    |
    v
Final Fertilizer Recommendation