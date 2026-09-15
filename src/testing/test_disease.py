import tensorflow as tf
from pathlib import Path

from src.config.settings import (
    DISEASE_TEST_DIR,
    DISEASE_MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
)


print("=" * 60)
print("DISEASE MODEL TEST EVALUATION")
print("=" * 60)

print(f"Test dataset: {DISEASE_TEST_DIR}")
print(f"Model: {DISEASE_MODEL_PATH}")
print()

# ------------------------------------------------------------
# CHECK PATHS
# ------------------------------------------------------------

if not Path(DISEASE_TEST_DIR).exists():
    raise FileNotFoundError(
        f"Test dataset not found: {DISEASE_TEST_DIR}"
    )

if not Path(DISEASE_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Trained model not found: {DISEASE_MODEL_PATH}"
    )

# ------------------------------------------------------------
# LOAD TEST DATASET
# ------------------------------------------------------------

test_dataset = tf.keras.utils.image_dataset_from_directory(
    DISEASE_TEST_DIR,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

print(f"Test images: {test_dataset.cardinality().numpy() * BATCH_SIZE}")
print(f"Number of classes: {len(test_dataset.class_names)}")
print()

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(
    DISEASE_MODEL_PATH
)

print("Model loaded successfully.")
print()

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------

print("Evaluating on the separate test dataset...")
print()

results = model.evaluate(
    test_dataset,
    verbose=1,
)

loss = results[0]
accuracy = results[1]

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Test Loss:     {loss:.4f}")
print(f"Test Accuracy: {accuracy:.2%}")

print("=" * 60)
