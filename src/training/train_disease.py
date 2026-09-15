import tensorflow as tf

from src.config.settings import (
    DISEASE_TRAIN_DIR,
    DISEASE_VALIDATION_DIR,
    DISEASE_MODEL_PATH,
    DISEASE_CLASSES_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    DISEASE_EPOCHS,
    RANDOM_SEED,
    NUM_DISEASE_CLASSES,
)


def create_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DISEASE_TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DISEASE_VALIDATION_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    return train_dataset, validation_dataset


def build_model():
    # Load the existing trained model instead of starting from scratch.
    model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH
    )

    # MobileNetV2 is the third layer in the saved model.
    base_model = model.layers[2]

    # Allow fine-tuning of the MobileNetV2 backbone.
    base_model.trainable = True

    # Keep the first 134 layers frozen.
    # Fine-tune only the final 20 layers.
    fine_tune_from = len(base_model.layers) - 20

    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    # Keep BatchNormalization layers frozen for stability.
    for layer in base_model.layers[fine_tune_from:]:
        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):
            layer.trainable = False

    # Use a very small learning rate for fine-tuning.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_class_names(class_names):
    with open(
        DISEASE_CLASSES_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        for class_name in class_names:
            file.write(class_name + "\n")


def train_model():
    print("=" * 60)
    print("DISEASE MODEL TRAINING")
    print("=" * 60)

    print("\nLoading datasets...")

    train_dataset, validation_dataset = create_datasets()

    class_names = train_dataset.class_names

    print(f"\nClasses detected: {len(class_names)}")

    if len(class_names) != NUM_DISEASE_CLASSES:
        raise ValueError(
            f"Expected {NUM_DISEASE_CLASSES} classes, "
            f"but found {len(class_names)}."
        )

    save_class_names(class_names)

    print("\nBuilding model...")

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            DISEASE_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting training...")
    print(f"Epochs: {DISEASE_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=DISEASE_EPOCHS,
        callbacks=callbacks,
    )

    model.save(DISEASE_MODEL_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print("Model saved to:")
    print(DISEASE_MODEL_PATH)

    print("\nBest validation accuracy:")
    print(max(history.history["val_accuracy"]))

    print("=" * 60)


if __name__ == "__main__":
    train_model()

