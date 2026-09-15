from pathlib import Path

from src.config.settings import (
    DISEASE_COLOR_DIR,
    DISEASE_GRAYSCALE_DIR,
    DISEASE_SEGMENTED_DIR,
    DISEASE_TRAIN_DIR,
    DISEASE_VALIDATION_DIR,
    DISEASE_TEST_DIR,
    FERTILIZER_CSV,
    SUPPORTED_IMAGE_EXTENSIONS,
    NUM_DISEASE_CLASSES,
)


def get_class_names(dataset_directory: Path) -> list[str]:
    """Return sorted class-folder names."""
    if not dataset_directory.exists():
        return []

    return sorted(
        folder.name
        for folder in dataset_directory.iterdir()
        if folder.is_dir()
    )


def count_images(directory: Path) -> int:
    """Count supported image files recursively."""
    if not directory.exists():
        return 0

    return sum(
        1
        for file_path in directory.rglob("*")
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def count_images_by_class(directory: Path) -> dict[str, int]:
    """Count images inside each class folder."""
    if not directory.exists():
        return {}

    counts = {}

    for class_directory in sorted(directory.iterdir()):
        if class_directory.is_dir():
            counts[class_directory.name] = count_images(class_directory)

    return counts


def inspect_original_dataset() -> None:
    """Inspect the original PlantVillage datasets."""

    print("\n" + "=" * 60)
    print("PLANTVILLAGE ORIGINAL DATASET")
    print("=" * 60)

    datasets = {
        "Color": DISEASE_COLOR_DIR,
        "Grayscale": DISEASE_GRAYSCALE_DIR,
        "Segmented": DISEASE_SEGMENTED_DIR,
    }

    for name, directory in datasets.items():
        classes = get_class_names(directory)
        image_count = count_images(directory)

        print(f"\n{name}")
        print("-" * 60)
        print(f"Path: {directory}")
        print(f"Classes: {len(classes)}")
        print(f"Images: {image_count}")


def inspect_split_dataset() -> None:
    """Inspect train, validation, and test datasets."""

    print("\n" + "=" * 60)
    print("DISEASE DATASET SPLITS")
    print("=" * 60)

    datasets = {
        "Train": DISEASE_TRAIN_DIR,
        "Validation": DISEASE_VALIDATION_DIR,
        "Test": DISEASE_TEST_DIR,
    }

    for name, directory in datasets.items():
        classes = get_class_names(directory)
        image_count = count_images(directory)

        print(f"\n{name}")
        print("-" * 60)
        print(f"Path: {directory}")
        print(f"Classes: {len(classes)}")
        print(f"Images: {image_count}")


def validate_disease_classes() -> bool:
    """Verify that the color dataset contains 38 classes."""

    classes = get_class_names(DISEASE_COLOR_DIR)

    print("\n" + "=" * 60)
    print("DISEASE CLASS VALIDATION")
    print("=" * 60)

    print(f"Expected classes: {NUM_DISEASE_CLASSES}")
    print(f"Found classes:    {len(classes)}")

    if len(classes) == NUM_DISEASE_CLASSES:
        print("Status: PASSED")
        return True

    print("Status: FAILED")
    return False


def validate_split_classes() -> bool:
    """Verify that train, validation, and test use the same classes."""

    train_classes = set(get_class_names(DISEASE_TRAIN_DIR))
    validation_classes = set(get_class_names(DISEASE_VALIDATION_DIR))
    test_classes = set(get_class_names(DISEASE_TEST_DIR))

    print("\n" + "=" * 60)
    print("SPLIT CLASS VALIDATION")
    print("=" * 60)

    if not train_classes:
        print("Train directory contains no class folders.")
        return False

    if not validation_classes:
        print("Validation directory contains no class folders.")
        return False

    if not test_classes:
        print("Test directory contains no class folders.")
        return False

    if train_classes == validation_classes == test_classes:
        print("Status: PASSED")
        print(f"All splits contain {len(train_classes)} classes.")
        return True

    print("Status: FAILED")
    return False


def inspect_fertilizer_dataset() -> None:
    """Inspect the fertilizer prediction CSV."""

    print("\n" + "=" * 60)
    print("FERTILIZER DATASET")
    print("=" * 60)

    print(f"Path: {FERTILIZER_CSV}")

    if not FERTILIZER_CSV.exists():
        print("Status: FILE NOT FOUND")
        return

    try:
        import pandas as pd

        dataframe = pd.read_csv(FERTILIZER_CSV)

        print(f"Rows: {len(dataframe)}")
        print(f"Columns: {len(dataframe.columns)}")

        print("\nColumns:")
        for column in dataframe.columns:
            print(f"  - {column}")

        missing_values = dataframe.isnull().sum().sum()
        print(f"\nMissing values: {missing_values}")

    except Exception as error:
        print(f"Could not read fertilizer dataset: {error}")


def print_class_distribution(directory: Path) -> None:
    """Print image counts for every disease class."""

    distribution = count_images_by_class(directory)

    if not distribution:
        print("No class folders found.")
        return

    print("\nClass distribution:")
    print("-" * 60)

    for class_name, image_count in distribution.items():
        print(f"{class_name}: {image_count}")


def main() -> None:
    """Run the complete dataset inspection."""

    inspect_original_dataset()
    inspect_split_dataset()
    validate_disease_classes()
    validate_split_classes()
    inspect_fertilizer_dataset()

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
