from pathlib import Path
import random
import shutil

from src.config.settings import (
    DISEASE_COLOR_DIR,
    DISEASE_TRAIN_DIR,
    DISEASE_VALIDATION_DIR,
    DISEASE_TEST_DIR,
    RANDOM_SEED,
    SUPPORTED_IMAGE_EXTENSIONS,
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def get_image_files(directory: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )


def split_class_files(files: list[Path]):
    files = files.copy()
    random.shuffle(files)

    total = len(files)

    train_count = int(total * TRAIN_RATIO)
    validation_count = int(total * VALIDATION_RATIO)

    train_files = files[:train_count]

    validation_files = files[
        train_count:train_count + validation_count
    ]

    test_files = files[
        train_count + validation_count:
    ]

    return train_files, validation_files, test_files


def copy_files(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for source_file in files:
        shutil.copy2(
            source_file,
            destination / source_file.name
        )


def split_dataset() -> None:
    print("=" * 60)
    print("PLANTVILLAGE DATASET SPLITTING")
    print("=" * 60)

    if not DISEASE_COLOR_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DISEASE_COLOR_DIR}"
        )

    random.seed(RANDOM_SEED)

    class_directories = sorted(
        directory
        for directory in DISEASE_COLOR_DIR.iterdir()
        if directory.is_dir()
    )

    print(f"Classes found: {len(class_directories)}")
    print("Split: 70% train / 15% validation / 15% test")
    print(f"Random seed: {RANDOM_SEED}")
    print()

    total_train = 0
    total_validation = 0
    total_test = 0

    for class_directory in class_directories:
        class_name = class_directory.name
        image_files = get_image_files(class_directory)

        train_files, validation_files, test_files = split_class_files(
            image_files
        )

        copy_files(
            train_files,
            DISEASE_TRAIN_DIR / class_name
        )

        copy_files(
            validation_files,
            DISEASE_VALIDATION_DIR / class_name
        )

        copy_files(
            test_files,
            DISEASE_TEST_DIR / class_name
        )

        total_train += len(train_files)
        total_validation += len(validation_files)
        total_test += len(test_files)

        print(
            f"{class_name}: "
            f"{len(train_files)} train | "
            f"{len(validation_files)} validation | "
            f"{len(test_files)} test"
        )

    print()
    print("=" * 60)
    print("SPLIT COMPLETE")
    print("=" * 60)

    print(f"Train:      {total_train}")
    print(f"Validation: {total_validation}")
    print(f"Test:       {total_test}")

    print(
        f"Total:      "
        f"{total_train + total_validation + total_test}"
    )

    print("=" * 60)


if __name__ == "__main__":
    split_dataset()
