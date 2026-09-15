from pathlib import Path

# ============================================================

# PROJECT PATHS

# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CODE_DIR = PROJECT_ROOT / "code"
DATASET_DIR = PROJECT_ROOT / "dataset"

DISEASE_DATASET_DIR = DATASET_DIR / "disease"
DISEASE_ORIGINAL_DIR = DISEASE_DATASET_DIR / "original"

DISEASE_COLOR_DIR = DISEASE_ORIGINAL_DIR / "color"
DISEASE_GRAYSCALE_DIR = DISEASE_ORIGINAL_DIR / "grayscale"
DISEASE_SEGMENTED_DIR = DISEASE_ORIGINAL_DIR / "segmented"

DISEASE_TRAIN_DIR = DISEASE_DATASET_DIR / "train"
DISEASE_VALIDATION_DIR = DISEASE_DATASET_DIR / "validation"
DISEASE_TEST_DIR = DISEASE_DATASET_DIR / "test"

FERTILIZER_DATASET_DIR = DATASET_DIR / "fertilizer"
FERTILIZER_CSV = FERTILIZER_DATASET_DIR / "Fertilizer Prediction.csv"
FERTILIZER_DATA_PATH = FERTILIZER_CSV

MODELS_DIR = CODE_DIR / "models"

DISEASE_MODEL_PATH = MODELS_DIR / "disease_model.keras"
FERTILIZER_MODEL_PATH = MODELS_DIR / "fertilizer_model.pkl"
FERTILIZER_ENCODER_PATH = MODELS_DIR / "fertilizer_encoder.pkl"

DISEASE_CLASSES_PATH = MODELS_DIR / "disease_classes.txt"

# ============================================================

# DISEASE MODEL SETTINGS

# ============================================================

IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

BATCH_SIZE = 64

DISEASE_EPOCHS = 10

RANDOM_SEED = 42

NUM_DISEASE_CLASSES = 38

# ============================================================

# DATASET SETTINGS

# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
".jpg",
".jpeg",
".png",
".bmp",
".gif",
".webp",
}

# ============================================================

# FERTILIZER MODEL SETTINGS

# ============================================================

FERTILIZER_TARGET_COLUMN = "Fertilizer Name"

FERTILIZER_TEST_SIZE = 0.20

# ============================================================

# APPLICATION SETTINGS

# ============================================================

APP_TITLE = "Crop Disease & Fertilizer Recommendation System"

APP_ICON = "🌱"
MAX_IMAGE_SIZE_MB = 10

# ============================================================

# CREATE REQUIRED DIRECTORIES

# ============================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DISEASE_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
