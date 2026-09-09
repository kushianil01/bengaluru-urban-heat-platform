from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"

LANDSAT_RAW_DIR = RAW_DATA_DIR / "landsat"
WEATHER_RAW_DIR = RAW_DATA_DIR / "weather"
BOUNDARIES_RAW_DIR = RAW_DATA_DIR / "boundaries"

LANDSAT_PROCESSED_DIR = PROCESSED_DATA_DIR / "landsat"
SPATIAL_PROCESSED_DIR = PROCESSED_DATA_DIR / "spatial"
FEATURES_PROCESSED_DIR = PROCESSED_DATA_DIR / "features"

HEAT_POCKETS_OUTPUT_DIR = OUTPUT_DATA_DIR / "heat_pockets"
PERSISTENCE_OUTPUT_DIR = OUTPUT_DATA_DIR / "persistence"
PREDICTIONS_OUTPUT_DIR = OUTPUT_DATA_DIR / "predictions"
RISK_OUTPUT_DIR = OUTPUT_DATA_DIR / "risk"
REPORTS_OUTPUT_DIR = OUTPUT_DATA_DIR / "reports"

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# STUDY AREA
# ============================================================

STUDY_AREA_NAME = "Bengaluru"

# Current GBA ward geography
WARD_COUNT = 369


# ============================================================
# SATELLITE CONFIGURATION
# ============================================================

SATELLITES = ["Landsat 8", "Landsat 9"]

STUDY_START_DATE = "2016-01-01"
STUDY_END_DATE = "2026-09-02"

LANDSAT_PATH = 144
LANDSAT_ROW = 51

MAX_CLOUD_COVER = 20


# ============================================================
# HEAT-POCKET CONFIGURATION
# ============================================================

HEAT_PERCENTILE = 95

HEAT_CONNECTIVITY = 8

MIN_HEAT_POCKET_PIXELS = 20


# ============================================================
# MACHINE LEARNING
# ============================================================

RANDOM_STATE = 42