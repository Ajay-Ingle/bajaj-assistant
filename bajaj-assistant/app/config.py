"""Env vars, model names, and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CUSTOMER_CSV_PATH = RAW_DATA_DIR / "Customer_Data_.csv"
FACTSHEET_PDF_PATH = RAW_DATA_DIR / "Factsheet_July-2026.pdf"

MODELS_DIR = PROJECT_ROOT / "models"
CROSSSELL_MODEL_PATH = MODELS_DIR / "crosssell_model.joblib"
CROSSSELL_FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.joblib"
