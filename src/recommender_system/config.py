from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../recommender_system
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"