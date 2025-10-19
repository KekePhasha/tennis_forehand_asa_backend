from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
CLEAN_CORS = ["http://localhost:3000"]

# calibration defaults (used by pure_siamese)
TAU_DEFAULT = 1.0
SCALE_DEFAULT = 0.1
