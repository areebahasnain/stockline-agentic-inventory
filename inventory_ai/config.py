import os, logging
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("INV_DB", ROOT / "data" / "inventory.db"))
MODEL_PATH = ROOT / "models" / "forecaster.joblib"
SEED, HORIZON, HOLDOUT = 42, 28, 56
REVIEW_DAYS, SERVICE_LEVEL = 7, float(os.getenv("SERVICE_LEVEL", 0.95))
WEIGHTS = {"cost": 0.4, "lead": 0.3, "reliability": 0.3}   # supplier scoring
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
DB_PATH.parent.mkdir(parents=True, exist_ok=True); MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
