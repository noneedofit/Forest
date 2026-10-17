"""Paths and secrets, read from the environment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FOREST_DATA_DIR", ROOT / "data"))
DB_PATH = Path(os.environ.get("FOREST_DB_PATH", DATA_DIR / "forest.db"))
MODEL_PATH = Path(
    os.environ.get("FOREST_MODEL_PATH", "/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf")
)
# instruct model for the incident summaries (mounted from ~/shared/models on the VM)
SUMMARY_MODEL_PATH = Path(
    os.environ.get("FOREST_SUMMARY_MODEL_PATH", "/models/granite-3.3-2b-instruct-be.Q4_K_M.gguf")
)
HMAC_KEY = os.environ.get("FOREST_HMAC_KEY", "")
ENC_KEY = os.environ.get("FOREST_ENC_KEY", "")
