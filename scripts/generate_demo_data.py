"""Generate the SYNTHETIC demo dataset (fictional games and reviews)."""
import json

import _bootstrap  # noqa: F401

from src.config import get_settings
from src.ingestion.demo_generator import write

if __name__ == "__main__":
    print(json.dumps(write(get_settings().raw_dir), indent=2))
