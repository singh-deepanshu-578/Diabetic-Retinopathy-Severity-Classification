import sys
from pathlib import Path

application_root = Path(__file__).resolve().parent.parent / '.venv'
sys.path.insert(0, str(application_root))

from app import app
