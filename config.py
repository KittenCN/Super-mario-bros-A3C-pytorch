# Compatibility shim to avoid mypy duplicate module naming.
# Re-export everything from src.config so running scripts as top-level finds same symbols.
from src.config import *  # noqa: F401,F403
