# Renamed from config.py to avoid mypy duplicate module name (config vs src.config)
# Original content preserved by moving the file; if there were additional edits,
# they should be replicated here. For now we import original module contents.
from .config import *  # type: ignore  # noqa: F401,F403
