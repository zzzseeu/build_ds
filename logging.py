"""Project logging helper.

This module keeps compatibility with Python's standard ``logging`` module while
providing ``init_logger`` for project scripts.
"""

from __future__ import annotations

import importlib.util
import sys
import sysconfig
from pathlib import Path


# Load stdlib logging explicitly to avoid recursive self-import.
_STDLIB_LOGGING_PATH = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
_spec = importlib.util.spec_from_file_location("_stdlib_logging", _STDLIB_LOGGING_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load stdlib logging from {_STDLIB_LOGGING_PATH}")
_stdlog = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stdlog)


def init_logger(name: str, log_file: str | Path | None = None, level: int = 20):
    """Initialize and return a reusable logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_file : str | Path | None
        Optional log file path. If provided, logs are written to file and stdout.
    level : int
        Logging level. Default 20 (INFO).
    """
    logger = _stdlog.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = _stdlog.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = _stdlog.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = _stdlog.FileHandler(str(log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def __getattr__(name: str):
    """Proxy attributes to stdlib logging for compatibility."""
    return getattr(_stdlog, name)
