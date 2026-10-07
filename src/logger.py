"""
Logging system for Music Analyzer V1.
Provides file logging in logs/ and formatted console output via Rich.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler

_LOGGER_INITIALIZED = False

def setup_logger(verbose: bool = False, log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Configures and returns the central logger.
    - Console handler uses RichHandler for pretty, readable output.
    - File handler logs detailed messages and tracebacks to logs/music_analyzer.log.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger("music_analyzer")

    if _LOGGER_INITIALIZED:
        if verbose:
            logger.setLevel(logging.DEBUG)
        return logger

    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Console Handler (Rich)
    console_level = logging.DEBUG if verbose else logging.INFO
    console_handler = RichHandler(
        level=console_level,
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    # File Handler
    if log_dir is None:
        log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "music_analyzer.log"

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s:%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Suppress noisy external loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    _LOGGER_INITIALIZED = True
    return logger

def get_logger() -> logging.Logger:
    """Returns the application logger."""
    return logging.getLogger("music_analyzer")
