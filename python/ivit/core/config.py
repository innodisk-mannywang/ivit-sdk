"""
Configuration management for iVIT-SDK.
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Global configuration
_config = {
    "log_level": "info",
    "cache_dir": os.path.expanduser("~/.cache/ivit"),
    "default_device": "auto",
    "default_precision": "fp16",
    "num_threads": 0,  # 0 = auto
}


def set_log_level(level: str) -> None:
    """
    Set global log level.

    Args:
        level: Log level (debug, info, warning, error, critical)
    """
    level = level.lower()
    _config["log_level"] = level

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger("ivit").setLevel(log_level)

    logger.debug(f"Log level set to: {level}")


def set_cache_dir(path: str) -> None:
    """
    Set global cache directory.

    Args:
        path: Cache directory path
    """
    cache_dir = Path(path).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    _config["cache_dir"] = str(cache_dir)
    logger.info(f"Cache directory set to: {cache_dir}")


def set_default_device(device: str) -> None:
    """
    Set default inference device.

    Args:
        device: Device string (auto, cpu, cuda:0, etc.)
    """
    _config["default_device"] = device
    logger.info(f"Default device set to: {device}")


def set_default_precision(precision: str) -> None:
    """
    Set default inference precision.

    Args:
        precision: Precision (fp32, fp16, int8)
    """
    _config["default_precision"] = precision
    logger.info(f"Default precision set to: {precision}")


def get_config() -> Dict[str, Any]:
    """
    Get current configuration.

    Returns:
        Configuration dictionary
    """
    return _config.copy()


def get_cache_dir() -> Path:
    """Get cache directory path."""
    cache_dir = Path(_config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


# Initialize logging
logging.getLogger("ivit").setLevel(logging.INFO)
