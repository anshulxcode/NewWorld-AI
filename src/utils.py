"""
Utility functions for NetWorld-AI: Reproducibility, configuration management, and logging.
"""

import os
import random
import logging
from typing import Dict, Any, Optional
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Sets global random seed across Python, NumPy, and PyTorch for strict reproducibility.

    Args:
        seed: The integer random seed value (default: 42).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads configuration YAML file containing hyperparameters, feature names, and file paths.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If parsing of YAML fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def setup_logger(
    name: str = "networld_ai",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Sets up a standardized logger for NetWorld-AI modules with console and file output.

    Args:
        name: Name of the logger instance.
        log_file: Optional filepath to output log statements.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logging.Logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream Handler (Console)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File Handler (Optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
