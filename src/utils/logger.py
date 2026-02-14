"""
Logging utilities for LLM Inference Benchmarking

Provides structured logging with multiple output formats and severity levels.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    structured: bool = False,
) -> None:
    """
    Configure the global logger with specified settings.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console: Whether to log to console
        structured: Whether to use JSON-formatted structured logging
    """
    # Remove default handler
    logger.remove()
    
    # Console handler
    if console:
        if structured:
            # JSON format for structured logging
            log_format = (
                "{{\"time\": \"{time:YYYY-MM-DD HH:mm:ss}\", "
                "\"level\": \"{level}\", "
                "\"module\": \"{name}\", "
                "\"function\": \"{function}\", "
                "\"line\": {line}, "
                "\"message\": \"{message}\"}}"
            )
        else:
            # Human-readable format
            log_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        logger.add(
            sys.stdout,
            format=log_format,
            level=log_level,
            colorize=not structured,
        )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if structured:
            file_format = (
                "{{\"time\": \"{time:YYYY-MM-DD HH:mm:ss}\", "
                "\"level\": \"{level}\", "
                "\"module\": \"{name}\", "
                "\"function\": \"{function}\", "
                "\"line\": {line}, "
                "\"message\": \"{message}\"}}"
            )
        else:
            file_format = (
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            )
        
        logger.add(
            log_file,
            format=file_format,
            level=log_level,
            rotation="100 MB",  # Rotate when file reaches 100 MB
            retention="30 days",  # Keep logs for 30 days
            compression="zip",  # Compress rotated logs
        )


def get_logger(name: str = __name__):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)


# Context manager for timing operations
class LogTimer:
    """
    Context manager for timing and logging operations.
    
    Example:
        with LogTimer("model_loading"):
            model = load_model()
    """
    
    def __init__(self, operation_name: str, log_level: str = "INFO"):
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None
        
    def __enter__(self):
        logger.log(self.log_level, f"Starting: {self.operation_name}")
        import time
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        if exc_type is None:
            logger.log(
                self.log_level,
                f"Completed: {self.operation_name} (took {duration:.2f}s)"
            )
        else:
            logger.error(
                f"Failed: {self.operation_name} (took {duration:.2f}s) - {exc_val}"
            )
        return False  # Don't suppress exceptions


# Convenience function for progress logging
def log_progress(current: int, total: int, prefix: str = "Progress", interval: int = 10):
    """
    Log progress at regular intervals.
    
    Args:
        current: Current item number
        total: Total number of items
        prefix: Prefix for log message
        interval: Log every N items
    """
    if current % interval == 0 or current == total:
        percent = (current / total) * 100
        logger.info(f"{prefix}: {current}/{total} ({percent:.1f}%)")