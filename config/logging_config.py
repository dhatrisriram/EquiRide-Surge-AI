"""
Logging Configuration for Data Pipeline
"""
import logging
import os
import sys
from datetime import datetime

def setup_logging(log_file='logs/pipeline.log', log_level='INFO'):
    """
    Setup logging configuration with both console and file handlers
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        logger: Configured logger instance
    """
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('Member3Pipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

def log_stage(logger, stage_name, status, **kwargs):
    """
    Log pipeline stage with structured information
    
    Args:
        logger: Logger instance
        stage_name: Name of the pipeline stage
        status: Status (START, SUCCESS, FAILURE)
        **kwargs: Additional metadata to log
    """
    timestamp = datetime.now().isoformat()
    
    msg = f"[{stage_name}] {status}"
    if kwargs:
        details = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        msg += f" | {details}"
    
    if status == 'SUCCESS':
        logger.info(msg)
    elif status == 'FAILURE':
        logger.error(msg)
    else:  # START or other
        logger.info(msg)