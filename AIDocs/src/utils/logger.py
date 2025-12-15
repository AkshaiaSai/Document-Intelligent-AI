"""Logging utility for the RAG system."""
import logging
import os
from pathlib import Path
import yaml


def setup_logger(name: str, config_path: str = "config/settings.yaml") -> logging.Logger:
    """
    Set up a logger with configuration from settings.yaml.
    
    Args:
        name: Logger name
        config_path: Path to settings configuration file
        
    Returns:
        Configured logger instance
    """
    # Load logging configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_config = config.get('logging', {})
    except FileNotFoundError:
        log_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get('file', 'logs/rag_agent.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_config.get('level', 'INFO')))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_config.get('format'))
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_config.get('format'))
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
