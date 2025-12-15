"""Environment and configuration utilities."""
import os
from pathlib import Path
from typing import Dict, Any
import json
import yaml
from dotenv import load_dotenv


def load_env() -> None:
    """Load environment variables from .env file."""
    load_dotenv()


def get_api_key(key_name: str = "GOOGLE_API_KEY") -> str:
    """
    Get API key from environment variables.
    
    Args:
        key_name: Name of the environment variable
        
    Returns:
        API key value
        
    Raises:
        ValueError: If API key is not found
    """
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"{key_name} not found in environment variables. Please set it in .env file.")
    return api_key


def load_config(config_type: str = "paths") -> Dict[str, Any]:
    """
    Load configuration from config files.
    
    Args:
        config_type: Type of config to load ('paths', 'model', 'settings')
        
    Returns:
        Configuration dictionary
    """
    config_files = {
        "paths": "config/paths_config.json",
        "model": "config/model_config.json",
        "settings": "config/settings.yaml"
    }
    
    config_path = config_files.get(config_type)
    if not config_path:
        raise ValueError(f"Unknown config type: {config_type}")
    
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path}")


def get_paths() -> Dict[str, str]:
    """Get all configured paths."""
    return load_config("paths")


def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return load_config("model")


def get_settings() -> Dict[str, Any]:
    """Get pipeline settings."""
    return load_config("settings")
