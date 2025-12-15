"""File utility functions for the RAG system."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List
import shutil


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_text(text: str, filepath: str) -> None:
    """Save text to file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)


def load_text(filepath: str) -> str:
    """Load text from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def list_files(directory: str, extension: str = None) -> List[str]:
    """
    List all files in directory with optional extension filter.
    
    Args:
        directory: Directory path
        extension: File extension filter (e.g., '.pdf')
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    if not path.exists():
        return []
    
    if extension:
        return [str(f) for f in path.glob(f'*{extension}')]
    return [str(f) for f in path.iterdir() if f.is_file()]


def clear_directory(directory: str) -> None:
    """Clear all files in directory."""
    path = Path(directory)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def get_file_size(filepath: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(filepath)


def file_exists(filepath: str) -> bool:
    """Check if file exists."""
    return Path(filepath).exists()
