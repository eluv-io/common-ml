from copy import deepcopy
import os
from typing import Literal

def nested_update(original: dict, updates: dict) -> dict:
    original = deepcopy(original)
    updates = deepcopy(updates)
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            nested_update(original[key], value)
        else:
            original[key] = value
    return original

def get_file_type(file_path: str) -> Literal["image", "video", "audio", "unknown"]:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".mpeg"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".wma", ".m4a"}

    # Get the file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Determine the file type
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    else:
        return "unknown"
    
def dict_to_str(d: dict) -> str:
    return '{' + ', '.join(f'"{k}": "{v}"' for k, v in d.items()) + '}'