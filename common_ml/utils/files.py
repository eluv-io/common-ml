from typing import Literal
import os

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