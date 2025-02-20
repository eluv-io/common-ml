import os
import base64
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

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

def encode_path(path):
    base, ext = os.path.splitext(path)  # Separate filename and extension
    encoded_base = base64.urlsafe_b64encode(base.encode()).decode().rstrip("=")
    return f"{encoded_base}{ext}"

def decode_path(encoded):
    base, ext = os.path.splitext(encoded)  # Separate encoded part and extension
    padding = "=" * (-len(base) % 4)  # Ensure proper padding
    decoded_base = base64.urlsafe_b64decode(base + padding).decode()
    return f"{decoded_base}{ext}"