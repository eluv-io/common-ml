from typing import Optional, Dict
from dataclasses import dataclass

class Message: ...

@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    box: Dict[str, float]

@dataclass(frozen=True)
class Tag(Message):
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: str = ""
    additional_info: Optional[Dict] = None
    frame_info: Optional[FrameInfo] = None

@dataclass(frozen=True)
class Progress(Message):
    source_media: str

@dataclass(frozen=True)
class Error(Message):
    message: str
    source_media: Optional[str] = None