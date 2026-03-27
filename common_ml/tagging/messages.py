from typing import Literal, Optional, Dict
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

@dataclass
class Progress(Message):
    source_media: str

@dataclass
class Error(Message):
    message: str
    source_media: Optional[str] = None

@dataclass
class TagMessage(Message):
    type: Literal['tag']
    data: Tag

@dataclass
class ProgressMessage(Message):
    type: Literal['progress']
    data: Progress

@dataclass
class ErrorMessage(Message):
    type: Literal['error']
    data: Error