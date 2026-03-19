
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    box: Dict[str, float]

@dataclass(frozen=True)
class FrameTag:
    tag: str
    box: Dict[str, float]

@dataclass(frozen=True)
class Tag:
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: str = ""
    additional_info: Optional[Dict] = None
    frame_info: Optional[FrameInfo] = None
