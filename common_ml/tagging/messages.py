from typing import Optional, Dict
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    box: Dict[str, float]

@dataclass(frozen=True)
class Tag:
    start_time: int
    end_time: int
    tag: str
    source_media: str
    track: str
    frame_info: Optional[FrameInfo]

@dataclass
class FrameTag:
    tag: str
    box: Dict[str, float]