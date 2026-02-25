#
# Type definitions
#

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass(frozen=True)
class FrameInfo:
    frame_idx: int
    box: Dict[str, float]
    confidence: Optional[float]

@dataclass(frozen=True)
class Tag:
    start_time: int
    end_time: int
    text: str
    source_media: str
    track: str
    frame_info: Optional[FrameInfo]

@dataclass(frozen=True)
class FrameTag:
    text: str
    box: Dict[str, float]
    confidence: Optional[float]